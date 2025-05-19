import argparse
import os
import json
import csv
from typing import Any, TypeAlias
from io import BytesIO
import time

import h5py
import numpy as np
from pydantic import BaseModel
import yaml
import pickle

from openai import OpenAI
from openai.lib._pydantic import to_strict_json_schema

from graspmolmo.utils import tqdm

TasksSpec: TypeAlias = dict[str, dict[str, dict[str, Any]]]


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("task_json")
    parser.add_argument("obs_dir")
    parser.add_argument("out_dir")
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--retrieve", nargs="?", help="Retrieve a batch job with the given id", const="")
    return parser.parse_args()

SYS_PROMPT = """
You are a linguistic and robotic expert. You are tasked with matching a candidate grasp description to one or more of multiple options, called annotated grasp descriptions.

You will be given a candidate grasp description, which is a description of how a robot could grasp a specific object.
You will also be given a list of annotated grasp descriptions, which are multiple known descriptions of how a robot could grasp the same object.
You should choose the annotated grasp descriptions that have the same meaning as the candidate grasp description.
In this case, "meaning" means that the candidate grasp description and the annotated grasp description describe a grasp on a similar part of the object, in a similar manner.

For example, if the candidate grasp description is "grasp the midpoint of the handle of the mug", and one of the annotated grasp descriptions is "grasp the handle of the mug", then you should choose that annotated grasp description.
If there are multiple annotated grasp descriptions that have the same meaning as the candidate grasp description, you should return all of them.
If there are no suitably matching annotated grasp descriptions, you should return an empty list.

You should output a JSON object with the following fields:
- candidate_grasp_desc: the candidate grasp description which you are prompted with
- matching_grasp_descs: a list of annotated grasp descriptions that have the same meaning as the candidate grasp description
""".strip()


def get_annotated_grasp_library(obs_dir: str, out_dir: str) -> dict[str, dict[str, list[str]]]:
    """
    Args:
        obs_dir: observations directory
        out_dir: output directory
    Returns:
        annotated_grasp_library: dictionary mapping object categories to object ids to list of grasp descriptions
    """
    cache_path = os.path.join(out_dir, "annotated_grasp_library.json")
    if os.path.exists(cache_path):
        with open(cache_path, "r") as f:
            library = json.load(f)
        return library
    
    grasp_dict: dict[str, dict[str, set[str]]] = {}  # category -> object_id -> set of grasp descriptions
    for fn in tqdm(os.listdir(obs_dir), desc="Loading annotated grasp descriptions"):
        if not fn.endswith(".hdf5"):
            continue

        with h5py.File(os.path.join(obs_dir, fn), "r") as f:
            for view_id in f.keys():
                for obs_id in f[view_id].keys():
                    if not obs_id.startswith("obs_"):
                        continue
                    annotation = yaml.safe_load(f[view_id][obs_id]["annot"][()])
                    obj_category = annotation["object_category"]
                    obj_id = annotation["object_id"]
                    grasp_desc = annotation["grasp_description"]
                    if obj_category not in grasp_dict:
                        grasp_dict[obj_category] = {}
                    if obj_id not in grasp_dict[obj_category]:
                        grasp_dict[obj_category][obj_id] = set()
                    grasp_dict[obj_category][obj_id].add(grasp_desc)

    for obj_category in grasp_dict.keys():
        for obj_id in grasp_dict[obj_category].keys():
            grasp_dict[obj_category][obj_id] = list(grasp_dict[obj_category][obj_id])

    with open(cache_path, "w") as f:
        json.dump(grasp_dict, f, indent=2)

    return grasp_dict

def get_candidate_grasp_library(tasks_spec_path: str, out_dir: str) -> dict[str, list[str]]:
    cache_path = os.path.join(out_dir, "candidate_grasp_library.pkl")
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    with open(tasks_spec_path, "r") as f:
        tasks_spec: TasksSpec = json.load(f)

    grasp_dict: dict[str, list[str]] = {}
    for category in tasks_spec.keys():
        grasp_dict[category] = []
        for grasp in tasks_spec[category]:
            grasp_dict[category].append(tasks_spec[category][grasp]["info"]["natural_language"])

    with open(cache_path, "wb") as f:
        pickle.dump(grasp_dict, f)

    return grasp_dict

class GraspMatch(BaseModel):
    candidate_grasp_desc: str
    matching_grasp_descs: list[str]

def create_query(object_category: str, object_id: str, candidate_grasp: str, annotated_grasps: list[str]):
    user_message = f"The object is a(n) {object_category}. The candidate grasp description is: \"{candidate_grasp}\". The annotated grasp descriptions are:"
    for grasp_desc in annotated_grasps:
        user_message += f"\n- {grasp_desc}"
    request = {
        "custom_id": f"{object_category}_{object_id}___{candidate_grasp}",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": "gpt-4o",
            "messages": [
                {"role": "system", "content": SYS_PROMPT},
                {
                    "role": "user",
                    "content": user_message
                }
            ],
            "max_tokens": 8192,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "grasp_match",
                    "strict": True,
                    "schema": to_strict_json_schema(GraspMatch)
                }
            }
        }
    }
    return request

def submit_matching_job(obs_dir: str, task_json_path: str, out_dir: str, openai: OpenAI):
    batch_id_file = os.path.join(out_dir, "matching_job_id.txt")
    if os.path.exists(batch_id_file):
        with open(batch_id_file, "r") as f:
            return f.read().strip()

    annotated_grasp_library = get_annotated_grasp_library(obs_dir, out_dir)
    candidate_grasp_library = get_candidate_grasp_library(task_json_path, out_dir)

    queries = []
    for category, candidate_grasps in candidate_grasp_library.items():
        if category not in annotated_grasp_library:
            print(f"Skipping category {category} because it has no annotated grasp descriptions")
            continue
        for object_id, annotated_grasps in annotated_grasp_library[category].items():
            for candidate_grasp in candidate_grasps:
                queries.append(create_query(category, object_id, candidate_grasp, annotated_grasps))

    batch_file = BytesIO()
    for query in queries:
        batch_file.write((json.dumps(query) + "\n").encode("utf-8"))
    print(f"Submitting {len(queries)} queries to OpenAI, total size: {batch_file.tell():,} bytes")
    batch_file.seek(0)
    batch_file_id = openai.files.create(file=batch_file, purpose="batch").id
    batch = openai.batches.create(input_file_id=batch_file_id, endpoint="/v1/chat/completions", completion_window="24h")
    print(f"Submitted batch job with id: {batch.id}")
    with open(batch_id_file, "w") as f:
        f.write(batch.id)
    return batch.id

def get_matching_results(openai: OpenAI, batch_id: str):
    done_statuses = ["completed", "expired", "cancelled", "failed"]
    print("Waiting for batch job to complete...")
    while (batch := openai.batches.retrieve(batch_id)).status not in done_statuses:
        time.sleep(10)
    if batch.status != "completed":
        raise ValueError(f"Batch job {batch_id} did not complete successfully! Status: {batch.status}")
    batch_file = openai.files.content(batch.output_file_id)
    batch_file_lines = batch_file.content.decode("utf-8").splitlines()

    matched_grasps: dict[str, dict[str, dict[str, set[str]]]] = {}  # category -> object_id -> candidate_grasp_desc -> matching_grasp_descs
    n_success = 0
    for line in batch_file_lines:
        try:
            result = json.loads(line)
        except json.JSONDecodeError:
            print(f"Malformed JSON response: {line}")
            continue
        custom_id: str = result["custom_id"]
        object_category, object_id = custom_id.split("___")[0].split("_", 1)
        response = GraspMatch.model_validate_json(result["response"]["body"]["choices"][0]["message"]["content"])
        if response.matching_grasp_descs:
            if object_category not in matched_grasps:
                matched_grasps[object_category] = {}
            if object_id not in matched_grasps[object_category]:
                matched_grasps[object_category][object_id] = {}
            assert response.candidate_grasp_desc not in matched_grasps[object_category][object_id]
            matched_grasps[object_category][object_id][response.candidate_grasp_desc] = set(response.matching_grasp_descs)
            n_success += 1
    print(f"Filtering yield: {n_success}/{len(batch_file_lines)} ({n_success / len(batch_file_lines):.0%})")
    return matched_grasps

def retrieve_matching_job(openai: OpenAI, batch_id: str, task_json_path: str, obs_dir: str, out_dir: str):
    out_path = os.path.join(out_dir, "matched_tasks.csv")
    if os.path.exists(out_path):
        print(f"Skipping retrieval of matching job {batch_id}, already exists: {out_path}")
        return

    matched_grasps = get_matching_results(openai, batch_id)

    with open(task_json_path, "r") as f:
        tasks_spec: TasksSpec = json.load(f)

    n_samples = 0
    with open(out_path, "w") as out_csv:
        writer = csv.DictWriter(out_csv, ["scene_path", "scene_id", "view_id", "obs_id", "task", "original_grasp_desc", "matching_grasp_desc"])
        writer.writeheader()

        for scene_file in tqdm(os.listdir(obs_dir), desc="Generating samples"):
            if not scene_file.endswith(".hdf5"):
                continue

            scene_id = scene_file[:-len(".hdf5")]
            with h5py.File(os.path.join(obs_dir, scene_file), "r") as f:
                for view_id in f.keys():
                    grasps_in_view: dict[str, tuple[list[str], list[str]]] = {}  # category -> (list of obs_ids, list of grasp_descs)
                    for obs_id in f[view_id].keys():
                        if not obs_id.startswith("obs_"):
                            continue
                        annotation = yaml.safe_load(f[view_id][obs_id]["annot"][()])
                        grasp_desc = annotation["grasp_description"]
                        category = annotation["object_category"]
                        if category not in grasps_in_view:
                            grasps_in_view[category] = ([], [])
                        grasps_in_view[category][0].append(obs_id)
                        grasps_in_view[category][1].append(grasp_desc)

                    object_names = list(f[view_id]["object_names"].asstr())
                    for name in object_names:
                        if "_" not in name:
                            continue
                        category, object_id = name.split("_", 1)
                        if (
                            category not in tasks_spec or
                            category not in matched_grasps or
                            object_id not in matched_grasps[category] or
                            category not in grasps_in_view
                        ):
                            continue

                        for grasp_info in tasks_spec[category].values():
                            original_grasp_desc: str = grasp_info["info"]["natural_language"]
                            task_infos = grasp_info["tasks"]
                            if original_grasp_desc not in matched_grasps[category][object_id]:
                                continue
                            matching_grasp_descs = matched_grasps[category][object_id][original_grasp_desc]
                            assert len(matching_grasp_descs) > 0

                            obs_ids_for_obj, grasp_descs_for_obj = grasps_in_view[category]
                            matching_grasp_descs_in_view = [desc for desc in grasp_descs_for_obj if desc in matching_grasp_descs]
                            if len(matching_grasp_descs_in_view) == 0:
                                continue
                            matching_grasp_desc = np.random.choice(matching_grasp_descs_in_view)

                            matching_obs_id = next((i for i, desc in zip(obs_ids_for_obj, grasp_descs_for_obj) if desc == matching_grasp_desc), None)
                            if matching_obs_id is None:
                                raise ValueError(f"Matching grasp description {matching_grasp_desc} not found in {grasp_descs_for_obj}")

                            # print("=" * 100)
                            # print(f"Original: {original_grasp_desc}\nMatching: {matching_grasp_desc}")

                            for task_info in task_infos:
                                task = task_info["text"]
                                writer.writerow({
                                    "scene_path": scene_file,
                                    "scene_id": scene_id,
                                    "view_id": view_id,
                                    "obs_id": matching_obs_id,
                                    "task": task,
                                    "original_grasp_desc": original_grasp_desc,
                                    "matching_grasp_desc": matching_grasp_desc
                                })
                                n_samples += 1

    print(f"Final dataset size: {n_samples:,}")

def main():
    args = get_args()

    os.makedirs(args.out_dir, exist_ok=True)

    openai = OpenAI()

    if args.submit:
        batch_id = submit_matching_job(args.obs_dir, args.task_json, args.out_dir, openai)
    else:
        batch_id = args.retrieve

    if args.retrieve is not None:
        assert batch_id, "No batch id provided!"
        retrieve_matching_job(openai, batch_id, args.task_json, args.obs_dir, args.out_dir)

if __name__ == "__main__":
    main()
