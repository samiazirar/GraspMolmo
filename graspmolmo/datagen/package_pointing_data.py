import argparse
import os
import threading
from concurrent.futures import Future, ProcessPoolExecutor, as_completed, CancelledError, Executor
import traceback
from functools import partial

import pandas as pd
import numpy as np
import h5py
from PIL import Image
import json

from graspmolmo.utils import tqdm


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path")
    parser.add_argument("obs_dir")
    parser.add_argument("out_dir")
    parser.add_argument("--n-proc", type=int, default=32)
    parser.add_argument("--cot", action="store_true", help="Use chain of thought")
    parser.add_argument("--format", choices=["molmo", "robopoint"], default="molmo")
    parser.add_argument("--skip-images", action="store_false", dest="copy_images")
    return parser.parse_args()

def points_to_tuples(grasp_pts: np.ndarray):
    if grasp_pts.ndim == 1:
        grasp_pts = grasp_pts[None]
    assert grasp_pts.ndim == 2 and grasp_pts.shape[-1] == 2
    return f"[{', '.join([f'({x:.3f}, {y:.3f})' for x, y in grasp_pts])}]"

def create_robopoint_sample(row_id: str, image_path: str, task: str, grasp_desc: str, grasp_pt: np.ndarray, cot: bool):
    response = ""
    if cot:
        response = f"In order to accomplish the task \"{task}\", the optimal grasp is described as follows: \"{grasp_desc}\".\n\n"
    response += points_to_tuples(grasp_pt)
    return {
        "id": row_id,
        "image": image_path,
        "conversations": [
            {
                "from": "human",
                "value": f"<image>\nPoint to where to grasp to accomplish the following task:\n{task}\n\nYour answer should be formatted as first an explanation of how to grasp in order to accomplish the task, and then a list of tuples, i.e. [(x1, y1), (x2, y2), ...], where each tuple contains the x and y coordinates of a point satisfying the conditions above. The coordinates should be between 0 and 1, indicating the normalized pixel locations of the points in the image."
            },
            {
                "from": "gpt",
                "value": response
            }
        ]
    }

def point_to_xml(grasp_pt: np.ndarray):
    if grasp_pt.ndim == 2:
        assert grasp_pt.shape == (1, 2)
        grasp_pt = grasp_pt[0]
    assert grasp_pt.shape == (2,)
    point_desc = "Where to grasp the object"
    return f"<point x=\"{grasp_pt[0]*100:.1f}\" y=\"{grasp_pt[1]*100:.1f}\" alt=\"{point_desc}\">{point_desc}</point>"

def create_molmo_sample(row_id: str, image_path: str, task: str, grasp_desc: str, grasp_pt: np.ndarray, cot: bool):
    response = ""
    if cot:
        response = f"In order to accomplish the task \"{task}\", the optimal grasp is described as follows: \"{grasp_desc}\".\n\n"
    response += point_to_xml(grasp_pt)

    return {
        "id": row_id,
        "image": image_path,
        "conversations": [
            {
                "from": "human",
                "value": f"Point to the grasp that would accomplish the following task: {task}"
            },
            {
                "from": "gpt",
                "value": response
            }
        ]
    }

def create_sample(obs_dir: str, row: pd.Series, cot: bool, format: str):
    sample_fn = create_molmo_sample if format == "molmo" else create_robopoint_sample

    scene_id = row["scene_id"]
    view_id = row["view_id"]
    task = row["task"]
    grasp_desc = row["matching_grasp_desc"]
    img_relpath = os.path.join("images", f"{scene_id}-{view_id}.png")

    with h5py.File(os.path.join(obs_dir, row["scene_path"]), "r") as f:
        view_group = f[row["view_id"]]
        img_shape = view_group["rgb"].shape
        grasp_pt = view_group[row["obs_id"]]["grasp_point_px"][:]
    grasp_pt = grasp_pt / np.array([img_shape[1], img_shape[0]])

    row_id = str(row.name)
    sample = sample_fn(row_id, img_relpath, task, grasp_desc, grasp_pt, cot)
    return sample

def on_job_done(f: Future, submit_semaphore: threading.Semaphore, pbar: tqdm, executor: Executor):
    submit_semaphore.release()
    pbar.update(1)
    try:
        f.result()
    except CancelledError:
        pass
    except:
        traceback.print_exc()
        executor.shutdown(wait=False, cancel_futures=True)

def copy_images(scene_path: str, out_dir: str):
    scene_id = os.path.basename(scene_path).rsplit(".", 1)[0]
    with h5py.File(scene_path, "r") as f:
        for view_id in f.keys():
            rgb_arr = f[view_id]["rgb"][:]
            img = Image.fromarray(rgb_arr)
            img.save(os.path.join(out_dir, "images", f"{scene_id}-{view_id}.png"))

def main():
    args = get_args()
    df = pd.read_csv(args.csv_path)

    os.makedirs(args.out_dir, exist_ok=True)

    submit_semaphore = threading.Semaphore(4 * args.n_proc)
    with ProcessPoolExecutor(max_workers=args.n_proc) as executor:
        save_path = os.path.join(args.out_dir, f"{args.format}_data{'_cot' if args.cot else ''}.json")
        if not os.path.exists(save_path):
            with tqdm(total=len(df), desc="Constructing samples") as pbar:
                lines: list[str] = []
                callback = partial(on_job_done, submit_semaphore=submit_semaphore, pbar=pbar, executor=executor)
                futures: list[Future] = []
                for _, row in df.iterrows():
                    submit_semaphore.acquire()
                    future = executor.submit(create_sample, args.obs_dir, row, args.cot, args.format)
                    future.add_done_callback(callback)
                    futures.append(future)
                for future in as_completed(futures):
                    lines.append(future.result())

            print(f"Generated {len(lines)} data points, saving to {save_path}")
            with open(save_path, "w") as f:
                json.dump(lines, f, indent=2)
        else:
            print(f"Already generated JSON data, skipping.")

        if args.copy_images:
            os.makedirs(os.path.join(args.out_dir, "images"), exist_ok=True)
            obs_files = [fn for fn in os.listdir(args.obs_dir) if fn.endswith(".hdf5")]
            with tqdm(total=len(obs_files), desc="Copying images") as pbar:
                callback = partial(on_job_done, submit_semaphore=submit_semaphore, pbar=pbar, executor=executor)
                futures: list[Future] = []
                for fn in obs_files:
                    submit_semaphore.acquire()
                    future = executor.submit(copy_images, os.path.join(args.obs_dir, fn), args.out_dir)
                    future.add_done_callback(callback)
                    futures.append(future)
                for future in as_completed(futures):
                    if (e := future.exception()) is not None:
                        raise e

if __name__ == "__main__":
    main()
