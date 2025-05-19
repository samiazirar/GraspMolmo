import argparse
import os
import threading
from concurrent.futures import Future, ProcessPoolExecutor, as_completed, CancelledError, Executor
import traceback
from functools import partial
from typing import Any

import numpy as np
import torch
from PIL import Image
import json

from graspmolmo.utils import tqdm
from graspmolmo.eval.utils import TaskGraspScanLibrary


GRASP_VOLUME_SIZE = np.array([0.082, 0.01, 0.112-0.066])
GRASP_VOLUME_CENTER = np.array([0, 0, (0.066+0.112)/2])


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("taskgrasp_dir")
    parser.add_argument("out_dir")
    parser.add_argument("whitelist_path")
    parser.add_argument("--n-proc", type=int, default=32)
    parser.add_argument("--skip-images", action="store_false", dest="copy_images")
    return parser.parse_args()


@torch.compile
def get_grasp_constraint_torch(grasps: torch.Tensor):
    """
    Args:
        grasp: (K, 4, 4) The grasp pose.
    Returns:
        A: (K, 6, 3) and b: (K, 6) such that Ax <= b implies that x is in the grasp volume.
    """
    min_pos, max_pos = GRASP_VOLUME_CENTER - GRASP_VOLUME_SIZE/2, GRASP_VOLUME_CENTER + GRASP_VOLUME_SIZE/2
    min_pos = torch.as_tensor(min_pos, dtype=grasps.dtype, device=grasps.device)
    max_pos = torch.as_tensor(max_pos, dtype=grasps.dtype, device=grasps.device)
    R, t = grasps[:, :3, :3], grasps[:, :3, 3]
    Rt = R.transpose(2, 1)
    Rt_times_t = (Rt @ torch.unsqueeze(t, dim=-1))[..., 0]

    A = torch.cat([Rt, -Rt], dim=1)
    b = torch.cat([max_pos + Rt_times_t, -min_pos - Rt_times_t], dim=1)

    return A, b

@torch.compile
def get_grasp_points_torch(pc: torch.Tensor, grasps: torch.Tensor):
    """
    Args:
        pc: (N, 3) The point cloud of the scene.
        grasps: (K, 4, 4) The grasp poses to get the points for.
    Returns:
        The grasp points in the point cloud, shape (K, 3).
    """
    A, b = get_grasp_constraint_torch(grasps)

    A_bc = torch.broadcast_to(torch.unsqueeze(A, dim=1), (A.shape[0], pc.shape[0], 6, 3))  # (K, N, 6, 3)
    b_bc = torch.broadcast_to(torch.unsqueeze(b, dim=1), (b.shape[0], pc.shape[0], 6))  # (K, N, 6)
    pc_bc = torch.broadcast_to(torch.unsqueeze(pc, dim=0), (A.shape[0], pc.shape[0], 3))[..., None]  # (K, N, 3, 1)

    trf_pc = torch.squeeze(A_bc @ pc_bc, dim=-1)  # (K, N, 6)
    in_grasp_mask = torch.all(trf_pc <= b_bc, dim=-1)  # (K, N), (i, j) = True iff pc[j] is in grasp i

    grasp_points = []
    for i in range(len(grasps)):
        grasp_ref_pos = grasps[i, :3, 3] + grasps[i, :3, 2] * 0.066
        if not torch.any(in_grasp_mask[i]):
            closest_idx = torch.argmin(torch.linalg.norm(pc - grasp_ref_pos[None], dim=-1))
            grasp_points.append(pc[closest_idx])
        else:
            in_grasp_points = pc[in_grasp_mask[i]]
            closest_idx = torch.argmin(torch.linalg.norm(in_grasp_points - grasp_ref_pos[None], dim=-1)).item()
            grasp_points.append(in_grasp_points[closest_idx])
    grasp_points = torch.stack(grasp_points, dim=0)
    return grasp_points

def get_grasp_points(pc: np.ndarray, grasps: np.ndarray):
    """
    Args:
        pc: (N, 3) The point cloud of the scene.
        grasps: (K, 4, 4) The grasp poses to get the points for.
    Returns:
        The grasp points, shape (K, 3).
    """
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    pc_torch = torch.as_tensor(pc, dtype=torch.float32, device=device)
    grasps_torch = torch.as_tensor(grasps, dtype=torch.float32, device=device)
    return get_grasp_points_torch(pc_torch, grasps_torch).cpu().numpy().astype(pc.dtype)


def point_to_xml(grasp_pt: np.ndarray):
    if grasp_pt.ndim == 2:
        assert grasp_pt.shape == (1, 2)
        grasp_pt = grasp_pt[0]
    assert grasp_pt.shape == (2,)
    point_desc = "Where to grasp the object"
    return f"<point x=\"{grasp_pt[0]*100:.1f}\" y=\"{grasp_pt[1]*100:.1f}\" alt=\"{point_desc}\">{point_desc}</point>"

def create_molmo_sample(sample_id: str, image_path: str, task: str, grasp_pt: np.ndarray):
    return {
        "id": sample_id,
        "image": image_path,
        "conversations": [
            {
                "from": "human",
                "value": f"Point to the grasp that would accomplish the following task: {task}"
            },
            {
                "from": "gpt",
                "value": point_to_xml(grasp_pt)
            }
        ]
    }

def create_sample(out_dir: str, data: dict[str, Any], grasp_labels: dict[str, dict[str, int]], copy_images: bool):
    if data["registered_grasps"] is None or data["segmented_pc"] is None or data["object_id"] not in grasp_labels:
        return None

    object_id = data["object_id"]
    object_name = data["object_name"]
    scan_id = data["scan_id"]
    cam_K = data["cam_params"]
    img_relpath = os.path.join("images", f"{object_id}_{scan_id}.png")
    image: Image.Image = data["rgb"]
    grasps = data["registered_grasps"]

    grasp_points = get_grasp_points(data["segmented_pc"][:, :3], grasps)
    grasp_points_2d = grasp_points @ cam_K.T
    grasp_points_2d = grasp_points_2d[:, :2] / grasp_points_2d[:, 2:3]
    grasp_points_px = grasp_points_2d / np.array([image.width, image.height])

    if copy_images:
        image.save(os.path.join(out_dir, img_relpath))

    samples: list[dict] = []
    for task in grasp_labels[object_id]:
        grasp_id = grasp_labels[object_id][task]
        sample_id = f"{object_id}_{scan_id}_{task}_{grasp_id}"

        grasp_pt = grasp_points_px[grasp_id]
        task_text = f"Grasp the {object_name} to {task}"

        sample = create_molmo_sample(sample_id, img_relpath, task_text, grasp_pt)
        samples.append(sample)

    return samples

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

def main():
    args = get_args()

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, "images"), exist_ok=True)
    library = TaskGraspScanLibrary(args.taskgrasp_dir)

    whitelist: set[tuple[str, str]] = set()
    with open(args.whitelist_path, "r") as f:
        for line in f.read().strip().splitlines():
            obj_id, _, task = line.split(":")[0].split("-")
            whitelist.add((obj_id, task))

    grasp_labels: dict[str, dict[str, int]] = {}  # object_id -> task -> grasp_id
    with open(os.path.join(args.taskgrasp_dir, "task2_results.txt"), "r") as f:
        for line in tqdm(f.read().strip().splitlines(), desc="Loading grasp labels"):
            part1, label = line.split(":")
            object_id, grasp_id, task = part1.split("-")
            if label == "1" and (object_id, task) in whitelist:
                if object_id not in grasp_labels:
                    grasp_labels[object_id] = {}
                grasp_labels[object_id][task] = int(grasp_id)

    submit_semaphore = threading.Semaphore(4 * args.n_proc)
    with ProcessPoolExecutor(max_workers=args.n_proc) as executor:
        save_path = os.path.join(args.out_dir, f"tg_train_molmo_data.json")
        if not os.path.exists(save_path):
            with tqdm(total=len(library), desc="Constructing samples") as pbar:
                all_samples: list[dict] = []
                callback = partial(on_job_done, submit_semaphore=submit_semaphore, pbar=pbar, executor=executor)
                futures: list[Future] = []
                for data in library:
                    submit_semaphore.acquire()
                    future = executor.submit(create_sample, args.out_dir, data, grasp_labels, args.copy_images)
                    future.add_done_callback(callback)
                    futures.append(future)
                for future in as_completed(futures):
                    samples = future.result()
                    if samples is not None:
                        all_samples.extend(samples)

            print(f"Generated {len(all_samples)} data points, saving to {save_path}")
            with open(save_path, "w") as f:
                json.dump(all_samples, f, indent=2)
        else:
            print(f"Already generated JSON data, skipping.")

if __name__ == "__main__":
    main()
