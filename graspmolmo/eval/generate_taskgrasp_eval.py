import argparse
import os

import numpy as np
import trimesh
from tqdm import tqdm

from acronym_tools import create_gripper_marker

from graspmolmo.eval.mask_detection import MaskDetector
from graspmolmo.eval.pointcloud import CompositePCRegistration, DeepGMRRegistration, ICPRegistration
from graspmolmo.eval.utils import TaskGraspScanLibrary, img_to_pc

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scan-dir", type=str, required=True)
    parser.add_argument("--gen-scene-dir", type=str)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()

def main():
    args = get_args()
    scan_dir = args.scan_dir
    gen_scene_dir = args.gen_scene_dir
    if gen_scene_dir is not None:
        os.makedirs(gen_scene_dir, exist_ok=True)

    tg_library = TaskGraspScanLibrary(scan_dir)
    mask_detector = MaskDetector()
    pc_registration = CompositePCRegistration(
        DeepGMRRegistration(),
        ICPRegistration(),
    )

    for elem in tqdm(tg_library):
        out_grasps_file = os.path.join(scan_dir, elem["object_id"], f"{elem['scan_id']}_registered_grasps.npy")
        out_pc_file = os.path.join(scan_dir, elem["object_id"], f"{elem['scan_id']}_segmented_pc.npy")
        out_trf_file = os.path.join(scan_dir, elem["object_id"], f"{elem['scan_id']}_pc_to_img_trf.npy")
        if not args.overwrite and os.path.isfile(out_grasps_file) and os.path.isfile(out_pc_file) and os.path.isfile(out_trf_file):
            continue
        object_mask = mask_detector.detect_mask(elem["object_name"], elem["rgb"])
        if object_mask is None:
            print(f"No mask detected for {elem['object_id']}_{elem['scan_id']}")
            continue
        rgb_array = np.asarray(elem["rgb"])
        object_pc = img_to_pc(rgb_array, elem["depth"], elem["cam_params"], object_mask)  # (N, 6)

        x_min, x_max = np.percentile(object_pc[:, 0], [2, 98])
        y_min, y_max = np.percentile(object_pc[:, 1], [2, 98])
        z_min, z_max = np.percentile(object_pc[:, 2], [2, 98])
        object_pc_crop = object_pc[
            (object_pc[:, 0] >= x_min) &
            (object_pc[:, 0] <= x_max) &
            (object_pc[:, 1] >= y_min) &
            (object_pc[:, 1] <= y_max) &
            (object_pc[:, 2] >= z_min) &
            (object_pc[:, 2] <= z_max)
        ]

        trf, object_pc_trf, cost = pc_registration.register(elem["fused_pc"][:,:3], object_pc_crop[:,:3])

        if cost >= 0.006:
            print(f"Failed to register {elem['object_id']}_{elem['scan_id']} with cost {cost}")
            continue

        trf_inv = np.linalg.inv(trf)
        fused_pc_registered = elem["fused_pc"][:, :3] @ trf_inv[:3,:3].T + trf_inv[:3,3]
        fused_pc_obj = trimesh.PointCloud(fused_pc_registered[:, :3], np.tile([255, 0, 0], (len(fused_pc_registered), 1)))

        scan_pc = img_to_pc(rgb_array, elem["depth"], elem["cam_params"], elem["depth"] < 2)
        scan_pc_obj = trimesh.PointCloud(scan_pc[:, :3], scan_pc[:, 3:].astype(np.uint8))

        scene = trimesh.Scene([fused_pc_obj, scan_pc_obj])
        grasps = trf_inv @ elem["fused_grasps"]
        for grasp in grasps:
            marker: trimesh.Trimesh = create_gripper_marker([255, 255, 0])
            marker.apply_transform(grasp)
            scene.add_geometry(marker)
        print(f"Registered {elem['object_id']}_{elem['scan_id']} with cost {cost}")
        np.save(out_grasps_file, grasps)
        np.save(out_pc_file, object_pc_crop)
        np.save(out_trf_file, trf_inv)
        if gen_scene_dir is not None:
            scene.export(os.path.join(gen_scene_dir, f"{elem['object_id']}_{elem['scan_id']}.glb"))

if __name__ == "__main__":
    main()
