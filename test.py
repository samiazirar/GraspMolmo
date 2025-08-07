#!/usr/bin/env python3
"""
Uses GraspMolmo on an RGB-D frame that is fetched *automatically* from the
internet (Open3D’s sample Redwood dataset).  No local sensors needed.
"""

from pathlib import Path
import argparse, random

import numpy as np
import open3d as o3d              # auto-downloads the sample when first used
import cv2
import torch

from graspmolmo.inference.grasp_predictor import GraspMolmo


# --------------------------------------------------------------------------- #
#                        Step-by-step helper functions                         #
# --------------------------------------------------------------------------- #
def fetch_sample_frame():
    """Download 1st RGB-D pair of Redwood living-room1 and its intrinsics."""
    data = o3d.data.SampleRedwoodRGBDImages()         # triggers HTTP fetch
    color = o3d.io.read_image(data.color_paths[0])
    depth = o3d.io.read_image(data.depth_paths[0])
    Kjson = data.camera_intrinsic_path               # intrinsics in JSON
    intr = o3d.io.read_pinhole_camera_intrinsic(Kjson)
    return color, depth, intr


def rgbd_to_pointcloud(color_o3d, depth_o3d, intr):
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_o3d, depth_o3d, depth_scale=1000.0, depth_trunc=3.0,
        convert_rgb_to_intensity=False)
    return o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intr)


def dummy_grasp_sampler(pcd, n=64):
    """Uniformly sample `n` points & make dummy parallel-jaw grasps."""
    xyz   = np.asarray(pcd.points)
    sel   = xyz[np.random.choice(len(xyz), size=min(n, len(xyz)), replace=False)]
    quat  = np.tile([0, 0, 0, 1], (len(sel), 1))      # identity quaternion
    width = np.full((len(sel), 1), 0.06)              # 6 cm
    return np.hstack([sel, quat, width])


# --------------------------------------------------------------------------- #
#                                    main                                     #
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="grasp_it",
                        help="Natural-language instruction for the grasp")
    args = parser.parse_args()

    # 1) Internet-fetched RGB-D sample
    color_img, depth_img, intr = fetch_sample_frame()

    # 2) Build point cloud + dummy grasps
    pcd    = rgbd_to_pointcloud(color_img, depth_img, intr)
    grasps = dummy_grasp_sampler(pcd)

    # 3) Call GraspMolmo with camera intrinsics
    K  = np.asarray(intr.intrinsic_matrix)      # (3,3) NumPy array
    gm = GraspMolmo()
    idx = gm.pred_grasp(np.asarray(color_img),  # RGB image (H,W,3) uint8
                        pcd,                    # Open3D point cloud
                        args.task,              # natural-language task
                        grasps,                 # N×8 candidate grasps
                        cam_K=K)                # NEW ➜ intrinsics!

    print(f"Task      : {args.task}")
    print(f"Best idx  : {idx}")
    print(f"6-DoF pose: {grasps[idx]}")

if __name__ == "__main__":
    main()