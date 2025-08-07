#!/usr/bin/env python3
"""
demo_graspmolmo.py
Minimal end-to-end example for GraspMolmo inference.

What it does
------------
1.  Acquire or load an RGB-D frame
2.  Back-project the depth map to an Open3D point-cloud (camera frame)
3.  Generate a handful of dummy 6-DoF grasp proposals      (replace with M2T2, GIGA, etc.)
4.  Ask GraspMolmo which grasp satisfies the textual task
"""

from pathlib import Path
import argparse, time, random

import numpy as np
import open3d as o3d
import cv2
import torch

from graspmolmo.inference.grasp_predictor import GraspMolmo


# --------------------------------------------------------------------------- #
#                             Helper functions                                #
# --------------------------------------------------------------------------- #
def get_image(live: bool):
    """
    Returns
    -------
    rgb   : H×W×3  uint8
    depth : H×W    float32   (metres)
    """
    if live:
        # --- live capture from an Intel RealSense ---
        import pyrealsense2 as rs
        pipe, cfg = rs.pipeline(), rs.config()
        cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        pipe.start(cfg)
        time.sleep(0.2)                    # warm-up
        frames = pipe.wait_for_frames()
        rgb   = np.asanyarray(frames.get_color_frame().get_data())
        depth = np.asanyarray(frames.get_depth_frame().get_data()).astype(np.float32) / 1000.0
        pipe.stop()
    else:
        # --- fallback to sample files ---------------------------------------
        rgb   = cv2.imread("assets/example_rgb.png")
        depth = np.load("assets/example_depth.npy").astype(np.float32)         # metres
    return rgb[..., ::-1], depth           # BGR→RGB


def backproject(rgb: np.ndarray,
                depth: np.ndarray,
                K: np.ndarray) -> o3d.geometry.PointCloud:
    """Depth → point-cloud in the camera frame."""
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d.geometry.Image(rgb), o3d.geometry.Image(depth),
        depth_scale=1.0, depth_trunc=1.5, convert_rgb_to_intensity=False
    )

    intr = o3d.camera.PinholeCameraIntrinsic()
    h, w = depth.shape
    intr.set_intrinsics(w, h, K[0, 0], K[1, 1], K[0, 2], K[1, 2])

    return o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intr)


def predict_grasps(pcd: o3d.geometry.PointCloud, n: int = 64) -> np.ndarray:
    """
    Dummy uniform grasp sampler.
    Replace this with any real 6-DoF grasp detector (M2T2, ContactGraspNet …).

    Returns
    -------
    grasps : n × 8 array
             [x, y, z, qx, qy, qz, qw, width]
    """
    pts  = np.asarray(pcd.points)
    sel  = pts[random.sample(range(len(pts)), k=min(n, len(pts)))]
    quat = np.tile([0, 0, 0, 1], (len(sel), 1))          # identity orientation
    w    = np.full((len(sel), 1), 0.06)                  # 6 cm gripper width
    return np.hstack([sel, quat, w])


# --------------------------------------------------------------------------- #
#                                   Main                                      #
# --------------------------------------------------------------------------- #
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="grasp_it", help="Natural-language task")
    ap.add_argument("--live", action="store_true",
                    help="Grab a live RealSense frame instead of reading from assets")
    return ap.parse_args()


def main():
    args = parse_args()

    # 1) RGB-D
    rgb, depth = get_image(args.live)

    # 2) Camera intrinsics (RealSense D435 default @640×480)
    K = np.array([[615.0,   0.0, 320.0],
                  [  0.0, 615.0, 240.0],
                  [  0.0,   0.0,   1.0]], dtype=np.float32)

    # 3) back-project + candidate grasps
    pcd    = backproject(rgb, depth, K)
    grasps = predict_grasps(pcd)

    # 4) GraspMolmo decision
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gm     = GraspMolmo(device=device)
    idx    = gm.pred_grasp(rgb, pcd, args.task, grasps)

    print(f"\nTask          : {args.task}")
    print(f"Chosen index  : {idx}")
    print(f"6-DoF grasp   : {grasps[idx]}\n")


if __name__ == "__main__":
    main()
