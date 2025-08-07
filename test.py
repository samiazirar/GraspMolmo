#!/usr/bin/env python3
"""
run_graspmolmo_demo.py
Fully working example that:
  • grabs / loads an RGB-D frame
  • back-projects it to a point cloud
  • samples dummy 6-DoF grasp proposals
  • asks GraspMolmo which grasp best satisfies a language task
"""

# --------------------------------------------------------------------------- #
# Imports                                                                     #
# --------------------------------------------------------------------------- #
import argparse, time, random, re
from pathlib import Path

import numpy as np
import open3d as o3d
import cv2
import torch

from graspmolmo.inference.grasp_predictor import GraspMolmo


# --------------------------------------------------------------------------- #
# Data helpers                                                                #
# --------------------------------------------------------------------------- #
def get_image(live: bool):
    """Return (rgb[H,W,3,uint8], depth[H,W,float32, metres])."""
    if live:
        import pyrealsense2 as rs
        pipe, cfg = rs.pipeline(), rs.config()
        cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        pipe.start(cfg)
        time.sleep(0.2)                                    # warm-up
        frames = pipe.wait_for_frames()
        rgb   = np.asanyarray(frames.get_color_frame().get_data())
        depth = np.asanyarray(frames.get_depth_frame().get_data()).astype(np.float32) / 1000.0
        pipe.stop()
    else:
        rgb_path   = Path("assets/example_rgb.png")
        depth_path = Path("assets/example_depth.npy")
        if not (rgb_path.exists() and depth_path.exists()):
            raise FileNotFoundError("Place test data in assets/ before running without --live")
        rgb   = cv2.imread(str(rgb_path))
        depth = np.load(depth_path).astype(np.float32)
    return cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB), depth


def backproject(rgb, depth, K):
    """Convert depth to point cloud in camera frame."""
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d.geometry.Image(rgb), o3d.geometry.Image(depth),
        depth_scale=1.0, depth_trunc=1.5, convert_rgb_to_intensity=False
    )
    intr = o3d.camera.PinholeCameraIntrinsic()
    h, w = depth.shape
    intr.set_intrinsics(w, h, K[0, 0], K[1, 1], K[0, 2], K[1, 2])
    return o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intr)


def sample_grasps(pcd, n=64):
    """Uniformly sample grasp centres + identity orientation + 6 cm width."""
    pts  = np.asarray(pcd.points)
    sel  = pts[random.sample(range(len(pts)), k=min(n, len(pts)))]
    quat = np.tile([0, 0, 0, 1], (len(sel), 1))            # no rotation
    w    = np.full((len(sel), 1), 0.06)                    # 6 cm width
    return np.hstack([sel, quat, w])                       # (n, 8)


# --------------------------------------------------------------------------- #
# GraspMolmo call with shape-fix                                              #
# --------------------------------------------------------------------------- #
def predict_point(gm, rgb, task):
    """
    Run Molmo language+vision backbone and return 3-D point (numpy, shape (3,))
    """
    inputs = gm.processor.process(images=rgb, text=task, return_tensors="pt")

    # ---- FIX: make tensors 4-D (B, T, N, D) for images / masks --------------
    inputs["images"]      = inputs["images"].unsqueeze(0).unsqueeze(0)
    inputs["image_masks"] = inputs["image_masks"].unsqueeze(0).unsqueeze(0)
    # text needs only batch dim
    inputs["input_ids"]      = inputs["input_ids"].unsqueeze(0)
    inputs["attention_mask"] = inputs["attention_mask"].unsqueeze(0)

    inputs = {k: v.to(gm.device) for k, v in inputs.items()}

    with torch.no_grad():
        toks = gm.model.generate_from_batch(
            **inputs,
            do_sample=True,
            top_p=0.9,
            temperature=1.0,
            max_new_tokens=30,
        )

    text = gm.processor.tokenizer.batch_decode(toks, skip_special_tokens=False)[0]
    m = re.search(r"<point>\s*([0-9eE\+\-\. ]+)\s*</point>", text)
    if m is None:
        raise RuntimeError(f"Model output missing <point> tag:\n{text}")
    return np.fromstring(m.group(1), sep=" ")


def select_grasp(grasps, point):
    """Return index of the grasp whose centre is closest to `point`."""
    dists = np.linalg.norm(grasps[:, :3] - point, axis=1)
    return int(np.argmin(dists))


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="grasp_it", help="Language task prompt")
    ap.add_argument("--live", action="store_true",
                    help="Capture RealSense frame instead of reading assets/")
    return ap.parse_args()


def main():
    args = parse_args()

    rgb, depth = get_image(args.live)

    # Intel RealSense D435 intrinsics @ 640×480 (adjust if needed)
    K = np.array([[615.0,   0.0, 320.0],
                  [  0.0, 615.0, 240.0],
                  [  0.0,   0.0,   1.0]], dtype=np.float32)

    pcd     = backproject(rgb, depth, K)
    grasps  = sample_grasps(pcd, n=64)

    gm      = GraspMolmo(device="cuda" if torch.cuda.is_available() else "cpu")
    point   = predict_point(gm, rgb, args.task)
    idx     = select_grasp(grasps, point)

    print("\n=== GraspMolmo demo ===")
    print(f"Task prompt     : {args.task}")
    print(f"Predicted point : {point}")
    print(f"Chosen grasp id : {idx}")
    print(f"Chosen grasp    : {grasps[idx]}\n")


if __name__ == "__main__":
    main()
