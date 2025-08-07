#!/usr/bin/env python3
"""
web_demo_graspmolmo.py
Fully self-contained GraspMolmo demo – no depth camera required.

It will:
  • download a sample RGB image + PLY point cloud the first time you run it
  • sample dummy 6-DoF grasp proposals
  • ask GraspMolmo which grasp best fits the language task
"""

# ---------------------------------------------------------------------- #
# Imports                                                                #
# ---------------------------------------------------------------------- #
import argparse, random, re, urllib.request, os
from pathlib import Path

import numpy as np
import cv2
import open3d as o3d
import torch

from graspmolmo.inference.grasp_predictor import GraspMolmo


# ---------------------------------------------------------------------- #
# Constants: public sample assets                                        #
# ---------------------------------------------------------------------- #
ASSETS_DIR   = Path("assets")
ASSETS_DIR.mkdir(exist_ok=True)

IMAGE_URL    = (
    "https://raw.githubusercontent.com/opencv/opencv/master/"
    "samples/data/baboon.jpg"
)
IMAGE_FILE   = ASSETS_DIR / "sample_rgb.jpg"

PCLOUD_URL   = (
    "https://github.com/isl-org/Open3D/raw/main/"
    "examples/test_data/fragment.ply"
)
PCLOUD_FILE  = ASSETS_DIR / "sample_pointcloud.ply"


# ---------------------------------------------------------------------- #
# Utility: download once                                                 #
# ---------------------------------------------------------------------- #
def _download(url: str, dst: Path):
    if dst.exists():
        return
    print(f"[INFO] downloading {dst.name} …")
    try:
        urllib.request.urlretrieve(url, dst)
    except Exception as e:
        print(f"[WARNING] Failed to download {dst.name}: {e}")
        return False
    return True


def create_dummy_pointcloud():
    """Create a simple dummy point cloud if download fails"""
    print("[INFO] Creating dummy point cloud...")
    # Create a simple cube point cloud
    points = []
    for x in np.linspace(-0.5, 0.5, 10):
        for y in np.linspace(-0.5, 0.5, 10):
            for z in np.linspace(-0.5, 0.5, 10):
                points.append([x, y, z])
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.array(points))
    o3d.io.write_point_cloud(str(PCLOUD_FILE), pcd)


def fetch_assets():
    _download(IMAGE_URL,  IMAGE_FILE)
    if not _download(PCLOUD_URL, PCLOUD_FILE):
        create_dummy_pointcloud()


# ---------------------------------------------------------------------- #
# Data helpers                                                           #
# ---------------------------------------------------------------------- #
def load_assets():
    rgb   = cv2.imread(str(IMAGE_FILE))         # BGR
    rgb   = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    pcd   = o3d.io.read_point_cloud(str(PCLOUD_FILE))
    return rgb, pcd


def sample_grasps(pcd, n=64):
    """
    Uniform random grasp centres + identity quaternion + 6 cm width.

    Returns
    -------
    grasps : (n, 8)  [x, y, z, qx, qy, qz, qw, width]
    """
    pts   = np.asarray(pcd.points)
    sel   = pts[random.sample(range(len(pts)), k=min(n, len(pts)))]
    quat  = np.tile([0, 0, 0, 1], (len(sel), 1))          # no rotation
    width = np.full((len(sel), 1), 0.06)
    return np.hstack([sel, quat, width])


# ---------------------------------------------------------------------- #
# GraspMolmo call (with the tensor-rank fix)                             #
# ---------------------------------------------------------------------- #
def predict_point(gm: GraspMolmo, rgb: np.ndarray, task: str):
    """
    Run GraspMolmo and return a 3-D point (numpy shape (3,)).
    """
    inputs = gm.processor.process(images=rgb, text=task, return_tensors="pt")

    # --- FIX: make image & mask 4-D (B, T, N, D) -------------------------
    inputs["images"]      = inputs["images"].unsqueeze(0).unsqueeze(0)
    inputs["image_masks"] = inputs["image_masks"].unsqueeze(0).unsqueeze(0)
    # text needs only batch dim
    inputs["input_ids"]      = inputs["input_ids"].unsqueeze(0)
    inputs["attention_mask"] = inputs["attention_mask"].unsqueeze(0)

    inputs = {k: v.to(gm.device) for k, v in inputs.items()}

    with torch.no_grad():
        toks = gm.model.generate_from_batch(
            **inputs,
            do_sample=True, top_p=0.9, temperature=1.0, max_new_tokens=30
        )

    txt = gm.processor.tokenizer.batch_decode(
        toks, skip_special_tokens=False
    )[0]
    m = re.search(r"<point>\s*([0-9eE\+\-\. ]+)\s*</point>", txt)
    if m is None:
        raise RuntimeError(f"Molmo output missing <point> tag:\n{txt}")
    return np.fromstring(m.group(1), sep=" ")


def choose_grasp(grasps, point):
    """closest centre → index"""
    d = np.linalg.norm(grasps[:, :3] - point, axis=1)
    return int(np.argmin(d))


# ---------------------------------------------------------------------- #
# Main                                                                   #
# ---------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="grasp_it",
                        help="Natural-language instruction")
    args = parser.parse_args()

    fetch_assets()
    rgb, pcd   = load_assets()
    grasps     = sample_grasps(pcd)

    gm         = GraspMolmo(device="cuda" if torch.cuda.is_available()
                            else "cpu")
    point      = predict_point(gm, rgb, args.task)
    idx        = choose_grasp(grasps, point)

    print("\n=== GraspMolmo Web Demo ===")
    print(f"Task prompt         : {args.task}")
    print(f"Predicted 3-D point : {point}")
    print(f"Chosen grasp index  : {idx}")
    print(f"Chosen grasp vector : {grasps[idx]}\n")


if __name__ == "__main__":
    main()
