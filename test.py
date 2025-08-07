#!/usr/bin/env python3
"""
Smoke-test for GraspMolmo on a single RGB-D frame.

• Downloads a sample frame + intrinsics from Open3D’s Redwood living-room1 dataset.
• Builds a point cloud and 64 dummy grasp proposals.
• Calls GraspMolmo with a natural-language task and prints the winning grasp.
"""

from pathlib import Path
import argparse, warnings, os, random
import numpy as np
import open3d as o3d
import torch

# GraspMolmo helper ----------------------------------------------------------
from graspmolmo.inference.grasp_predictor import GraspMolmo     # pip install graspmolmo

# ---------- Fix #1: batch-dimension hot-patch (drop after upstream merge) ---
def _add_batch_dim(func):
    """Wrap GraspMolmo._pred so every tensor has an explicit batch dim."""
    def wrapper(self, image, task, verbosity=0):
        inputs = self.processor.process(images=image, text=task,
                                        return_tensors="pt")
        inputs = {k: (v.unsqueeze(0) if v.ndim == 1 else v)
                  for k, v in inputs.items()}  # patch
        return self.model.generate_from_batch(
            inputs, self.gen_cfg, tokenizer=self.processor.tokenizer)
    return wrapper
# --------------------------------------------------------------------------- #

def fetch_sample_frame():
    """1st RGB-D pair from Redwood living-room1 + intrinsics."""
    data = o3d.data.SampleRedwoodRGBDImages()             # auto-downloads first time  [oai_citation:2‡Open3D](https://www.open3d.org/docs/latest/python_api/open3d.data.SampleRedwoodRGBDImages.html?utm_source=chatgpt.com)
    color = o3d.io.read_image(data.color_paths[0])
    depth = o3d.io.read_image(data.depth_paths[0])
    intr  = o3d.io.read_pinhole_camera_intrinsic(data.camera_intrinsic_path)
    return color, depth, intr                              # intr holds 3×3 K  [oai_citation:3‡Open3D](https://www.open3d.org/docs/latest/python_api/open3d.camera.PinholeCameraIntrinsic.html?utm_source=chatgpt.com)

def rgbd_to_pointcloud(color, depth, intr):
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color, depth, depth_scale=1000.0, depth_trunc=3.0,
        convert_rgb_to_intensity=False)                    # Open3D RGB-D pipeline  [oai_citation:4‡Open3D](https://www.open3d.org/docs/latest/tutorial/geometry/rgbd_image.html?utm_source=chatgpt.com)
    return o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intr)

def dummy_grasp_sampler(pcd, n=64):
    """Uniform random points with identity orientation + 6 cm width."""
    xyz   = np.asarray(pcd.points)
    sel   = xyz[random.sample(range(len(xyz)), k=min(n, len(xyz)))]
    quat  = np.tile([0, 0, 0, 1], (len(sel), 1))           # unit quaternion
    width = np.full((len(sel), 1), 0.06)
    return np.hstack([sel, quat, width])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="pick up the yellow cup",
                        help="Natural-language instruction for the grasp")
    args = parser.parse_args()

    # Silence TensorFlow duplicate-plugin spam (unrelated to PyTorch)  [oai_citation:5‡Stack Overflow](https://stackoverflow.com/questions/79096274/tensorflow-errors-cufft-cudnn-cublas-and-assertion-n-this-size-fail?utm_source=chatgpt.com) [oai_citation:6‡GitHub](https://github.com/tensorflow/tensorflow/issues/62075?utm_source=chatgpt.com)
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    warnings.filterwarnings("ignore")

    # 1 Fetch RGB-D sample
    color, depth, intr = fetch_sample_frame()

    # 2 Build point cloud & candidate grasps
    pcd    = rgbd_to_pointcloud(color, depth, intr)
    grasps = dummy_grasp_sampler(pcd)

    # 3 Initialise GraspMolmo
    gm = GraspMolmo()                                      # model card & paper  [oai_citation:7‡Hugging Face](https://huggingface.co/allenai/GraspMolmo?utm_source=chatgpt.com) [oai_citation:8‡arXiv](https://arxiv.org/html/2505.13441v1?utm_source=chatgpt.com)
    if not hasattr(gm, "_patched"):                        # one-shot monkey-patch
        gm._pred  = _add_batch_dim(gm._pred.__func__).__get__(gm, GraspMolmo)
        gm._patched = True

    # 4 Predict best grasp
    K = np.asarray(intr.intrinsic_matrix)                  # 3×3 cam_K
    idx = gm.pred_grasp(np.asarray(color), pcd,
                        args.task, grasps, cam_K=K)        # Fix #2: pass cam_K

    # 5 Report result
    print(f"\nTask        : {args.task}")
    print(f"Chosen index : {idx}")
    print("6-DoF grasp  :", grasps[idx])

if __name__ == "__main__":
    main()