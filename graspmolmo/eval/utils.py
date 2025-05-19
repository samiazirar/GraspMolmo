import os
import urllib.request
import hashlib
import glob
from typing import Any

import numpy as np
from PIL import Image

def download(url: str, filename: str):
    url_hash = hashlib.sha256(url.encode()).hexdigest()
    dl_path = f"/tmp/semantic-grasping-cache/{url_hash}/{filename}"
    if not os.path.exists(dl_path):
        os.makedirs(os.path.dirname(dl_path), exist_ok=True)
        urllib.request.urlretrieve(url, dl_path)
    return dl_path


def img_to_pc(rgb: np.ndarray, depth: np.ndarray, cam_info: np.ndarray, mask: np.ndarray | None = None):
    h, w = rgb.shape[:2]
    u, v = np.meshgrid(np.arange(w), np.arange(h), indexing="xy")
    depth_mask = (depth > 0)
    if mask is None:
        mask = np.ones_like(depth, dtype=bool)
    uvd = np.stack((u, v, np.ones_like(u)), axis=-1).astype(np.float32)
    uvd *= np.expand_dims(depth, axis=-1)
    uvd = uvd[depth_mask & mask]
    xyz = np.linalg.solve(cam_info, uvd.T).T
    return np.concatenate([xyz, rgb[depth_mask & mask]], axis=-1)

def pc_to_depth(xyz: np.ndarray, cam_info: np.ndarray, height: int, width: int):
    uvd = xyz @ cam_info.T
    uvd /= uvd[:, 2:]
    uv = uvd[:, :2].astype(np.int32)

    img = np.zeros((height, width), dtype=np.float32)
    mask = (uv[:, 0] >= 0) & (uv[:, 0] < width) & (uv[:, 1] >= 0) & (uv[:, 1] < height)
    uv = uv[mask]
    xyz = xyz[mask]
    img[uv[:, 1], uv[:, 0]] = xyz[:, 2]
    return img

class TaskGraspScanLibrary:
    def __init__(self, tg_dir: str):
        assert os.path.isdir(tg_dir), "TaskGrasp directory does not exist"
        self.tg_dir = tg_dir
        self.rgb_paths = sorted(glob.glob(os.path.join(tg_dir, "**", "[0-9]_color.png"), recursive=True))

    def __len__(self):
        return len(self.rgb_paths)

    def get(self, object_id: str, scan_id: int):
        for i, rgb_path in enumerate(self.rgb_paths):
            dirname = os.path.dirname(rgb_path)
            if os.path.basename(dirname) == object_id and os.path.basename(rgb_path).split("_", 1)[0] == str(scan_id):
                return self[i]
        raise ValueError(f"Scan {object_id}_{scan_id} not found")

    def get_views(self, object_id: str) -> list[int]:
        views = []
        for rgb_path in self.rgb_paths:
            dirname = os.path.dirname(rgb_path)
            if os.path.basename(dirname) == object_id:
                views.append(int(os.path.basename(rgb_path).split("_", 1)[0]))
        return views

    def __contains__(self, item: tuple[str, int] | tuple[str, int, str]):
        assert len(item) == 2 or len(item) == 3, "Item must be a tuple of (object_id, scan_id) or (object_id, scan_id, key)"
        object_id, scan_id = item[:2]
        for rgb_path in self.rgb_paths:
            dirname = os.path.dirname(rgb_path)
            filename = os.path.basename(rgb_path)
            if os.path.basename(dirname) == object_id and filename.split("_", 1)[0] == str(scan_id):
                if len(item) == 2:
                    return True
                key = item[2]
                return os.path.isfile(rgb_path[:-len("_color.png")] + key)
        return False

    def __getitem__(self, idx: int) -> dict[str, Any]:
        dirname = os.path.dirname(self.rgb_paths[idx])
        object_id = os.path.basename(dirname)
        object_name = object_id.split("_", 1)[1].replace("_", " ")
        rgb_path = self.rgb_paths[idx]

        scan_id = os.path.basename(rgb_path).split("_", 1)[0]

        rgb = Image.open(rgb_path)
        rgb_array = np.asarray(rgb)
        depth = np.load(rgb_path[:-len("_color.png")] + "_depth.npy") / 1000.0
        cam_params = np.load(rgb_path[:-len("_color.png")] + "_camerainfo.npy")

        pc = img_to_pc(rgb_array, depth, cam_params)
        pc[:, 0] += 0.021
        pc[:, 1] -= 0.002
        corr_depth = pc_to_depth(pc[:,:3], cam_params, rgb_array.shape[0], rgb_array.shape[1])

        fused_pc = np.load(os.path.join(dirname, "fused_pc_clean.npy"))
        fused_pc[:,:3] -= np.mean(fused_pc[:,:3], axis=0)

        if os.path.exists(rgb_path[:-len("_color.png")] + "_registered_grasps.npy"):
            registered_grasps = np.load(rgb_path[:-len("_color.png")] + "_registered_grasps.npy")
        else:
            registered_grasps = None

        if os.path.exists(rgb_path[:-len("_color.png")] + "_segmented_pc.npy"):
            segmented_pc = np.load(rgb_path[:-len("_color.png")] + "_segmented_pc.npy")
        else:
            segmented_pc = None

        fused_grasps = []
        for grasp_dirname in sorted(os.listdir(os.path.join(dirname, "grasps")), key=int):
            grasp = np.load(os.path.join(dirname, "grasps", grasp_dirname, "grasp.npy"))
            fused_grasps.append(grasp)
        fused_grasps = np.array(fused_grasps)

        grasp_trf = np.array([
            [0, 0, 1, -0.09],
            [-1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, 0, 1],
        ])
        fused_grasps = fused_grasps @ grasp_trf[None]

        return {
            "object_id": object_id,
            "scan_id": scan_id,
            "object_name": object_name,
            "rgb": rgb,
            "depth": corr_depth,
            "cam_params": cam_params,
            "fused_pc": fused_pc,
            "fused_grasps": fused_grasps,
            "registered_grasps": registered_grasps,
            "segmented_pc": segmented_pc,
        }

    def __iter__(self):
        for idx in range(len(self)):
            yield self[idx]
