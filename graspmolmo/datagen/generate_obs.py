from concurrent.futures import ProcessPoolExecutor, wait, Future, CancelledError
import multiprocessing as mp
from typing import Any

if __name__ == "__main__":
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass

from io import BytesIO
import os
import pickle
from base64 import b64decode
from contextlib import contextmanager
import signal
import threading
import traceback

import h5py
import hydra
from omegaconf import DictConfig, OmegaConf
import yaml
import open3d as o3d

if os.environ.get("PYOPENGL_PLATFORM") is None:
    os.environ["PYOPENGL_PLATFORM"] = "egl"

import numpy as np
import pyrender
import pyrender.light
import trimesh

from graspmolmo.annotation import Annotation
from graspmolmo.utils import tqdm
from graspmolmo.datagen.datagen_utils import trimesh_scene_to_pyrender

@contextmanager
def block_signals(signals: list[int]):
    previous_blocked = signal.pthread_sigmask(signal.SIG_BLOCK, [])
    try:
        signal.pthread_sigmask(signal.SIG_BLOCK, signals)
        yield
    finally:
        signal.pthread_sigmask(signal.SIG_SETMASK, previous_blocked)

worker_id = mp.Value("i", 0)

def worker_init(cfg: dict[str, Any]):
    n_gpus = int(os.popen("nvidia-smi --list-gpus | wc -l").read())
    with worker_id.get_lock():
        gpu_id = worker_id.value % n_gpus
        worker_id.value += 1
    os.environ["EGL_DEVICE_ID"] = str(gpu_id)

    height, width = cfg["img_size"]
    renderer = pyrender.OffscreenRenderer(width, height)
    globals()["renderer"] = renderer
    globals()["cfg"] = cfg

def build_scene(data: dict[str, any]):
    glb_bytes = BytesIO(b64decode(data["glb"].encode("utf-8")))
    tr_scene: trimesh.Scene = trimesh.load(glb_bytes, file_type="glb")
    scene = trimesh_scene_to_pyrender(tr_scene)

    for light in data["lighting"]:
        light_type = getattr(pyrender.light, light["type"])
        light_args = light["args"]
        light_args["color"] = np.array(light_args["color"]) / 255.0
        light_node = pyrender.Node(light["args"]["name"], matrix=light["transform"], light=light_type(**light_args))
        scene.add_node(light_node)
    return scene

def set_camera(scene: pyrender.Scene, cam_K: np.ndarray, cam_pose: np.ndarray):
    cam = pyrender.camera.IntrinsicsCamera(
        fx=cam_K[0, 0],
        fy=cam_K[1, 1],
        cx=cam_K[0, 2],
        cy=cam_K[1, 2],
        name="camera",
    )
    cam_node = pyrender.Node(name="camera", camera=cam, matrix=cam_pose)
    for n in (scene.get_nodes(name=cam_node.name) or []):
        scene.remove_node(n)
    scene.add_node(cam_node)

    cam_light = pyrender.light.PointLight(intensity=2.0, name="camera_light")
    camera_light_node = pyrender.Node(name="camera_light", matrix=cam_pose, light=cam_light)
    for n in (scene.get_nodes(name=camera_light_node.name) or []):
        scene.remove_node(n)
    scene.add_node(camera_light_node)

def backproject(cam_K: np.ndarray, depth: np.ndarray):
    """
    Args:
        cam_K: camera intrinsic matrix (3, 3)
        depth: depth image (H, W)
    Returns:
        xyz: xyz coordinates of the points in the camera frame (H, W, 3)
    """
    height, width = depth.shape
    u, v = np.meshgrid(np.arange(width), np.arange(height), indexing="xy")
    uvd = np.stack((u, v, np.ones_like(u)), axis=-1).astype(np.float32)
    uvd *= np.expand_dims(depth, axis=-1)
    xyz = uvd @ np.expand_dims(np.linalg.inv(cam_K).T, axis=0)
    return xyz

def estimate_normals(xyz: np.ndarray):
    """
    Args:
        xyz: xyz coordinates of the points in the camera frame (N, H, W, 3)
    Returns:
        normals: normals of the points in the camera frame (N, H, W, 3)
    """
    batched_points = xyz.reshape(xyz.shape[0], -1, 3)  # (B, H * W, 3)
    batched_normals = np.zeros_like(batched_points)
    print(f"Estimating normals for {batched_points.shape[0]} views")
    for i, points in enumerate(batched_points):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.estimate_normals()
        normals = np.asarray(pcd.normals)
        batched_normals[i] = normals
    normals = batched_normals.reshape(xyz.shape)  # (B, H, W, 3)
    return normals

def generate_renders(renderer: pyrender.OffscreenRenderer, scene: pyrender.Scene):
    color, depth = renderer.render(scene, flags=pyrender.RenderFlags.SHADOWS_DIRECTIONAL)

    node_map = {}
    object_names: list[str] = []
    for i, node in enumerate(scene.mesh_nodes, start=1):
        if node.name:
            name = node.name
            if name.startswith("object_"):
                name = name[len("object_"):]
            name = name.replace("/geometry_0", "")
        else:
            name = f"node_{i}"
        node_map[node] = ((i >> 16) & 0xFF, (i >> 8) & 0xFF, i & 0xFF)
        object_names.append(name)
    seg_rgb, _ = renderer.render(scene, seg_node_map=node_map, flags=pyrender.RenderFlags.SEG)
    seg_rgb = seg_rgb.astype(np.uint32)
    seg: np.ndarray = (seg_rgb[..., 0] << 16) | (seg_rgb[..., 1] << 8) | seg_rgb[..., 2]

    in_view_names = []
    new_seg = np.zeros_like(seg)
    for i, name in enumerate(object_names, start=1):
        mask = seg == i
        if np.any(mask):
            in_view_names.append(name)
            new_seg[mask] = len(in_view_names)
    seg = new_seg

    return color, depth, seg, in_view_names

def render(out_dir: str, scene_dir: str):
    scene_id = os.path.basename(scene_dir)
    out_scene_file = f"{out_dir}/{scene_id}.hdf5"
    if os.path.isfile(out_scene_file):
        print(f"Skipping {scene_id} because it already has observations")
        return 0

    with open(f"{scene_dir}/scene.pkl", "rb") as f:
        scene_data = pickle.load(f)
    # maps annot_id to (annotation, grasp_pose, grasp_point)
    all_annotations: dict[str, tuple[Annotation, np.ndarray, np.ndarray]] = scene_data["annotations"]
    scene = build_scene(scene_data)

    cfg: dict[str, Any] = globals()["cfg"]
    renderer: pyrender.OffscreenRenderer = globals()["renderer"]
    renderer.viewport_height, renderer.viewport_width = scene_data["img_size"]

    view_observations: list[list[tuple[np.ndarray, np.ndarray, np.ndarray, Annotation, str]]] = []
    view_rgb: list[np.ndarray] = []
    view_xyz: list[np.ndarray] = []
    view_seg: list[np.ndarray] = []
    view_poses: list[np.ndarray] = []  # in standard camera axes conventions
    view_cam_params: list[np.ndarray] = []
    view_object_names: list[list[str]] = []
    for view in scene_data["views"]:
        cam_K = np.array(view["cam_K"])
        cam_pose_trimesh = np.array(view["cam_pose"])
        set_camera(scene, cam_K, cam_pose_trimesh)

        standard_to_trimesh_cam_trf = np.array([
            [1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, 1]
        ])
        cam_pose_standard = cam_pose_trimesh @ standard_to_trimesh_cam_trf
        view_poses.append(cam_pose_standard)
        view_cam_params.append(cam_K)

        color, depth, seg, object_names = generate_renders(renderer, scene)

        xyz = backproject(cam_K, depth).astype(np.float32)
        view_rgb.append(color)
        view_xyz.append(xyz)
        view_seg.append(seg)
        view_object_names.append(object_names)
        observations = []
        for annot_id in view["annotations_in_view"]:
            annot, grasp_pose, grasp_point = all_annotations[annot_id]
            grasp_pose_in_cam_frame = np.linalg.solve(cam_pose_standard, grasp_pose)
            grasp_point_in_cam_frame: np.ndarray = cam_pose_standard[:3, :3].T @ (grasp_point - cam_pose_standard[:3, 3])  # closed form for rigid inverse
            grasp_point_px: np.ndarray = cam_K @ grasp_point_in_cam_frame
            grasp_point_px = grasp_point_px[:2] / grasp_point_px[2]
            observations.append((grasp_pose_in_cam_frame, grasp_point_in_cam_frame, grasp_point_px, annot, annot_id))
        view_observations.append(observations)

    if cfg["estimate_normals"]:
        view_normals = estimate_normals(np.stack(view_xyz, axis=0))
    else:
        view_normals = [None] * len(view_xyz)

    n_observations = 0
    with block_signals([signal.SIGINT]):
        with h5py.File(out_scene_file, "w") as f:
            for view_idx in range(len(view_rgb)):
                view_group = f.create_group(f"view_{view_idx}")

                rgb_ds = view_group.create_dataset("rgb", data=view_rgb[view_idx], compression="gzip")
                rgb_ds.attrs['CLASS'] = np.string_('IMAGE')
                rgb_ds.attrs['IMAGE_VERSION'] = np.string_('1.2')
                rgb_ds.attrs['IMAGE_SUBCLASS'] = np.string_('IMAGE_TRUECOLOR')
                rgb_ds.attrs['INTERLACE_MODE'] = np.string_('INTERLACE_PIXEL')

                view_group.create_dataset("xyz", data=view_xyz[view_idx], compression="gzip")
                view_group.create_dataset("seg", data=view_seg[view_idx], compression="gzip")
                view_group["object_names"] = view_object_names[view_idx]
                if view_normals[view_idx] is not None:
                    view_group.create_dataset("normals", data=view_normals[view_idx], compression="gzip")
                view_group.create_dataset("view_pose", data=view_poses[view_idx], compression="gzip")
                view_group.create_dataset("cam_params", data=view_cam_params[view_idx], compression="gzip")

                for obs_idx, (grasp_pose, grasp_point, grasp_point_px, annot, annot_id) in enumerate(view_observations[view_idx]):
                    obs_group = view_group.create_group(f"obs_{obs_idx}")
                    obs_group.create_dataset("grasp_pose", data=grasp_pose, compression="gzip")
                    obs_group.create_dataset("grasp_point", data=grasp_point, compression="gzip")
                    obs_group.create_dataset("grasp_point_px", data=grasp_point_px, compression="gzip")

                    annot_str = yaml.dump({
                        "annotation_id": annot_id,
                        "grasp_description": annot.grasp_description,
                        "object_description": annot.obj_description,
                        "object_category": annot.obj.object_category,
                        "object_id": annot.obj.object_id,
                        "grasp_id": annot.grasp_id
                    })
                    obs_group.create_dataset("annot", data=annot_str.encode("utf-8"))
                    n_observations += 1
    return n_observations

def get_desc(n_generated_obs: int):
    return f"Processing scenes ({n_generated_obs} new observations)"

@hydra.main(version_base=None, config_path="../../config", config_name="obs_gen.yaml")
def main(cfg: DictConfig):
    if missing_keys := OmegaConf.missing_keys(cfg):
        raise ValueError(f"Missing keys: {missing_keys}")

    in_dir = cfg["scene_dir"]
    out_dir = cfg["out_dir"]
    os.makedirs(out_dir, exist_ok=True)

    nproc = cfg["n_proc"] or os.cpu_count()
    generated_observations = 0
    gen_obs_lock = threading.Lock()
    submit_semaphore = threading.Semaphore(4 * nproc)
    with ProcessPoolExecutor(
        max_workers=nproc,
        initializer=worker_init,
        initargs=(OmegaConf.to_container(cfg),)
    ) as executor:
        scenes: set[str] = set(fn for fn in os.listdir(in_dir) if os.path.isdir(f"{in_dir}/{fn}"))
        processed_scenes: set[str] = set(fn.split(".")[0] for fn in os.listdir(out_dir) if fn.endswith(".hdf5"))
        to_process = list(scenes - processed_scenes)
        with tqdm(total=len(scenes), initial=len(processed_scenes), desc=get_desc(generated_observations)) as pbar:
            def on_job_done(future: Future):
                nonlocal generated_observations
                submit_semaphore.release()
                pbar.update(1)
                with gen_obs_lock:
                    try:
                        generated_observations += future.result()
                    except CancelledError:
                        pass
                    except:
                        traceback.print_exc()
                        executor.shutdown(wait=False, cancel_futures=True)
                    pbar.set_description(get_desc(generated_observations))

            futures: list[Future] = []
            for fn in to_process:
                submit_semaphore.acquire()
                future = executor.submit(render, out_dir, f"{in_dir}/{fn}")
                future.add_done_callback(on_job_done)
                futures.append(future)
            wait(futures, return_when="FIRST_EXCEPTION")

if __name__ == "__main__":
    main()
