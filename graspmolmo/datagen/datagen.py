from concurrent.futures import ProcessPoolExecutor, Future, as_completed
import json
import os
import pickle
import time
import uuid
import base64
import shutil

# pyrender spawns a lot of OMP threads, limiting to 1 significantly reduces overhead
if os.environ.get("OMP_NUM_THREADS") is None:
    os.environ["OMP_NUM_THREADS"] = "1"

import boto3
import numpy as np
import trimesh
from trimesh import transformations as tra
from pydantic import BaseModel
from itertools import compress
from PIL import Image, ImageColor
import yaml
import hydra
from omegaconf import DictConfig, OmegaConf

from graspmolmo.datagen.datagen_utils import (
    kelvin_to_rgb,
    MeshLibrary,
    look_at_rot,
    random_delta_rot,
    construct_cam_K,
    rejection_sample,
    RejectionSampleError,
    not_none,
    set_exit_event
)
from graspmolmo.annotation import Annotation
from graspmolmo.utils import list_s3_files, tqdm

from acronym_tools import create_gripper_marker

import scene_synthesizer as ss
from scene_synthesizer.utils import PositionIteratorUniform


ANNOTATIONS_DIR = "annotations_filtered"

class DatagenConfig(BaseModel):
    min_wall_dist: float = 2.0
    room_width_range: tuple[float, float] = (4.0, 10.0)  # meters
    room_depth_range: tuple[float, float] = (4.0, 10.0)  # meters

    cam_dfov_range: tuple[float, float] = (60.0, 90.0)  # degrees
    cam_dist_range: tuple[float, float] = (0.7, 1.3)  # meters
    cam_pitch_perturb: float = 0.02  # fraction of YFOV
    cam_yaw_perturb: float = 0.05  # fraction of XFOV
    cam_roll_perturb: float = np.pi/8  # radians
    cam_elevation_range: tuple[float, float] = (np.pi/8, np.pi/3)  # radians
    img_size: tuple[int, int] = (480, 640)  # (height, width)

    n_views: int = 10
    min_annots_per_view: int = 1
    n_objects_range: tuple[int, int] = (4, 6)
    n_background_range: tuple[int, int] = (4, 6)
    max_grasp_dist: float = 2.0  # meters

    color_temp_range: tuple[float, float] = (2000, 10000)  # K
    light_intensity_range: tuple[float, float] = (10, 40)  # lux
    light_azimuth_range: tuple[float, float] = (0, 2 * np.pi)  # radians
    light_inclination_range: tuple[float, float] = (0, np.pi/3)  # radians

SUPPORT_CATEGORIES = [
    # "Bookcase",
    "Table"
]

GRASP_LOCAL_POINTS = np.array([
    [0.041, 0, 0.066],
    [0.041, 0, 0.112],
    [0, 0, 0.066],
    [-0.041, 0, 0.112],
    [-0.041, 0, 0.066]
])

DATA_ROOT = os.path.join(os.path.dirname(__file__), "../../data")

with open(os.path.join(DATA_ROOT, "wall_colors.json"), "r") as f:
    WALL_COLORS = json.load(f)

def homogenize(arr: np.ndarray):
    if arr.ndim == 1:
        return np.concatenate([arr, np.ones(1)])
    else:
        return np.concatenate([arr, np.ones((len(arr), 1))], axis=-1)

def create_plane(width: float, depth: float, center: np.ndarray, normal: np.ndarray):
    corners = [
        (-width / 2.0, -depth / 2.0, 0),
        (width / 2.0, -depth / 2.0, 0),
        (-width / 2.0, depth / 2.0, 0),
        (width / 2.0, depth / 2.0, 0),
    ]
    transform = trimesh.geometry.align_vectors((0, 0, 1), normal)
    transform[:3, 3] = center
    vertices = tra.transform_points(corners, transform)
    faces = [[0, 1, 3, 2]]
    plane = trimesh.Trimesh(vertices=vertices, faces=faces)
    return plane

def generate_floor_and_walls(scene: ss.Scene, datagen_cfg: DatagenConfig):
    scene_bounds = scene.get_bounds()
    min_width, min_depth = scene_bounds[1, :2] - scene_bounds[0, :2] + datagen_cfg.min_wall_dist
    width = max(min_width, np.random.uniform(*datagen_cfg.room_width_range))
    depth = max(min_depth, np.random.uniform(*datagen_cfg.room_depth_range))

    center_lo = scene_bounds[1,:2] - np.array([width, depth])/2
    center_hi = scene_bounds[0,:2] + np.array([width, depth])/2
    center = np.append(np.random.uniform(center_lo, center_hi), 0)

    floor_plane = create_plane(width+0.2, depth+0.2, center, (0, 0, 1))
    uv = np.array([[0,0], [1,0], [0,1], [1,1]]) * np.array([width, depth])
    texture = Image.open(os.path.join(DATA_ROOT, "floor_textures", np.random.choice(os.listdir(os.path.join(DATA_ROOT, "floor_textures")))))
    floor_plane.visual = trimesh.visual.TextureVisuals(
        uv=uv,
        image=texture
    )
    scene.add_object(ss.TrimeshAsset(floor_plane), "floor")

    wall_height = scene_bounds[1, 2] + 2.0
    wall_color = ImageColor.getrgb(np.random.choice(WALL_COLORS))
    if len(wall_color) == 3:
        wall_color = (*wall_color, 255)

    def make_wall(sidelength, center, normal, name):
        plane = create_plane(wall_height, sidelength, center, normal)
        plane.visual.vertex_colors = wall_color
        scene.add_object(ss.TrimeshAsset(plane), name)
    make_wall(depth, center + np.array([-width/2, 0, wall_height/2]), (1, 0, 0), "x_pos_wall")
    make_wall(depth, center + np.array([width/2, 0, wall_height/2]), (-1, 0, 0), "x_neg_wall")
    make_wall(width, center + np.array([0, -depth/2, wall_height/2]), (0, 1, 0), "y_pos_wall")
    make_wall(width, center + np.array([0, depth/2, wall_height/2]), (0, -1, 0), "y_neg_wall")

def generate_lighting(scene: ss.Scene, datagen_cfg: DatagenConfig) -> list[dict]:
    light_temp = np.random.uniform(*datagen_cfg.color_temp_range)
    light_intensity = np.random.uniform(*datagen_cfg.light_intensity_range)
    light_azimuth = np.random.uniform(*datagen_cfg.light_azimuth_range)
    light_inclination = np.random.uniform(*datagen_cfg.light_inclination_range)
    light_trf = np.eye(4)
    light_direction = np.array([
        np.sin(light_inclination) * np.cos(light_azimuth),
        np.sin(light_inclination) * np.sin(light_azimuth),
        np.cos(light_inclination)
    ])  # opposite of direction light is pointing
    light_trf[:3, :3] = look_at_rot(light_direction, np.zeros(3))
    lights = [
        {
            "type": "DirectionalLight",
            "args": {
                "name": "light",
                "color": kelvin_to_rgb(light_temp),
                "intensity": light_intensity,
            },
            "transform": light_trf
        }
    ]
    return lights

def get_visible_points(scene_mesh: trimesh.Trimesh, cam_pose: np.ndarray, points: np.ndarray, eps: float=1e-5):
    """
    Check if points are visible from the camera. "Visible" means that a ray from the camera to the point
    does not intersect the scene mesh before hitting the point. The point may or may not be on the scene mesh.
    Args:
        scene_mesh: trimesh.Trimesh - scene mesh
        cam_pose: (4, 4) camera pose
        points: (N, 3) points in scene frame
    Returns:
        mask: (N,) mask of the same length as points, where each element is True if the point is visible.
    """
    ray_origins = np.tile(cam_pose[:3, 3], (len(points), 1))
    ray_directions = points - ray_origins
    ray_directions /= np.linalg.norm(ray_directions, axis=1, keepdims=True)

    visible = np.zeros(len(points), dtype=bool)

    intersect_points, ray_idxs, _ = scene_mesh.ray.intersects_location(ray_origins, ray_directions, multiple_hits=False)
    hit_dists = np.linalg.norm(intersect_points - ray_origins[ray_idxs], axis=1)
    point_dists = np.linalg.norm(points[ray_idxs] - ray_origins[ray_idxs], axis=1)
    visible[ray_idxs] = ~(hit_dists < point_dists - eps)

    return visible

def in_frustum_annotations(datagen_cfg: DatagenConfig, cam_K: np.ndarray, cam_pose: np.ndarray, grasps: np.ndarray):
    """
    Check if grasps are within the camera frustum.
    Args:
        datagen_cfg: DatagenConfig
        cam_K: (3, 3) camera intrinsic matrix
        cam_pose: (4, 4) camera pose
        grasps: (N, 4, 4) grasps in scene frame
    Returns:
        mask: (N,) mask of the same length as grasps, where each element is True if the grasp is within the camera frustum.
    """
    # grasps is (N, 4, 4) poses in scene frame
    trf = np.eye(4)
    trf[[1,2], [1,2]] = -1  # flip y and z axes, since for trimesh camera -z is forward
    grasps_cam_frame = trf @ np.linalg.inv(cam_pose)[None] @ grasps
    grasp_points_cam_frame = homogenize(GRASP_LOCAL_POINTS)[None] @ grasps_cam_frame[:, :-1].transpose(0, 2, 1)
    grasp_points_img = grasp_points_cam_frame @ cam_K.T
    grasp_points_img = grasp_points_img[..., :2] / grasp_points_img[..., 2:]

    close_mask = np.linalg.norm(grasps_cam_frame[:, :3, 3], axis=-1) <= datagen_cfg.max_grasp_dist

    in_front_mask = np.all(grasp_points_cam_frame[..., 2] > 0, axis=-1)

    img_h, img_w = datagen_cfg.img_size
    in_bounds_mask = np.all((grasp_points_img[..., 0] >= 0) & \
        (grasp_points_img[..., 0] < img_w) & \
        (grasp_points_img[..., 1] >= 0) & \
        (grasp_points_img[..., 1] < img_h), axis=-1)

    return close_mask & in_front_mask & in_bounds_mask

def visible_annotations(scene: ss.Scene, cam_pose: np.ndarray, grasps: np.ndarray, grasp_points: np.ndarray):
    """
    Check if grasps are visible from the camera, i.e. the grasp is not occluded by an object in the scene.
    Args:
        scene: ss.Scene
        cam_pose: (4, 4) camera pose
        grasps: (N, 4, 4) grasps in scene frame
        grasp_points: (N, 3) grasp points in scene frame
    Returns:
        mask: (N,) mask of the same length as grasps, where each element is True if the grasp is visible.
    """
    scene_mesh: trimesh.Trimesh = scene.scene.to_mesh()

    gripper_points = homogenize(GRASP_LOCAL_POINTS)[None] @ grasps[:, :-1].transpose(0, 2, 1)
    gripper_points = gripper_points.reshape(-1, 3)  # (N*5, 3)

    raycast_points = np.concatenate([gripper_points, grasp_points], axis=0)  # (N*5+N, 3)
    visible_mask = get_visible_points(scene_mesh, cam_pose, raycast_points)

    gripper_points_visible = visible_mask[:len(gripper_points)]  # (N*5,)
    grasp_points_visible = visible_mask[len(gripper_points):]  # (N,)

    grasps_visible = np.sum(gripper_points_visible.reshape(len(grasps), len(GRASP_LOCAL_POINTS)), axis=1) >= 3

    return grasps_visible & grasp_points_visible

def noncolliding_annotations(scene: ss.Scene, annots: list[Annotation], grasps: np.ndarray, collision_cache: dict[tuple[str, str, int], bool]):
    """
    Check if grasps are noncolliding, i.e. the grasp is not in collision with any other object in the scene.
    Returns a mask of the same length as grasps, where each element is True if the grasp is noncolliding.
    """
    # grasps is (N, 4, 4) poses in scene frame
    gripper_manager = trimesh.collision.CollisionManager()
    noncolliding = np.ones(len(grasps), dtype=bool)
    cache_miss_idxs = []
    for i, (annot, grasp) in enumerate(zip(annots, grasps)):
        if (annot.obj.object_category, annot.obj.object_id, annot.grasp_id) in collision_cache:
            noncolliding[i] = collision_cache[(annot.obj.object_category, annot.obj.object_id, annot.grasp_id)]
            continue
        gripper_manager.add_object(f"gripper_{i}", create_gripper_marker(), transform=grasp)
        cache_miss_idxs.append(i)

    if len(cache_miss_idxs) > 0:
        _, pairs = scene.in_collision_other(gripper_manager, return_names=True)
        for pair in pairs:
            if (name := next(filter(lambda x: x.startswith("gripper_"), pair), None)) is not None:
                idx = int(name.split("_")[-1])
                noncolliding[idx] = False

        for i in cache_miss_idxs:
            collision_cache[(annots[i].obj.object_category, annots[i].obj.object_id, annots[i].grasp_id)] = noncolliding[i]
    return noncolliding

def point_on_support(scene: ss.Scene):
    surfaces = scene.support_generator(sampling_fn=lambda x: x)
    weights = [s.polygon.area for s in surfaces]
    weights = np.array(weights) / np.sum(weights)
    surface = surfaces[np.random.choice(len(surfaces), p=weights)]

    it = PositionIteratorUniform()
    x, y = next(it(surface))[0]

    obj_pose = scene.get_transform(surface.node_name)
    return obj_pose[:-1] @ surface.transform @ np.array([x, y, 0, 1])

def sample_camera_pose(
    scene: ss.Scene,
    datagen_cfg: DatagenConfig,
    cam_dfov: float,
    in_scene_annotations: list[Annotation],
    annotation_grasps: np.ndarray,
    annotation_points: np.ndarray,
    collision_cache: dict[tuple[str, str, int], bool]
):
    img_h, img_w = datagen_cfg.img_size
    cam_K = construct_cam_K(img_w, img_h, cam_dfov)
    cam_xfov = 2 * np.arctan(img_w / (2 * cam_K[0, 0]))
    cam_yfov = 2 * np.arctan(img_h / (2 * cam_K[1, 1]))

    lookat_pos = point_on_support(scene)

    cam_dist = np.random.uniform(*datagen_cfg.cam_dist_range)
    inclination = np.pi/2 - np.random.uniform(*datagen_cfg.cam_elevation_range)
    azimuth = np.random.rand() * 2 * np.pi
    cam_pose = np.eye(4)
    cam_pose[:3, 3] = np.array([
        cam_dist * np.sin(inclination) * np.cos(azimuth),
        cam_dist * np.sin(inclination) * np.sin(azimuth),
        cam_dist * np.cos(inclination)
    ]) + lookat_pos
    cam_pose[:3, :3] = \
        look_at_rot(cam_pose[:3, 3], lookat_pos) @ \
        random_delta_rot(
            datagen_cfg.cam_roll_perturb,
            datagen_cfg.cam_pitch_perturb * np.radians(cam_yfov),
            datagen_cfg.cam_yaw_perturb * np.radians(cam_xfov)
        )

    in_view_annots, in_view_grasps, in_view_grasp_points = get_annotations_in_view(
        scene,
        datagen_cfg,
        cam_K,
        cam_pose,
        in_scene_annotations,
        annotation_grasps,
        annotation_points,
        collision_cache
    )
    if len(in_view_annots) < datagen_cfg.min_annots_per_view:
        return None

    return cam_K, cam_pose, in_view_annots, in_view_grasps, in_view_grasp_points


def sample_arrangement(
    datagen_cfg: DatagenConfig,
    object_keys: list[tuple[str, str]],
    object_meshes: list[trimesh.Trimesh],
    background_meshes: list[trimesh.Trimesh],
    support_mesh: trimesh.Trimesh,
    object_library: MeshLibrary,
    annotations: list[Annotation]
):
    scene = ss.Scene()
    scene.add_object(ss.TrimeshAsset(support_mesh, origin=("centroid", "centroid", "bottom")), "support")
    with np.errstate(divide="ignore", invalid="ignore"):
        support_surfaces = scene.label_support("support", min_area=0.05)
    if len(support_surfaces) == 0:
        return None

    generate_floor_and_walls(scene, datagen_cfg)

    objs_placed = 0
    for (category, obj_id), obj in zip(object_keys, object_meshes):
        asset = ss.TrimeshAsset(obj, origin=("centroid", "centroid", "bottom"))
        objs_placed += scene.place_object(f"object_{category}_{obj_id}", asset, "support")
    if objs_placed <= 1:
        return None
    for i, obj in enumerate(background_meshes):
        asset = ss.TrimeshAsset(obj, origin=("centroid", "centroid", "bottom"))
        scene.place_object(f"background_{i}", asset, "support")

    grasps_dict: dict[tuple[str, str], np.ndarray] = {}  # maps object in scene to its grasps in centroid frame
    grasp_points_dict: dict[tuple[str, str], np.ndarray] = {}  # maps object in scene to its grasp points in centroid frame
    for name in scene.get_object_names():
        assert isinstance(name, str)
        if not name.startswith("object_"):
            continue
        _, cat, obj_id = name.split("_", 2)
        grasps_dict[(cat, obj_id)] = object_library.grasps(cat, obj_id)[0]
        grasp_points_dict[(cat, obj_id)] = object_library.grasp_points(cat, obj_id)

    in_scene_annotations: list[Annotation] = []
    annotation_grasps = []  # grasps in scene frame
    annotation_points = []  # points in scene frame
    for annot in annotations:
        if (annot.obj.object_category, annot.obj.object_id) in grasps_dict:
            obj_name = f"object_{annot.obj.object_category}_{annot.obj.object_id}"
            in_scene_annotations.append(annot)
            grasp_local = grasps_dict[(annot.obj.object_category, annot.obj.object_id)][annot.grasp_id].copy()
            grasp_point_local = grasp_points_dict[(annot.obj.object_category, annot.obj.object_id)][annot.grasp_id].copy()
            geom_names = scene.get_geometry_names(obj_name)
            assert len(geom_names) == 1
            # the grasp is in the centroid frame of the object, so offset by centroid in local frame
            centroid = scene.get_centroid(geom_names[0], obj_name)
            grasp_local[:3, 3] += centroid
            grasp_point_local += centroid
            obj_trf = scene.get_transform(obj_name)
            grasp = obj_trf @ grasp_local  # transform to scene frame
            grasp_point = obj_trf[:-1] @ homogenize(grasp_point_local)  # transform to scene frame
            annotation_grasps.append(grasp)
            annotation_points.append(grasp_point)

    annotation_grasps = np.array(annotation_grasps)
    annotation_points = np.array(annotation_points)
    collision_cache: dict[tuple[str, str, int], bool] = {}  # (category, obj_id, grasp_id) -> is colliding

    views: list[tuple[np.ndarray, np.ndarray]] = []
    annots_in_scene: dict[str, tuple[Annotation, np.ndarray, np.ndarray]] = {}  # annotation_id -> (annotation, grasp, grasp_point)
    annots_per_view: list[list[str]] = []
    try:
        for i in range(datagen_cfg.n_views):
            cam_dfov = np.random.uniform(*datagen_cfg.cam_dfov_range)
            cam_K, cam_pose, in_view_annots, in_view_grasps, in_view_grasp_points = rejection_sample(
                lambda: sample_camera_pose(scene, datagen_cfg, cam_dfov, in_scene_annotations, annotation_grasps, annotation_points, collision_cache),
                not_none,
                100
            )
            views.append([cam_K, cam_pose])
            annots_per_view.append([])
            for annot, grasp, grasp_point in zip(in_view_annots, in_view_grasps, in_view_grasp_points):
                annot_id = f"{annot.obj.object_category}_{annot.obj.object_id}_{annot.grasp_id}"
                annots_in_scene[annot_id] = (annot, grasp, grasp_point)
                annots_per_view[-1].append(annot_id)
    except RejectionSampleError:
        return None

    return scene, views, annots_in_scene, annots_per_view

def sample_scene(datagen_cfg: DatagenConfig, annotations: list[Annotation], object_library: MeshLibrary, background_library: MeshLibrary, support_library: MeshLibrary):
    n_objects = min(np.random.randint(*datagen_cfg.n_objects_range), len(object_library.categories()))
    if datagen_cfg.n_background_range[1] > 0:
        n_background = min(np.random.randint(*datagen_cfg.n_background_range), len(background_library.categories()))
    else:
        n_background = 0

    object_keys, object_meshes = object_library.sample(n_objects)
    background_keys, background_meshes = background_library.sample(n_background, replace=True)
    support_key, support_mesh = support_library.sample()

    try:
        return rejection_sample(
            lambda: sample_arrangement(datagen_cfg, object_keys, object_meshes, background_meshes, support_mesh, object_library, annotations),
            not_none,
            10
        )
    except RejectionSampleError:
        return None

def get_annotations_in_view(
    scene: ss.Scene,
    datagen_cfg: DatagenConfig,
    cam_K: np.ndarray,
    cam_pose: np.ndarray,
    in_scene_annotations: list[Annotation],
    annotation_grasps: np.ndarray,
    annotation_points: np.ndarray,
    collision_cache: dict[tuple[str, str, int], bool]
):
    """
    Returns:
        in_view_annots: list[Annotation] - N annotations in view
        in_view_grasps: np.ndarray - (N, 4, 4) grasps in scene frame
        in_view_grasp_points: np.ndarray - (N, 3) grasp points in scene frame
    """
    in_view_annots = in_scene_annotations
    in_view_grasps = annotation_grasps
    in_view_grasp_points = annotation_points
    for mask_fn in [
        lambda grasps: in_frustum_annotations(datagen_cfg, cam_K, cam_pose, grasps),
        lambda grasps: noncolliding_annotations(scene, in_scene_annotations, grasps, collision_cache),
        lambda grasps: visible_annotations(scene, cam_pose, grasps, in_view_grasp_points)
    ]:
        mask = mask_fn(in_view_grasps)
        in_view_annots = list(compress(in_view_annots, mask))
        in_view_grasps = in_view_grasps[mask]
        in_view_grasp_points = in_view_grasp_points[mask]
        if not np.any(mask):
            break

    return in_view_annots, in_view_grasps, in_view_grasp_points

def generate_scene(datagen_cfg: DatagenConfig, annotations: list[Annotation], object_library: MeshLibrary, background_library: MeshLibrary, support_library: MeshLibrary):
    scene, views, annots_in_scene, annots_per_view = rejection_sample(
        lambda: sample_scene(datagen_cfg, annotations, object_library, background_library, support_library),
        not_none,
        -1
    )
    lighting = generate_lighting(scene, datagen_cfg)

    glb_bytes: bytes = scene.export(file_type="glb")

    data = {
        "annotations": annots_in_scene,
        "views": [],
        "lighting": lighting,
        "glb": base64.b64encode(glb_bytes).decode("utf-8"),
        "img_size": datagen_cfg.img_size
    }
    for (cam_K, cam_pose), annots in zip(views, annots_per_view):
        data["views"].append({
            "cam_K": cam_K,
            "cam_pose": cam_pose,
            "annotations_in_view": annots
        })
    return data, scene

def procgen_init(data_dir: str, blacklist_file: str | None, split_file: str | None):
    if blacklist_file:
        with open(blacklist_file, "r") as f:
            blacklist = set(f.read().strip().splitlines())
    else:
        blacklist = set()

    if split_file:
        with open(split_file, "r") as f:
            split: dict[str, list[str]] = json.load(f)
        split = {k: set(v) for k, v in split.items()}
    else:
        split = None

    annotations: list[Annotation] = []
    for annot_fn in os.listdir(ANNOTATIONS_DIR):
        with open(f"{ANNOTATIONS_DIR}/{annot_fn}", "r") as f:
            annotations.append(Annotation.model_validate_json(f.read()))

    annotated_instances: dict[str, set[str]] = {}
    for annot in annotations:
        obj_category, obj_id = annot.obj.object_category, annot.obj.object_id
        if f"{obj_category}_{obj_id}" in blacklist or (split is not None and (obj_category not in split or obj_id not in split[obj_category])):
            continue
        if obj_category not in annotated_instances:
            annotated_instances[obj_category] = set()
        annotated_instances[obj_category].add(obj_id)

    globals()["annotations"] = annotations
    globals()["object_library"] = MeshLibrary(data_dir, annotated_instances)
    all_object_categories = open("all_categories.txt").read().splitlines()
    background_categories = [cat for cat in all_object_categories if cat not in annotated_instances]
    globals()["background_library"] = MeshLibrary.from_categories(data_dir, background_categories)
    globals()["support_library"] = MeshLibrary.from_categories(data_dir, SUPPORT_CATEGORIES, load_kwargs={"scale": 0.025})

    import threading
    def exit_if_orphaned():
        import multiprocessing
        multiprocessing.parent_process().join()  # wait for parent process to die first; may never happen
        print("Orphaned process detected, exiting")
        os._exit(-1)
    threading.Thread(target=exit_if_orphaned, daemon=True).start()

def procgen_worker(datagen_cfg: DatagenConfig, out_dir: str):
    data, scene = generate_scene(
        datagen_cfg,
        globals()["annotations"],
        globals()["object_library"],
        globals()["background_library"],
        globals()["support_library"]
    )
    scene_id = uuid.uuid4().hex
    os.mkdir(f"{out_dir}/{scene_id}")
    with open(f"{out_dir}/{scene_id}/scene.pkl", "wb") as f:
        pickle.dump(data, f)
    objects_in_scene = [obj for obj in scene.get_object_names() if obj.startswith("object_")]
    with open(f"{out_dir}/{scene_id}/metadata.yaml", "w") as f:
        yaml.dump({
            "objects": objects_in_scene,
        }, f)

def fetch_annotations(hydra_cfg: DictConfig):
    os.makedirs(ANNOTATIONS_DIR, exist_ok=True)
    s3 = None
    for source in hydra_cfg["annotation_sources"]:
        match source["type"]:
            case "s3":
                if s3 is None:
                    s3 = boto3.client("s3")
                bucket_name = source["params"]["bucket_name"]
                annot_prefix = source["params"]["annotation_prefix"]
                annot_files = list_s3_files(s3, bucket_name, annot_prefix)
                for annot_file in tqdm(annot_files, desc="Downloading annotations", disable=len(annot_files) == 0):
                    s3.download_file(bucket_name, annot_file, f"{ANNOTATIONS_DIR}/{os.path.basename(annot_file)}")
            case "directory":
                annot_dir = source["params"]["dir"]
                assert os.path.isdir(annot_dir), f"Annotation directory doesn't exist: {annot_dir}"
                for annot_file in tqdm(os.listdir(annot_dir), desc="Copying annotations", disable=len(annot_dir) == 0):
                    shutil.copy(f"{annot_dir}/{annot_file}", f"{ANNOTATIONS_DIR}/{annot_file}")
            case _:
                raise ValueError(f"Unknown annotation source type: {source['type']}")

@hydra.main(version_base=None, config_path="../../config", config_name="scene_gen.yaml")
def main(hydra_cfg: DictConfig):
    if missing_keys := OmegaConf.missing_keys(hydra_cfg):
        raise ValueError(f"Missing keys: {missing_keys}")

    fetch_annotations(hydra_cfg)

    datagen_cfg = DatagenConfig(**OmegaConf.to_container(hydra_cfg["datagen"]))

    out_dir = hydra_cfg["out_dir"]
    os.makedirs(out_dir, exist_ok=True)

    n_existing_samples = sum(1 for fn in os.listdir(out_dir) if os.path.isdir(f"{out_dir}/{fn}"))
    total_samples = hydra_cfg["n_samples"]
    n_samples = total_samples - n_existing_samples
    if n_samples <= 0:
        print(f"Already have {n_existing_samples} samples, skipping")
        return
    print(f"{n_existing_samples} existing samples, generating {n_samples} more")
    nproc = hydra_cfg["n_proc"] if hydra_cfg["n_proc"] > 0 else os.cpu_count()
    with ProcessPoolExecutor(
        max_workers=nproc,
        initializer=procgen_init,
        initargs=(hydra_cfg["data_dir"], hydra_cfg["blacklist"], hydra_cfg["split_file"])
    ) as executor:
        with tqdm(total=total_samples, desc="Generating scenes", dynamic_ncols=True, initial=n_existing_samples) as pbar:
            futures: list[Future] = []
            n_submitted = 0
            try:
                while n_submitted < n_samples:
                    n_not_done = sum(not f.done() for f in futures)
                    if n_not_done < 4 * nproc:
                        futures.append(executor.submit(procgen_worker, datagen_cfg, out_dir))
                        n_submitted += 1
                    else:
                        time.sleep(0.5)

                    new_futures: list[Future] = []
                    for f in futures:
                        if f.done():
                            f.result()
                            pbar.update(1)
                        else:
                            new_futures.append(f)
                    futures = new_futures
                print("Waiting for remaining subprocesses to finish")
                if len(futures) > 0:
                    for f in as_completed(futures):
                        f.result()
                        pbar.update(1)
            except KeyboardInterrupt:
                print("Keyboard interrupt, shutting down subprocesses and exiting")
                set_exit_event()
                executor.shutdown(wait=False)
                raise
    print("Done!")

if __name__ == "__main__":
    main()
