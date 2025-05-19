import os
import base64
import io
import re
import json
import argparse
import time
import signal
from contextlib import contextmanager

if os.environ.get("PYOPENGL_PLATFORM") is None:
    os.environ["PYOPENGL_PLATFORM"] = "egl"
# pyrender spawns a lot of OMP threads, limiting to 1 significantly reduces overhead
if os.environ.get("OMP_NUM_THREADS") is None:
    os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import trimesh
from PIL import Image, ImageDraw
import pyrender
from openai import OpenAI
from tqdm import tqdm

from acronym_tools import create_gripper_marker

from graspmolmo.annotation import Annotation, Object, GraspLabel
from graspmolmo.datagen.datagen_utils import construct_cam_K, MeshLibrary

MAX_BATCH_SIZE_BYTES = 200_000_000  # max batch size in bytes, from openai api

GRIPPER_STYLE = "mesh"  # "mesh" or "marker"
GRASP_VOLUME_STYLE = "box"  # "sphere" or "box"

SYS_PROMPT = """
You are an expert in robotic grasp analysis. Your task is to generate precise and concise descriptions of robotic grasps in images.
Each image contains a robotic gripper interacting with an object. A red rectangle marks the area between the fingers of the gripper, which helps you identify the grasp location.
Your goal is to describe where the gripper is grasping the object and what the fingers are pinching. Follow these guidelines:

- Clearly specify the grasp location on the object (e.g., "on the handle," "near the rim," "on the body," "at the base").
- Indicate how the fingers of the gripper interact with the object (e.g., "gripping the inner and outer surfaces," "pinching from opposite sides").  
- The image may contain multiple viewpoints, but your description should focus on the grasp itself rather than commenting on different perspectives.  
- Do not speculate about grasp stability or effectiveness.  
- Keep the description concise but detailed, focusing only on the grasp.
- Do not mention the red rectangle in your description, it is only for visualization.

Example of a good description:
"The grasp is on the rim of the pan, approximately opposite the handle. The fingers are gripping the inside and outside of the pan's rim."

Your response should always describe the grasp clearly and concisely without asking for additional input.
""".strip()

def get_args():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    submit_parser = subparsers.add_parser("submit")
    submit_parser.add_argument("categories_file")
    submit_parser.add_argument("data_dir")
    submit_parser.add_argument("--batch-ids-file", help="If specified, write batch IDs to this file")
    submit_parser.add_argument("--collage_size", nargs=2, type=int, default=(2, 2))
    submit_parser.add_argument("--resolution", nargs=2, type=int, default=(640, 480), help="(width, height) in pixels")
    submit_parser.add_argument("--blacklist_file")
    submit_parser.add_argument("--out_dir", help="If specified, wait for batch to finish and save results to this directory. --batch-ids-file must also be specified.")
    submit_parser.set_defaults(func=submit)

    retrieve_parser = subparsers.add_parser("retrieve")
    retrieve_parser.add_argument("batch_ids_file")
    retrieve_parser.add_argument("out_dir")
    retrieve_parser.set_defaults(func=retrieve)

    return parser.parse_args()

@contextmanager
def block_signals(signals: list[int]):
    previous_blocked = signal.pthread_sigmask(signal.SIG_BLOCK, [])
    try:
        signal.pthread_sigmask(signal.SIG_BLOCK, signals)
        yield
    finally:
        signal.pthread_sigmask(signal.SIG_SETMASK, previous_blocked)

def create_scene(object_mesh: trimesh.Trimesh, grasp: np.ndarray):
    scene = trimesh.Scene([object_mesh])

    if GRIPPER_STYLE == "marker":
        marker: trimesh.Trimesh = create_gripper_marker([0, 255, 0])
        marker.apply_transform(grasp)
        scene.add_geometry(marker)

        bbox = marker.bounding_box_oriented.copy()
        bbox.visual.face_colors = [0, 0, 0, 0]
        scene.add_geometry(bbox, geom_name="gripper_obb")
    elif GRIPPER_STYLE == "mesh":
        mesh_trf = np.array([
            [0, 0, 1, 0],
            [1, 0, 0, 0],
            [0, 1, 0, 0.06],
            [0, 0, 0, 1]
        ])
        franka_gripper: trimesh.Trimesh = trimesh.load("data/franka_hand.obj")
        franka_gripper.apply_transform(grasp @ mesh_trf)
        scene.add_geometry(franka_gripper)

        bbox = franka_gripper.bounding_box_oriented.copy()
        bbox.visual.face_colors = [0, 0, 0, 0]
        scene.add_geometry(bbox, geom_name="gripper_obb")

    if GRASP_VOLUME_STYLE == "sphere":
        gripper_point = trimesh.primitives.Sphere(radius=0.005, center=grasp[:3, 3] + grasp[:3, 2] * 0.11)
        gripper_point.visual.face_colors = [255, 0, 0, 255]
        scene.add_geometry(gripper_point)
    elif GRASP_VOLUME_STYLE == "box":
        grasp_volume = trimesh.primitives.Box(extents=[0.08, 0.004, 0.048])
        grasp_volume.visual.face_colors = [255, 0, 0, 192]
        grasp_volume.apply_translation([0, 0, 0.09])
        grasp_volume.apply_transform(grasp)
        scene.add_geometry(grasp_volume)

    return scene

def trimesh_to_pyrender(tr_scene: trimesh.Scene):
    geometries = {name: pyrender.Mesh.from_trimesh(geom, smooth=False)
                for name, geom in tr_scene.geometry.items()}
    scene_pr = pyrender.Scene()
    for node in tr_scene.graph.nodes_geometry:
        pose, geom_name = tr_scene.graph[node]
        scene_pr.add(geometries[geom_name], pose=pose)
    return scene_pr

def adjust_camera_pose(cam_pose: np.ndarray, cam_K: np.ndarray, gripper_obb_verts: np.ndarray, width: int, height: int):
    cam_pose = cam_pose.copy()
    for _ in range(10):
        gripper_obb_verts_cam = np.column_stack([gripper_obb_verts, np.ones(len(gripper_obb_verts))]) @ np.linalg.inv(cam_pose)[:-1].T
        gripper_obb_verts_cam_proj = gripper_obb_verts_cam @ cam_K.T
        gripper_obb_verts_cam_proj /= gripper_obb_verts_cam_proj[:, 2:]
        gripper_obb_verts_cam_proj = gripper_obb_verts_cam_proj[:, :2]

        in_img_mask = (gripper_obb_verts_cam_proj[:, 0] >= 0) & (gripper_obb_verts_cam_proj[:, 0] <= width) & \
                      (gripper_obb_verts_cam_proj[:, 1] >= 0) & (gripper_obb_verts_cam_proj[:, 1] <= height)
        if np.all(in_img_mask):
            break
        cam_pose[:3, 3] += cam_pose[:3, 2] * 0.05
    else:
        print("Failed to adjust camera pose")

    return cam_pose

def generate_views(
    renderer: pyrender.OffscreenRenderer,
    tr_scene: trimesh.Scene,
    n_views: int,
    cam_K: np.ndarray,
    resolution: tuple[int, int],
    elevation_range: tuple[float, float]=(np.pi/8, np.pi/3)
):
    scene = trimesh_to_pyrender(tr_scene)

    cam = pyrender.camera.IntrinsicsCamera(
        fx=cam_K[0, 0],
        fy=cam_K[1, 1],
        cx=cam_K[0, 2],
        cy=cam_K[1, 2],
        name="camera",
    )
    cam_node = pyrender.Node(name="camera", camera=cam, matrix=np.eye(4))
    scene.add_node(cam_node)

    cam_light = pyrender.PointLight(color=np.array([255, 255, 255]), intensity=0.5)
    cam_light_node = pyrender.Node(name="cam_light", light=cam_light, matrix=np.eye(4))
    scene.add_node(cam_light_node)

    gripper_obb_verts = tr_scene.geometry["gripper_obb"].vertices

    r = tr_scene.bounding_sphere.primitive.radius * 3
    views: list[Image.Image] = []
    for azimuth in np.linspace(0, 2 * np.pi, n_views, endpoint=False) + np.random.uniform(0, 2 * np.pi):
        elevation = np.random.uniform(*elevation_range)
        inclination = np.pi/2 - elevation

        cam_pos = np.array([r * np.sin(inclination) * np.cos(azimuth), r * np.sin(inclination) * np.sin(azimuth), r * np.cos(inclination)])
        z_ax = cam_pos / np.linalg.norm(cam_pos)
        y_ax_ = np.array([0, 0, 1])
        x_ax = np.cross(y_ax_, z_ax)
        y_ax = np.cross(z_ax, x_ax)
        y_ax /= np.linalg.norm(y_ax)
        x_ax /= np.linalg.norm(x_ax)

        cam_pose = np.eye(4)
        cam_pose[:3, :3] = np.column_stack([x_ax, y_ax, z_ax])
        cam_pose[:3, 3] = cam_pos
        cam_pose = adjust_camera_pose(cam_pose, cam_K, gripper_obb_verts, *resolution)

        scene.set_pose(cam_node, cam_pose)
        scene.set_pose(cam_light_node, cam_pose)

        view_arr, _ = renderer.render(scene, pyrender.RenderFlags.NONE)
        view = Image.fromarray(view_arr)
        views.append(view)
    return views

def create_collage(views: list[Image.Image], nrows: int, ncols: int):
    assert len(views) == nrows * ncols
    resolution = views[0].size
    assert all(view.size == resolution for view in views)

    collage = Image.new("RGB", (resolution[0] * ncols, resolution[1] * nrows))
    for i, view in enumerate(views):
        view = view.resize(resolution)
        collage.paste(view, (i % ncols * resolution[0], i // ncols * resolution[1]))
    # Draw borders between images
    draw = ImageDraw.Draw(collage)
    # Vertical lines
    for x in range(1, ncols):
        draw.line([(x * resolution[0], 0), 
                   (x * resolution[0], resolution[1] * nrows)], 
                  fill='black', width=2)
    # Horizontal lines  
    for y in range(1, nrows):
        draw.line([(0, y * resolution[1]), 
                   (resolution[0] * ncols, y * resolution[1])],
                  fill='black', width=2)
    return collage

def encode_image(image: Image.Image):
    image_bytes = io.BytesIO()
    image.save(image_bytes, format="PNG")
    return base64.b64encode(image_bytes.getvalue()).decode("utf-8")

def create_query(object_category: str, object_id: str, grasp_id: int, collage: Image.Image):
    object_name = " ".join(map(str.lower, re.split(r"(?<!^)(?=[A-Z])", object_category)))

    request = {
        "custom_id": f"{object_category}__{object_id}__{grasp_id}",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": "gpt-4o",
            "messages": [
                {"role": "system", "content": SYS_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"These are multiple views of a(n) {object_name}. Describe the grasp."
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{encode_image(collage)}",
                                "detail": "high"
                            }
                        }
                    ]
                }
            ],
            "max_tokens": 8192
        }
    }

    return request

def submit(args, client: OpenAI):
    if args.out_dir and not args.batch_ids_file:
        raise ValueError("If --out-dir is specified, --batch-ids-file must also be specified")

    # If the batch IDs file exists, skip resubmission
    if args.batch_ids_file and os.path.exists(args.batch_ids_file):
        print("Batch IDs file exists, skipping resubmission")
        retrieve(args, client)
        return

    with open(args.categories_file, "r") as f:
        categories = f.read().strip().splitlines()
    with open(args.blacklist_file, "r") as f:
        blacklist = set(f.read().strip().splitlines())

    asset_library = MeshLibrary.from_categories(args.data_dir, categories)
    n_views = args.collage_size[0] * args.collage_size[1]

    renderer = pyrender.OffscreenRenderer(*args.resolution)

    dfov = 60.0
    cam_K = construct_cam_K(*args.resolution, dfov)

    batch_files = [io.BytesIO()]
    total_annotations = 0
    for category, obj_id in tqdm(asset_library):
        if f"{category}_{obj_id}" in blacklist:
            continue
        grasps, _ = asset_library.grasps(category, obj_id)
        object_mesh = asset_library[category, obj_id]
        for grasp_idx in asset_library.subsampled_grasp_idxs(category, obj_id):
            grasp = grasps[grasp_idx]
            scene = create_scene(object_mesh, grasp)
            views = generate_views(renderer, scene, n_views, cam_K, args.resolution)
            collage = create_collage(views, *args.collage_size)
            query = create_query(category, obj_id, grasp_idx, collage)
            bytes_to_write = (json.dumps(query) + "\n").encode("utf-8")
            if batch_files[-1].tell() + len(bytes_to_write) > MAX_BATCH_SIZE_BYTES:
                batch_files.append(io.BytesIO())
            batch_files[-1].write(bytes_to_write)
            total_annotations += 1
    print(f"Total annotations: {total_annotations}")

    with block_signals([signal.SIGINT]):  # make sure no preemption during batch submission
        batch_file_ids = []
        for batch_file in batch_files:
            batch_file.seek(0)
            batch_file_id = client.files.create(file=batch_file, purpose="batch").id
            batch_file_ids.append(batch_file_id)

        batch_ids = []
        for batch_file_id in batch_file_ids:
            batch = client.batches.create(input_file_id=batch_file_id, endpoint="/v1/chat/completions", completion_window="24h")
            batch_ids.append(batch.id)

        print(f"Submitted {len(batch_ids)} batch job(s) with ID(s): {' '.join(batch_ids)}")
        if args.batch_ids_file:
            with open(args.batch_ids_file, "w") as f:
                f.write("\n".join(batch_ids))

    if args.out_dir:
        retrieve(args, client)


def retrieve(args, client: OpenAI):
    os.makedirs(args.out_dir, exist_ok=True)
    done_statuses = ["completed", "expired", "cancelled", "failed"]

    with open(args.batch_ids_file, "r") as f:
        batch_ids = f.read().strip().splitlines()

    results_lines = []
    for batch_id in batch_ids:
        time_to_wait = 5
        while (batch := client.batches.retrieve(batch_id)).status not in done_statuses:
            print(f"Batch job {batch_id} is {batch.status}, waiting {time_to_wait} seconds before checking again")
            time.sleep(time_to_wait)
            time_to_wait = min(time_to_wait * 2, 10 * 60)  # cap at 10 minutes
        if batch.status != "completed":
            raise ValueError(f"Batch job {batch_id} did not complete successfully!")

        batch_file = client.files.content(batch.output_file_id)
        batch_file_lines = batch_file.content.decode("utf-8").splitlines()
        results_lines.extend(batch_file_lines)

    for line in results_lines:
        result = json.loads(line)
        annot_id: str = result["custom_id"]
        grasp_description = result["response"]["body"]["choices"][0]["message"]["content"]
        category, obj_id, grasp_id = annot_id.split("__")
        category_name = " ".join(map(str.lower, re.split(r"(?<!^)(?=[A-Z])", category)))
        annot = Annotation(
            obj=Object(object_category=category, object_id=obj_id),
            grasp_id=grasp_id,
            obj_description=f"The grasp is on the {category_name}.",
            grasp_description=grasp_description,
            grasp_label=GraspLabel.GOOD,
            user_id="openai"
        )
        with open(f"{args.out_dir}/synthetic__{annot_id}__{annot.user_id}.json", "w") as f:
            f.write(annot.model_dump_json())

def main():
    args = get_args()
    client = OpenAI()

    args.func(args, client)

if __name__ == "__main__":
    main()
