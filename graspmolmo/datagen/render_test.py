import os
import pickle
import time
if os.environ.get("PYOPENGL_PLATFORM") is None:
    os.environ["PYOPENGL_PLATFORM"] = "egl"

import pyrender
import numpy as np
import trimesh
import base64
import io
from PIL import Image
import pyrender.light

from graspmolmo.datagen.datagen_utils import Annotation


result = Image.new("RGB", (640 * 3, 480 * 3))

with open("tmp/scene.pkl", "rb") as f:
    data = pickle.load(f)

glb_bytes = base64.b64decode(data["glb"].encode("utf-8"))
glb_bytes_io = io.BytesIO(glb_bytes)
tr_scene: trimesh.Scene = trimesh.load(glb_bytes_io, file_type="glb")
scene = pyrender.Scene.from_trimesh_scene(tr_scene)
renderer = pyrender.OffscreenRenderer(640, 480)

annot_dict: dict[str, tuple[Annotation, np.ndarray]] = data["annotations"]

lighting = data["lighting"] if "lighting" in data else data["views"][0]["lighting"]
for light in lighting:
    light_type = getattr(pyrender.light, light["type"])
    light_args = light["args"]
    light_args["color"] = np.array(light_args["color"]) / 255.0
    light_node = pyrender.Node(light["args"]["name"], matrix=light["transform"], light=light_type(**light_args))
    scene.add_node(light_node)

for i, view in enumerate(data["views"]):
    cam_K = np.array(view["cam_K"])
    cam_pose = np.array(view["cam_pose"])

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

    start = time.perf_counter()
    color, depth = renderer.render(scene, flags=pyrender.RenderFlags.SHADOWS_DIRECTIONAL)
    end = time.perf_counter()
    print(f"Render time: {1000 * (end - start):.2f} ms")
    r, c = i // 3, i % 3
    result.paste(Image.fromarray(color), (640 * c, 480 * r))

result.save("renders.png")
