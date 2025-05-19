import base64
import io
import pickle
import trimesh
import numpy as np
from acronym_tools import create_gripper_marker

from graspmolmo.annotation import Annotation


with open("tmp/scene.pkl", "rb") as f:
    data = pickle.load(f)

glb_bytes = base64.b64decode(data["glb"].encode("utf-8"))
glb_bytes_io = io.BytesIO(glb_bytes)
base_scene: trimesh.Scene = trimesh.load(glb_bytes_io, file_type="glb")

annot_dict: dict[str, tuple[Annotation, np.ndarray]] = data["annotations"]

for i, view in enumerate(data["views"]):
    if len(view["annotations_in_view"]) == 0:
        print(f"No grasps for view {i}")
        continue
    print(f"Rendering view {i}")

    cam_K = np.array(view["cam_K"])
    cam_pose = np.array(view["cam_pose"])

    scene = base_scene.copy()
    scene.camera_transform = cam_pose
    scene.camera_intrinsics = cam_K

    for annot_id in view["annotations_in_view"]:
        _, grasp = annot_dict[annot_id]
        marker: trimesh.Trimesh = create_gripper_marker()
        marker.apply_transform(grasp)
        scene.add_geometry(marker)

    print(f"Number of grasps: {len(view['annotations_in_view'])}")
    scene.show()
