import argparse
from collections import defaultdict
import os
import shutil
import re
import pickle
import trimesh
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

import h5py

from graspmolmo.subsample_grasps import sample_grasps, load_aligned_meshes_and_grasps, load_unaligned_mesh_and_grasps
from graspmolmo.utils import tqdm

GRASP_START_OFFSET = 0.066
GRASP_END_OFFSET = 0.112

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("grasps_root")
    parser.add_argument("shapenet_root")
    parser.add_argument("output_dir")
    parser.add_argument("--blacklist", help="File containing object assets to blacklist")
    parser.add_argument("--n-proc", type=int, default=16, help="Number of processes to use")
    parser.add_argument("--n-grasps", type=int, default=4, help="Minimum number of grasps per object instance in a category")
    parser.add_argument("--min-grasps", type=int, default=32, help="Minimum number of grasps per category")
    parser.add_argument("--step", choices=["copy", "project", "subsample"])
    parser.add_argument("--sampling-categories-file", help="File containing categories to resample grasps for")
    return parser.parse_args()


def copy_assets(args):
    output_mesh_dir = os.path.join(args.output_dir, "meshes")
    output_grasp_dir = os.path.join(args.output_dir, "grasps")
    os.makedirs(output_mesh_dir, exist_ok=True)
    os.makedirs(output_grasp_dir, exist_ok=True)

    if args.blacklist:
        with open(args.blacklist, "r") as f:
            blacklist = set(f.read().strip().splitlines())
    else:
        blacklist = set()

    for grasp_filename in tqdm(os.listdir(args.grasps_root)):
        if grasp_filename[:-len(".h5")] in blacklist:
            continue
        category = grasp_filename.split("_", 1)[0]
        mesh_src_dir = os.path.join(args.shapenet_root, "models-OBJ", "models")
        mesh_dst_dir = os.path.join(output_mesh_dir, category)
        os.makedirs(mesh_dst_dir, exist_ok=True)

        shutil.copy2(
            os.path.join(args.grasps_root, grasp_filename),
            os.path.join(output_grasp_dir, grasp_filename)
        )
        with h5py.File(os.path.join(output_grasp_dir, grasp_filename), "r") as f:
            _, c, mesh_fn = f["object/file"][()].decode("utf-8").split("/")
            assert c == category
            mesh_id = mesh_fn[:-len(".obj")]
        
        shutil.copy2(
            os.path.join(mesh_src_dir, f"{mesh_id}.obj"),
            os.path.join(mesh_dst_dir, f"{mesh_id}.obj")
        )

        texture_files = set()
        with open(os.path.join(mesh_src_dir, f"{mesh_id}.mtl"), "r") as mtl_f:
            mtl_lines = []
            for line in mtl_f:
                line = line.strip()
                if m := re.fullmatch(r"d (\d+\.?\d*)", line):
                    mtl_lines.append(f"d {1-float(m.group(1))}")
                elif m := re.fullmatch(r"Kd 0(?:.0)? 0(?:.0)? 0(?:.0)?", line):
                    mtl_lines.append("Kd 1 1 1")
                elif m := re.fullmatch(r".+ (.+\.jpg)", line):
                    texture_files.add(m.group(1))
                    mtl_lines.append(line)
                else:
                    mtl_lines.append(line)
        with open(os.path.join(mesh_dst_dir, f"{mesh_id}.mtl"), "w") as mtl_f:
            mtl_f.write("\n".join(mtl_lines))

        for texture_file in texture_files:
            shutil.copy2(
                os.path.join(args.shapenet_root, "models-textures", "textures", texture_file),
                os.path.join(mesh_dst_dir, texture_file)
            )


def project_grasps_for_asset(args, grasp_filename: str):
    output_grasp_dir = os.path.join(args.output_dir, "grasps")
    mesh, grasps, succs = load_unaligned_mesh_and_grasps(args.output_dir, os.path.join(output_grasp_dir, grasp_filename))
    mesh: trimesh.Trimesh
    grasp_points = np.full((len(grasps), 3), np.nan)

    succ_idxs = np.flatnonzero(succs)
    succ_grasps = grasps[succ_idxs]
    raycast_pos = succ_grasps[:, :3, 3] + succ_grasps[:, :3, 2] * GRASP_START_OFFSET
    raycast_dir = succ_grasps[:, :3, 2]

    points, ray_idxs, _ = mesh.ray.intersects_location(raycast_pos, raycast_dir, multiple_hits=False)
    dists = np.linalg.norm(points - raycast_pos[ray_idxs], axis=1)
    in_bounds_mask = dists < (GRASP_END_OFFSET - GRASP_START_OFFSET)
    grasp_points[succ_idxs[ray_idxs[in_bounds_mask]]] = points[in_bounds_mask]

    # if the raycast misses, or lands on a point out of the grasp, fall back to the closest point on the mesh
    fallback_grasps_mask = np.ones(len(succ_grasps), dtype=bool)
    fallback_grasps_mask[ray_idxs] = False
    fallback_grasps_mask[ray_idxs[~in_bounds_mask]] = True
    if np.any(fallback_grasps_mask):
        fallback_points, _, _ = trimesh.proximity.closest_point(mesh, raycast_pos[fallback_grasps_mask])
        grasp_points[succ_idxs[fallback_grasps_mask]] = fallback_points

    with h5py.File(os.path.join(output_grasp_dir, grasp_filename), "r+") as f:
        if "grasps/points" in f:
            del f["grasps/points"]
        f["grasps/points"] = grasp_points


def project_grasps(args):
    output_grasp_dir = os.path.join(args.output_dir, "grasps")
    with ProcessPoolExecutor(args.n_proc) as executor:
        futures = [executor.submit(project_grasps_for_asset, args, grasp_filename) for grasp_filename in os.listdir(output_grasp_dir)]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Projecting grasps"):
            future.result()


def subsample_grasps(args):
    # maps (category, object_id, grasp_id) -> whether the grasp is annotated
    annotation_skeleton: dict[str, dict[str, dict[int, bool]]] = {}
    output_grasp_dir = os.path.join(args.output_dir, "grasps")

    category_objects: dict[str, list[str]] = defaultdict(list)
    for grasp_filename in os.listdir(output_grasp_dir):
        category, obj_id = grasp_filename.split("_", 1)
        obj_id = obj_id[:-len(".h5")]
        category_objects[category].append(obj_id)

    if args.sampling_categories_file:
        with open(args.sampling_categories_file, "r") as f:
            sampling_categories = f.read().splitlines()
    else:
        sampling_categories = sorted(category_objects.keys())

    for category in tqdm(sampling_categories, desc="Subsampling grasps"):
        obj_ids = category_objects[category]
        _, grasps, succs, aligned_obj_ids, unaligned_obj_ids = load_aligned_meshes_and_grasps(args.output_dir, category, obj_ids, args.n_proc)

        if len(aligned_obj_ids) > 0:
            n_grasps = max(args.n_grasps * len(aligned_obj_ids), args.min_grasps)
            grasp_idxs_per_obj = sample_grasps(grasps, succs, n_grasps)
            for obj_id, grasp_idxs in zip(aligned_obj_ids, grasp_idxs_per_obj):
                grasp_filename = f"{category}_{obj_id}.h5"
                with h5py.File(os.path.join(output_grasp_dir, grasp_filename), "r+") as f:
                    if "grasps/sampled_idxs" in f:
                        del f["grasps/sampled_idxs"]
                    f["grasps/sampled_idxs"] = grasp_idxs

                if category not in annotation_skeleton:
                    annotation_skeleton[category] = {}
                annotation_skeleton[category][obj_id] = {i.item(): False for i in grasp_idxs}

        for obj_id in unaligned_obj_ids:
            path = f"{output_grasp_dir}/{category}_{obj_id}.h5"
            _, grasps, succs = load_unaligned_mesh_and_grasps(args.output_dir, path)
            grasp_idxs = sample_grasps([grasps], [succs], args.n_grasps)[0]
            grasp_filename = os.path.basename(path)
            with h5py.File(os.path.join(output_grasp_dir, grasp_filename), "r+") as f:
                if "grasps/sampled_idxs" in f:
                    del f["grasps/sampled_idxs"]
                f["grasps/sampled_idxs"] = grasp_idxs

            if category not in annotation_skeleton:
                annotation_skeleton[category] = {}
            annotation_skeleton[category][obj_id] = {i.item(): False for i in grasp_idxs}

    with open(os.path.join(args.output_dir, "annotation_skeleton.pkl"), "wb") as f:
        pickle.dump(annotation_skeleton, f)

def main():
    args = get_args()
    has_step = args.step is not None

    if not has_step or args.step == "copy":
        copy_assets(args)
    if not has_step or args.step == "project":
        project_grasps(args)
    if not has_step or args.step == "subsample":
        subsample_grasps(args)

if __name__ == "__main__":
    main()
