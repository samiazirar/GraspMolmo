import os
import numpy as np
import trimesh
from trimesh.path import Path2D
from tqdm import tqdm
from scipy.spatial.transform import Rotation as scipyR
from itertools import takewhile, count
import multiprocessing as mp
from multiprocessing.pool import AsyncResult

from acronym_tools import load_mesh, load_grasps, create_gripper_marker


GRIPPER_POS_OFFSET = 0.075

class MalformedMeshError(Exception):
    pass

def cvh(mesh: trimesh.Trimesh, max_iter=5):
    """
    Get the convex hull of a mesh.
    Due to numerical issues, mesh.convex_hull is sometimes not watertight (despite by definition being watertight).
    Repeatedly computing the convex hull seems to fix this.
    see: https://github.com/mikedh/trimesh/issues/535
    """
    cvh = mesh
    for _ in range(max_iter):
        if (cvh := cvh.convex_hull).is_watertight:
            return cvh
    raise MalformedMeshError("Failed to compute convex hull")

def is_zero_measure_2d(p: Path2D):
    return p.vertices.size == 0 or np.isclose(p.area, 0)

def icp_2d(src_mesh: trimesh.Trimesh, target_mesh: trimesh.Trimesh, N=1000, max_iterations=50, tolerance=1e-5):
    src_mesh = src_mesh.copy()
    target_mesh = target_mesh.copy()
    src_mesh.fix_normals()
    target_mesh.fix_normals()
    if not src_mesh.is_volume or not target_mesh.is_volume:
        return np.eye(3)

    src_proj = src_mesh.projected(np.array([0, 0, 1]))
    target_proj = target_mesh.projected(np.array([0, 0, 1]))
    if is_zero_measure_2d(src_proj) or is_zero_measure_2d(target_proj):
        return np.eye(3)
    source_pts = src_proj.sample(N)
    target_pts = target_proj.sample(N)
    # sometimes the projected points are empty (weird meshes)
    if source_pts.size == 0 or target_pts.size == 0:
        return np.eye(3)

    def centroid_align(A, B):
        centroid_A = np.mean(A, axis=0)
        centroid_B = np.mean(B, axis=0)
        return centroid_A, centroid_B

    def scale_normalize(A, B):
        scale_A = np.sqrt(np.mean(np.linalg.norm(A, axis=1) ** 2))
        scale_B = np.sqrt(np.mean(np.linalg.norm(B, axis=1) ** 2))
        scale = scale_B / scale_A
        return scale

    def estimate_pca_rotation(A, B):
        def principal_axis(points):
            cov_matrix = np.cov(points.T)
            eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
            return eigenvectors[:, np.argmax(eigenvalues)]  # Eigenvector with max variance

        # Compute principal axes
        axis_A = principal_axis(A)
        axis_B = principal_axis(B)

        # Try both signs for the principal axis of A and choose the one with lower cost
        rotations = []
        for sign_A in [1, -1]:
            axis_A_signed = sign_A * axis_A
            ax = np.array([0, 0, np.cross(axis_A_signed, axis_B)])
            theta = np.arctan2(np.linalg.norm(ax), np.dot(axis_A_signed, axis_B))
            rotmat = scipyR.from_rotvec(ax / np.linalg.norm(ax) * theta).as_matrix()
            rotations.append(rotmat[:2, :2])

        best_R = None
        best_cost = float('inf')
        for R in rotations:
            trf = np.eye(4)
            trf[:2, :2] = R
            src_copy = src_mesh.copy()
            src_copy.apply_transform(trf)
            if not src_copy.is_watertight:
                src_copy = cvh(src_copy)
            cost = src_copy.volume + target_mesh.volume - 2 * src_copy.intersection(target_mesh).volume
            if cost < best_cost:
                best_cost = cost
                best_R = R

        return best_R

    def estimate_transform(A, B):
        """Estimate optimal similarity transform (s, R, t) using Procrustes."""
        H = A.T @ B
        U, _, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T
        s = np.trace(R.T @ H) / np.trace(A.T @ A)
        t = np.mean(B - s * (A @ R.T), axis=0)
        return s, R, t

    # Step 1: Initial Alignment
    centroid_s, centroid_t = centroid_align(source_pts, target_pts)
    source_pts -= centroid_s
    target_pts -= centroid_t
    src_mesh.apply_translation(-np.concatenate([centroid_s, [0]]))
    target_mesh.apply_translation(-np.concatenate([centroid_t, [0]]))

    scale = scale_normalize(source_pts, target_pts)
    source_pts *= scale
    src_mesh.apply_scale(scale)
    
    # Step 2: Coarse Rotation with PCA
    R_pca = estimate_pca_rotation(source_pts, target_pts)
    source_pts = source_pts @ R_pca.T  # Apply PCA-based rotation

    # Step 3: ICP Refinement
    prev_error = float('inf')
    for i in range(max_iterations):
        # Find closest points
        indices = np.array([np.argmin(np.linalg.norm(target_pts - sp, axis=1)) for sp in source_pts])
        matched_pts = target_pts[indices]

        # Compute optimal transform
        s, R, t = estimate_transform(source_pts, matched_pts)
        source_pts = s * (source_pts @ R.T) + t  # Apply transformation

        # Check for convergence
        mean_error = np.mean(np.linalg.norm(matched_pts - source_pts, axis=1))
        if abs(prev_error - mean_error) < tolerance:
            break
        prev_error = mean_error

    # Compute final transformation matrix
    T = np.eye(3)
    T[:2, :2] = s * R @ R_pca  # Include PCA-based rotation in final transform
    T[:2, 2] = t + centroid_t - s * (centroid_s @ R.T @ R_pca)
    return T


def rot_distance(rot_deltas: np.ndarray):
    return scipyR.from_matrix(rot_deltas).magnitude()


def grasp_dist(grasp: np.ndarray, all_grasps: np.ndarray):
    """Distance between a grasp and a set of grasps."""
    assert grasp.ndim == 2
    if all_grasps.ndim == 2:
        all_grasps = all_grasps[None]
    grasp_pos = grasp[:3, 3] + GRIPPER_POS_OFFSET * grasp[:3, 2]
    all_grasps_pos = all_grasps[:, :3, 3] + GRIPPER_POS_OFFSET * all_grasps[:, :3, 2]
    pos_dist = np.linalg.norm(grasp_pos[None] - all_grasps_pos, axis=1)

    rd1 = rot_distance(all_grasps[:, :3, :3].transpose(0,2,1) @ grasp[None, :3, :3])
    rd2 = rot_distance(all_grasps[:, :3, :3].transpose(0,2,1) @ grasp[None, :3, :3] @ scipyR.from_euler("z", [np.pi]).as_matrix())
    rot_dist = np.minimum(rd1, rd2)

    return pos_dist + 0.01 * rot_dist

def load_and_align(data_dir: str, path: str, target_cvh: trimesh.Trimesh):
    mesh = load_mesh(path, mesh_root_dir=data_dir)
    mesh_grasps, succ = load_grasps(path)
    mesh_grasps[..., :3, 3] -= mesh.centroid
    mesh.apply_translation(-mesh.centroid)

    trf_2d = icp_2d(cvh(mesh), target_cvh)
    trf = np.eye(4)
    trf[:2, :2] = trf_2d[:2, :2]
    trf[:2, 3] = trf_2d[:2, 2]
    mesh.apply_transform(trf)
    mesh_grasps = trf @ mesh_grasps
    return mesh, mesh_grasps, succ

def load_unaligned_mesh_and_grasps(data_dir: str, path: str):
    mesh = load_mesh(path, mesh_root_dir=data_dir)
    mesh_grasps, succ = load_grasps(path)
    mesh_grasps[..., :3, 3] -= mesh.centroid
    mesh.apply_translation(-mesh.centroid)
    return mesh, mesh_grasps, succ

def raise_error(e: Exception):
    raise e

def load_aligned_meshes_and_grasps(data_dir: str, category: str, obj_ids: list[str], n_proc=16):
    meshes = []
    grasps = []
    grasp_succs = []
    first_cvh: trimesh.Trimesh | None = None

    with mp.Pool(processes=n_proc) as pool:
        futures: list[AsyncResult] = []
        for obj_id in obj_ids:
            path = f"{data_dir}/grasps/{category}_{obj_id}.h5"

            if first_cvh is not None:
                futures.append(pool.apply_async(load_and_align, (data_dir, path, first_cvh)))
            else:
                try:
                    mesh, mesh_grasps, succ = load_unaligned_mesh_and_grasps(data_dir, path)
                    first_cvh = cvh(mesh)
                    ar = AsyncResult(pool, None, None)
                    ar._set(0, (True, (mesh, mesh_grasps, succ)))
                    futures.append(ar)
                except MalformedMeshError as e:
                    # propagate the error into a AsyncResult
                    futures.append(pool.apply_async(raise_error, (e,)))

        aligned_obj_ids = []
        unaligned_obj_ids = []
        for i, future in tqdm(enumerate(futures), leave=False, desc=f"Aligning meshes for {category}", total=len(futures)):
            try:
                mesh, mesh_grasps, succ = future.get()
                meshes.append(mesh)
                grasps.append(mesh_grasps)
                grasp_succs.append(succ)
                aligned_obj_ids.append(obj_ids[i])
            except MalformedMeshError:
                print(f"Couldn't align mesh {category}_{obj_ids[i]}, will fall back to per-instance sampling")
                unaligned_obj_ids.append(obj_ids[i])
    return meshes, grasps, grasp_succs, aligned_obj_ids, unaligned_obj_ids

def sample_grasps(grasps: list[np.ndarray], grasp_succs: list[np.ndarray], n_grasps: int) -> list[list[int]]:
    grasp_succ_idxs = [np.nonzero(succ)[0] for succ in grasp_succs]
    n_instances = len(grasps)

    all_grasps = np.concatenate([g[idxs] for g, idxs in zip(grasps, grasp_succ_idxs)], axis=0)
    grasp_obj_idxs = np.concatenate([np.full(len(grasp_succ_idxs[i]), i) for i in range(len(grasp_succ_idxs))], axis=0)
    points_left_mask = np.ones(len(all_grasps), dtype=bool)
    sample_inds = []
    dists = np.full(len(all_grasps), np.inf, dtype=float)

    if n_grasps >= len(all_grasps):
        sample_inds = np.arange(len(all_grasps))
    else:
        selected = 0
        sample_inds.append(selected)
        points_left_mask[selected] = False

        with tqdm(total=n_grasps, desc="Sampling grasps", leave=False) as pbar:
            pbar.update(1)
            for i in takewhile(lambda _: len(sample_inds) < n_grasps, count(1)):
                instance_idx = i % n_instances
                eligible_points_mask = (grasp_obj_idxs == instance_idx) & points_left_mask
                if not eligible_points_mask.any():
                    continue

                last_added_idx = sample_inds[-1]
                dists_to_last_added = grasp_dist(all_grasps[last_added_idx], all_grasps[points_left_mask])
                dists[points_left_mask] = np.minimum(dists[points_left_mask], dists_to_last_added)

                eligible_dists = np.where(eligible_points_mask, dists, -np.inf)
                selected = np.argmax(eligible_dists)
                sample_inds.append(selected)
                points_left_mask[selected] = False
                pbar.update(1)

    # maps instance index to the start index of its grasps in all_grasps
    obj_idx_cumsum = np.cumsum([0] + [len(grasp_succ_idxs[i]) for i in range(len(grasp_succ_idxs))])
    ret = [[] for _ in range(n_instances)]
    for i in sample_inds:
        instance_idx = grasp_obj_idxs[i]
        ret[instance_idx].append(grasp_succ_idxs[instance_idx][i - obj_idx_cumsum[instance_idx]])
    return ret


def viz_obj_grasps(meshes: list[trimesh.Trimesh], grasps: list[np.ndarray], grasp_idxs_per_obj: list[list[int]]):
    scene = trimesh.Scene()
    for m, grasps, grasp_idxs in zip(meshes, grasps, grasp_idxs_per_obj):
        scene.add_geometry(m)
        for grasp_id in grasp_idxs:
            gripper_marker = create_gripper_marker()
            gripper_marker.apply_transform(grasps[grasp_id])
            scene.add_geometry(gripper_marker)
    scene.show()

def main():
    category = "Pan"
    obj_ids = []
    for fn in os.listdir("data/grasps"):
        if fn.startswith(category + "_"):
            obj_ids.append(fn[len(category) + 1:-len(".h5")])
    obj_ids.sort()

    meshes, grasps_per_obj, grasp_succs_per_obj, _, _ = load_aligned_meshes_and_grasps("data", category, obj_ids)

    # print("Per-Instance Sampling")
    # grasp_idxs_per_obj = []
    # for grasps, succs in zip(grasps_per_obj, grasp_succs_per_obj):
    #     from preprocess_shapenet import subsample_grasps
    #     grasp_ids = subsample_grasps(succs, grasps, 2)
    #     grasp_idxs_per_obj.append(grasp_ids)
    # viz_obj_grasps(meshes, grasps_per_obj, grasp_idxs_per_obj)

    print("Cross-Instance Sampling")
    grasp_idxs_per_obj = sample_grasps(grasps_per_obj, grasp_succs_per_obj, 2*len(obj_ids))
    viz_obj_grasps(meshes, grasps_per_obj, grasp_idxs_per_obj)

if __name__ == "__main__":
    main()
