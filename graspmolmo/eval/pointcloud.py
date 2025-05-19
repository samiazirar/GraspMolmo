import abc

import torch
from torch import nn
import numpy as np
import trimesh
import torch.nn.functional as F

from scipy.spatial import cKDTree

from learning3d.models.deepgmr import Conv1dBNReLU, TNet, gmm_params, gmm_register, transform
from learning3d.data_utils.dataloaders import get_rri

from graspmolmo.eval.utils import download

def process(pc: np.ndarray, n_neighbors: int) -> np.ndarray:
    arr = np.concatenate([pc, get_rri(pc - pc.mean(axis=0), n_neighbors)], axis=1)
    return arr

class PointNet(nn.Module):
	def __init__(self, use_rri, use_tnet=False, nearest_neighbors=20):
		super(PointNet, self).__init__()
		self.use_tnet = use_tnet
		self.tnet = TNet() if self.use_tnet else None
		d_input = nearest_neighbors * 4 if use_rri else 3
		self.encoder = nn.Sequential(
			Conv1dBNReLU(d_input, 64),
			Conv1dBNReLU(64, 128),
			Conv1dBNReLU(128, 256),
			Conv1dBNReLU(256, 1024))
		self.decoder = nn.Sequential(
			Conv1dBNReLU(1024 * 2, 512),
			Conv1dBNReLU(512, 256),
			Conv1dBNReLU(256, 128),
			nn.Conv1d(128, 16, kernel_size=1))

	def forward(self, pts):
		pts = self.tnet(pts) if self.use_tnet else pts
		f_loc = self.encoder(pts)
		f_glob, _ = f_loc.max(dim=2)
		f_glob = f_glob.unsqueeze(2).expand_as(f_loc)
		y = self.decoder(torch.cat([f_loc, f_glob], dim=1))
		return y.transpose(1, 2)

class DeepGMR(nn.Module):
	def __init__(self, use_rri=True, feature_model=None, nearest_neighbors=20):
		super(DeepGMR, self).__init__()
		self.backbone = feature_model if not None else PointNet(use_rri=use_rri, nearest_neighbors=nearest_neighbors)
		self.use_rri = use_rri

	def forward(self, template, source):
		if self.use_rri:
			self.template = template[..., :3]
			self.source = source[..., :3]
			template_features = template[..., 3:].transpose(1, 2)
			source_features = source[..., 3:].transpose(1, 2)
		else:
			self.template = template
			self.source = source
			template_features = (template - template.mean(dim=2, keepdim=True)).transpose(1, 2)
			source_features = (source - source.mean(dim=2, keepdim=True)).transpose(1, 2)

		self.template_gamma = F.softmax(self.backbone(template_features), dim=2)
		self.template_pi, self.template_mu, self.template_sigma = gmm_params(self.template_gamma, self.template)
		self.source_gamma = F.softmax(self.backbone(source_features), dim=2)
		self.source_pi, self.source_mu, self.source_sigma = gmm_params(self.source_gamma, self.source)

		est_T_inverse = gmm_register(self.template_pi, self.template_mu, self.source_mu, self.source_sigma)
		est_T = gmm_register(self.source_pi, self.source_mu, self.template_mu, self.template_sigma) # [template = source * est_T]

		transformed_source = transform.transform_point_cloud(self.source, est_T[:, :3, :3], est_T[:, :3, 3])

		result = {'est_R': est_T[:, :3, :3],
				  'est_t': est_T[:, :3, 3],
				  'est_R_inverse': est_T_inverse[:, :3, :3],
				  'est_t_inverese': est_T_inverse[:, :3, 3],
				  'est_T': est_T,
				  'est_T_inverse': est_T_inverse,
				  'r': template_features - source_features,
				  'transformed_source': transformed_source}

		return result

class PCRegistration(abc.ABC):
    @abc.abstractmethod
    def register(self, template: np.ndarray, source: np.ndarray, **kwargs) -> tuple[np.ndarray, np.ndarray, float]:
        """
        template: (N, 3)
        source: (M, 3)
        returns: (4, 4) transformation matrix, (M, 3) transformed source, final cost
        """
        raise NotImplementedError

class DeepGMRRegistration(PCRegistration):
    def __init__(self, device="cuda"):
        use_rri = True
        self.n_neighbors = 20
        self.device = device
        self.deepgmr = DeepGMR(use_rri=use_rri, nearest_neighbors=self.n_neighbors, feature_model=PointNet(use_rri=use_rri, nearest_neighbors=self.n_neighbors))

        ckpt_path = download("https://github.com/vinits5/learning3d/raw/refs/heads/master/pretrained/exp_deepgmr/models/best_model.pth", "deepgmr.pth")
        self.deepgmr.load_state_dict(torch.load(ckpt_path, weights_only=True))
        self.deepgmr.to(device)

    def to(self, device: str):
        self.device = device
        self.deepgmr.to(device)

    def register(self, template: np.ndarray, source: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        n_points = min(len(template), len(source))

        template_idxs = np.random.choice(len(template), n_points, replace=False) if len(template) > n_points else np.arange(len(template))
        source_idxs = np.random.choice(len(source), n_points, replace=False) if len(source) > n_points else np.arange(len(source))
        template_sampled = template[template_idxs]
        source_sampled = source[source_idxs]

        template_torch = torch.from_numpy(process(template_sampled, self.n_neighbors)).unsqueeze(0).float().to(self.device)
        source_torch = torch.from_numpy(process(source_sampled, self.n_neighbors)).unsqueeze(0).float().to(self.device)

        with torch.no_grad():
            output = self.deepgmr(template_torch, source_torch)

        transformed_source = output['transformed_source'].cpu().numpy()[0]
        trf_gmr = output["est_T"].cpu().numpy()[0]

        cost = np.mean(np.linalg.norm(transformed_source - template_sampled, axis=1))
        return trf_gmr, transformed_source, cost

class ICPRegistration(PCRegistration):
    def register(self, template: np.ndarray, source: np.ndarray, max_iterations: int = 100, threshold: float = 1e-8) -> tuple[np.ndarray, np.ndarray, float]:
        trf_icp, transformed_source, _ = trimesh.registration.icp(source, template, max_iterations=max_iterations, threshold=threshold, scale=False, reflection=False)

        template_kdtree = cKDTree(template)
        dists, _ = template_kdtree.query(transformed_source, 1)
        cost = np.mean(dists)

        return trf_icp, transformed_source, cost


class CompositePCRegistration:
    def __init__(self, *steps: PCRegistration):
        self.steps = steps

    def register(self, template: np.ndarray, source: np.ndarray, **kwargs) -> tuple[np.ndarray, np.ndarray, float]:
        trf = np.eye(4)
        transformed_source = source
        for i, step in enumerate(self.steps):
            step_kwargs = kwargs[i] if i in kwargs else {}
            step_trf, transformed_source, cost = step.register(template, transformed_source, **step_kwargs)
            trf = step_trf @ trf
        return trf, transformed_source, cost
