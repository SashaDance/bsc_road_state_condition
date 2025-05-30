import os
import glob
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from scipy.ndimage import distance_transform_edt
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from tqdm import tqdm
from typing import Tuple, Dict
from .metric_depth.depth_anything_v2.dpt import DepthAnythingV2


MODEL_CONFIGS = {
    'small': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'base':  {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'large': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]}
}


class DepthInterpolation:
    """
    Standalone LiDAR-guided densification with integrated depth inference,
    projection, residual/scale fusion, and hold-out evaluation.
    """
    def __init__(self,
                 model_path: str,
                 device: str,
                 camera_resolution: Tuple[int, int],
                 proj_matr: np.ndarray,
                 model_type: str = 'large',
                 sigma_pix: float = 8.0,
                 interp_method: str = 'linear'):
        params = MODEL_CONFIGS[model_type]
        self.model = DepthAnythingV2(**params)
        self.model.load_state_dict(torch.load(model_path))
        self.model.to(device)
        self.device = device

        self.camera_resolution = camera_resolution
        self.proj_matr = proj_matr

        self.sigma_pix = sigma_pix
        self.interp_method = interp_method

    def _infer_model(self, img: np.ndarray) -> np.ndarray:
        return self.model.infer_image(img)

    def project_points_to_camera(self,
                                 points: np.ndarray
                                 ) -> Tuple[np.ndarray, np.ndarray]:
        if points.shape[0] == 3:
            points = np.vstack((points, np.ones((1, points.shape[1]))))
        if len(points.shape) != 2 or points.shape[0] != 4:
            raise ValueError(
                f'Wrong shape of points array: {points.shape}; expected: (4, n), where n - number of points.'
            )
        if self.proj_matr.shape != (3, 4):
            raise ValueError(f'Wrong proj_matrix shape: {self.proj_matr}; expected: (3, 4).')
        in_image = points[2, :] > 0
        depths = np.sqrt(points[0, in_image] ** 2 + points[1, in_image] ** 2 + points[2, in_image] ** 2)
        uvw = np.dot(self.proj_matr, points[:, in_image])
        uv = uvw[:2, :]
        w = uvw[2, :]
        uv[0, :] /= w
        uv[1, :] /= w
        in_image = (
            (uv[0, :] >= 0) 
            * (uv[0, :] < self.camera_resolution[0]) 
            * (uv[1, :] >= 0) 
            * (uv[1, :] < self.camera_resolution[1])
        )
        return uv[:, in_image].astype(int), depths[in_image]

    def _confidence_map(self,
                        H: int,
                        W: int,
                        uv: np.ndarray
                        ) -> Tuple[np.ndarray, np.ndarray]:
        mask = np.zeros((H, W), dtype=bool)
        mask[uv[1], uv[0]] = True
        dmap = distance_transform_edt(~mask)
        conf = np.exp(-(dmap ** 2) / (2 * self.sigma_pix ** 2))
        return conf.astype(np.float32), dmap

    def _interpolate_sparse_field(self,
                                  uv: np.ndarray,
                                  values: np.ndarray,
                                  shape: Tuple[int, int]
                                  ) -> np.ndarray:
        H, W = shape
        grid_x, grid_y = np.meshgrid(np.arange(W), np.arange(H))
        full = griddata(
            uv.T, values,
            (grid_x, grid_y),
            method=self.interp_method,
            fill_value=0.0
        )
        return full.astype(np.float32)

    def _fuse(self,
              d_pred: np.ndarray,
              corr: np.ndarray,
              conf: np.ndarray
              ) -> np.ndarray:
        blended_ratio = 1.0 + conf * (corr - 1.0)
        return d_pred * blended_ratio

    def densify(self,
                d_pred: np.ndarray,
                uv: np.ndarray,
                d_lidar: np.ndarray
                ) -> Tuple[np.ndarray, np.ndarray]:
        H, W = d_pred.shape
        conf, _ = self._confidence_map(H, W, uv)

        dlidar = d_lidar.flatten()
        dpred  = d_pred[uv[1], uv[0]].flatten()
        sparse_vals = dlidar / (dpred + 1e-8)

        interp_field = self._interpolate_sparse_field(uv, sparse_vals, (H, W))
        d_fused = self._fuse(d_pred, interp_field, conf)
        return d_fused, conf

    def evaluate_on_dataset(self,
                            dataset_dir: str,
                            split_ratio: float = 0.85
                            ) -> Dict[str, Dict[str, list]]:
        metrics = {'mae': [], 'mape': []}
        img_dir = os.path.join(dataset_dir, 'img')
        pc_dir  = os.path.join(dataset_dir, 'pc')
        img_files = sorted(glob.glob(os.path.join(img_dir, '*.png')))

        for img_path in tqdm(img_files):
            img = plt.imread(img_path)
            base = os.path.splitext(os.path.basename(img_path))[0]
            pc = np.fromfile(os.path.join(pc_dir, f'{base}.bin')).reshape(-1, 4)
            points = pc[:, :3].T

            d_pred = self._infer_model(img)
            uv, d_lidar = self.project_points_to_camera(points)
            if uv.shape[1] < 10:
                continue

            idx = np.random.permutation(uv.shape[1])
            n_train = int(uv.shape[1] * split_ratio)
            train_idx, val_idx = idx[:n_train], idx[n_train:]

            d_fused, _ = self.densify(
                d_pred,
                uv[:, train_idx],
                d_lidar[train_idx]
            )

            d_gt  = d_lidar[val_idx]
            d_est = d_fused[uv[1, val_idx], uv[0, val_idx]]
            metrics['mae' ].append(mean_absolute_error      (d_gt, d_est))
            metrics['mape'].append(mean_absolute_percentage_error(d_gt, d_est))

        mae_arr  = np.array(metrics['mae'])
        mape_arr = np.array(metrics['mape'])
        results = {
            'mean_mae':   float(np.mean(mae_arr)),
            'median_mae': float(np.median(mae_arr)),
            'mean_mape':   float(np.mean(mape_arr)),
            'median_mape': float(np.median(mape_arr)),
            'mae_values':  metrics['mae'],
            'mape_values': metrics['mape']
        }
        return results

    @staticmethod
    def print_metrics(results: Dict[str, Dict[str, list]]):
        """
        Nicely print out aggregated MAE/MAPE for a run.
        """
        print(f"MAE:  Mean={results['mean_mae']:.4f}, Median={results['median_mae']:.4f}")
        print(f"MAPE: Mean={results['mean_mape']*100:.2f}%, Median={results['median_mape']*100:.2f}%")