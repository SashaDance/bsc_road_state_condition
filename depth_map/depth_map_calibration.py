import numpy as np
import torch
import os
import matplotlib.pyplot as plt
import glob
from tqdm import tqdm
from .metric_depth.depth_anything_v2.dpt import DepthAnythingV2
from sklearn.base import BaseEstimator
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error
from typing import Tuple, Optional, Dict, Union, List

MODEL_CONFIGS = {
    'small': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'base': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'large': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]}
}

class ExponentialCalibrator(BaseEstimator):
    def __init__(self):
        self.a_ = None
        self.b_ = None
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> "ExponentialCalibrator":
        X_flat = X.ravel()
        y_flat = y.ravel()
        
        if np.any(y_flat <= 0):
            raise ValueError('All true depth values must be positive for exponential calibration')
        
        log_y = np.log(y_flat)
        
        X_design = np.column_stack((np.ones_like(X_flat), X_flat))
        coeffs, _, _, _ = np.linalg.lstsq(X_design, log_y, rcond=None)
        
        self.log_a_ = coeffs[0]
        self.b_ = coeffs[1]
        self.a_ = np.exp(self.log_a_)
        
        return self
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.a_ * np.exp(self.b_ * X.ravel())
    
class DepthCalibrator:
    def __init__(self, 
                 model_path: str, 
                 device: str, 
                 camera_resolution: Tuple[int, int],
                 proj_matr: np.ndarray, 
                 type: str = 'large',
                 calibrator: Optional[BaseEstimator] = None):
        
        params = MODEL_CONFIGS[type]
        self.model = DepthAnythingV2(**params)
        self.model.load_state_dict(torch.load(model_path))
        self.model.to(device)
        self.proj_matr = proj_matr
        self.camera_resolution = camera_resolution
        
        self.calibrator = calibrator if calibrator is not None else LinearRegression()

    def project_points_to_camera(self, points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
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
    
    def _infer_model(self, img: np.ndarray) -> np.ndarray:
        return self.model.infer_image(img)

    def _calibrate_depth_map(self, depth: np.ndarray, lidar_points: np.ndarray, 
                             depth_lidar: np.ndarray, visualize: bool = False, 
                             save_dir: str = None, split_ratio: float = 0.8) -> Tuple[np.ndarray, dict]:
        n_points = lidar_points.shape[1]
        n_train = int(n_points * split_ratio)
        indices = np.random.permutation(n_points)
        train_indices = indices[:n_train]
        val_indices = indices[n_train:]
        
        uv_train = lidar_points[:, train_indices]
        depth_lidar_train = depth_lidar[train_indices]
        depth_vals_train = depth[uv_train[1], uv_train[0]]
        
        uv_val = lidar_points[:, val_indices]
        depth_lidar_val = depth_lidar[val_indices]
        depth_vals_val = depth[uv_val[1], uv_val[0]]
        
        residuals = depth_lidar_train - depth_vals_train
        q1, q3 = np.percentile(residuals, [25, 75])
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        mask = (residuals >= lower) & (residuals <= upper)
        uv_train_f = uv_train[:, mask]
        d_lidar_f = depth_lidar_train[mask]
        d_pred_f = depth_vals_train[mask]

        X_train = d_pred_f.reshape(-1, 1)
        y_train = d_lidar_f.reshape(-1, 1)
        calibrator = self.calibrator
        calibrator.fit(X_train, y_train)
        
        preds_val = calibrator.predict(depth_vals_val.reshape(-1, 1))
        mape = mean_absolute_percentage_error(depth_lidar_val, preds_val)
        mae = mean_absolute_error(depth_lidar_val, preds_val)
        
        calibrated_depth = calibrator.predict(depth.reshape(-1, 1)).reshape(depth.shape)
        
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            np.save(os.path.join(save_dir, 'depth_map.npy'), calibrated_depth)
        if visualize:
            plt.figure(figsize=(15, 5))
            
            plt.subplot(1, 3, 1)
            plt.imshow(depth)
            plt.title('Original Depth')
            plt.colorbar()
            
            plt.subplot(1, 3, 2)
            plt.imshow(calibrated_depth)
            plt.title('Calibrated Depth')
            plt.colorbar()
            
            plt.subplot(1, 3, 3)
            plt.scatter(depth_vals_val, depth_lidar_val, alpha=0.3, label='Actual')
            plt.scatter(depth_vals_val, preds_val, alpha=0.3, label='Predicted')
            plt.xlabel('Predicted Depth')
            plt.ylabel('True Depth')
            plt.legend()
            plt.title('Depth Comparison')
            
            plt.tight_layout()
            plt.show()

        return calibrated_depth, {'mae': mae, 'mape': mape}

    def get_depth_map(self, img: np.ndarray, point_cloud: np.ndarray) -> np.ndarray:
        depth = self._infer_model(img)
        uv, depth_lidar = self.project_points_to_camera(point_cloud)
        return self._calibrate_depth_map(depth, uv, depth_lidar)
    
    def evaluate_on_dataset(self, dataset_dir: str, calibrator_type: str = 'both', 
                            split_ratio: float = 0.8) -> Dict[str, Dict[str, List[float]]]:
        calibrators = {}
        if calibrator_type in ['linear', 'both']:
            calibrators['linear'] = LinearRegression()
        if calibrator_type in ['exponential', 'both']:
            calibrators['exponential'] = ExponentialCalibrator()
        
        metrics = {name: {'mae': [], 'mape': []} for name in calibrators.keys()}
        
        img_dir = os.path.join(dataset_dir, 'img')
        pc_dir = os.path.join(dataset_dir, 'pc')
        img_files = sorted(glob.glob(os.path.join(img_dir, '*.png')))
        
        for img_path in tqdm(img_files):
            img = plt.imread(img_path)
            
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            pc_path = os.path.join(pc_dir, f'{base_name}.bin')
            point_cloud = np.fromfile(pc_path).reshape(-1, 4)
            points = point_cloud[:, :3].T
            
            depth = self._infer_model(img)
            
            uv, depth_lidar = self.project_points_to_camera(points)
            
            if uv.shape[1] < 10:
                continue
            
            for name, calibrator in calibrators.items():
                self.calibrator = calibrator
                try:
                    _, img_metrics = self._calibrate_depth_map(
                        depth, uv, depth_lidar, 
                        split_ratio=split_ratio
                    )
                    metrics[name]['mae'].append(img_metrics['mae'])
                    metrics[name]['mape'].append(img_metrics['mape'])
                except Exception as e:
                    print(f'Error processing {base_name} with {name}: {str(e)}')
        
        results = {}
        for name, values in metrics.items():
            mae_values = np.array(values['mae'])
            mape_values = np.array(values['mape'])
            
            results[name] = {
                'mean_mae': np.mean(mae_values),
                'median_mae': np.median(mae_values),
                'mean_mape': np.mean(mape_values),
                'median_mape': np.median(mape_values),
                'mae_values': mae_values.tolist(),
                'mape_values': mape_values.tolist()
            }
        
        return results

    @staticmethod
    def print_metrics(results: Dict[str, Dict[str, Union[float, List[float]]]]):
        for calibrator_name, metrics in results.items():
            print(f"\n{calibrator_name.capitalize()} Calibrator Performance:")
            print(f"  MAE:  Mean = {metrics['mean_mae']:.4f}, Median = {metrics['median_mae']:.4f}")
            print(f"  MAPE: Mean = {metrics['mean_mape'] * 100:.2f}%, Median = {metrics['median_mape'] * 100:.2f}%")
