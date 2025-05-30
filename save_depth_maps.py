import random
import numpy as np
from PIL import Image
from evaluate_depth_interpolation import calibrator

for i in random.sample(list(range(4172)), 100):
    img_pth = f"depth_map_dataset/img/{i}.png"
    bin_path = f"depth_map_dataset/pc/{i}.bin"
    depth_path = f'saved_depth/depth_map_{i}.bin'
    img = np.asarray(Image.open(img_pth))
    pc = np.fromfile(bin_path).reshape(-1, 4)
    points = pc[:, :3].T

    d_pred = calibrator._infer_model(img)
    uv, d_lidar = calibrator.project_points_to_camera(points)

    idx = np.random.permutation(uv.shape[1])
    n_train = int(uv.shape[1] * 0.85)
    train_idx, val_idx = idx[:n_train], idx[n_train:]

    d_fused, _ = calibrator.densify(
        d_pred,
        uv[:, train_idx],
        d_lidar[train_idx]
    )
    d_fused.tofile(depth_path)

# print(np.fromfile('saved_depth/depth_map_93.bin', dtype=np.float32).shape)