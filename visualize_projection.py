import numpy as np
from typing import Tuple
import open3d as o3d
import cv2
import matplotlib.pyplot as plt

proj_matrix = np.array([
    [954.398, 0, 628.246, 0],
    [0, 954.398, 354.768, 0],
    [0, 0, 1, 0],
])

img_path = 'depth_map_dataset/img/0.png'
pc_path = 'depth_map_dataset/pc/0.bin'


def project_points_to_camera(
    points: np.ndarray, proj_matrix: np.ndarray, cam_res: Tuple[int, int]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:  # Return original valid indices
    if points.shape[0] == 3:
        points = np.vstack((points, np.ones((1, points.shape[1]))))
    in_image = points[2, :] > 0  # Initial filter for points in front of the camera
    depths = points[2, in_image]
    uvw = np.dot(proj_matrix, points[:, in_image])
    uv = uvw[:2, :]
    w = uvw[2, :]
    uv[0, :] /= w
    uv[1, :] /= w
    valid = (uv[0, :] >= 0) & (uv[0, :] < cam_res[0]) & (uv[1, :] >= 0) & (uv[1, :] < cam_res[1])
    uv_valid = uv[:, valid].astype(int)
    depths_valid = depths[valid]
    # Get indices of original points that are valid
    original_indices = np.where(in_image)[0][valid]
    return uv_valid, depths_valid, original_indices

def depths_to_colors(depths: np.ndarray, max_depth: int = 10, cmap: str = "hsv") -> np.ndarray:
    depths /= max_depth
    to_colormap = plt.get_cmap(cmap)
    rgba_values = to_colormap(depths, bytes=True)
    return rgba_values[:, :3].astype(int)

points = np.fromfile(pc_path)
points = points.reshape((points.shape[0] // 4, 4)).T
points = np.vstack([points[0], points[1], points[2], np.ones((1, points[0].shape[0]))])

uv, depths, valid_indices = project_points_to_camera(points, proj_matrix, (1280, 720))

image = cv2.imread(img_path)
rgb_distances = depths_to_colors(depths, max_depth=10)
for point, d in zip(uv.T, rgb_distances):
    c = (int(d[0]), int(d[1]), int(d[2]))
    cv2.circle(image, point, radius=2, color=c, thickness=cv2.FILLED)

plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis(False)
plt.show()
