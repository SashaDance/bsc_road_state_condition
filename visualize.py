import numpy as np
import rerun as rr
from PIL import Image

def get_point_cloud(depth_map: np.ndarray, camera_matrix: np.ndarray) -> np.ndarray:
    """
    Returns point cloud of an image

    Args:
        depth_map: a depth map of an image in target units
        camera_matrix: focal length of a camera in meters

    Returns:
        Stacked real x, y and z coordinates
    """
    h, w = depth_map.shape
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    u = (x - camera_matrix[0][2]) / camera_matrix[0, 0]
    v = (y - camera_matrix[1][2]) / camera_matrix[1, 1]
    cos_angels = 1 / np.sqrt(u ** 2 + v ** 2 + 1)
    z = np.multiply(cos_angels, depth_map)
    point_cloud = np.stack(
        (np.multiply(u, z), np.multiply(v, z), z)  # 3, h, w
    )
    return point_cloud

def visualize_rerun_from_paths(
    image_path: str,
    bin_path: str,
    depth_path: str,
    K: np.ndarray,
):
    img = np.asarray(Image.open(image_path))
    pc = np.fromfile(bin_path)
    pc = pc.reshape((pc.shape[0] // 4, 4))
    initial_pc = pc[:, :3]
    depth_map = np.fromfile(depth_path, dtype=np.float32).reshape((img.shape[0], img.shape[1]))
    dense_pc = get_point_cloud(depth_map, K).reshape((3, img.shape[0] * img.shape[1])).T

    rr.init("depth_densification_run", spawn=True)

    rr.log("view/rgb_image", rr.Image(img))

    rr.log("view/final_depth", rr.Image(depth_map))

    # rr.log("world/sparse_lidar", rr.Points3D(initial_pc))

    rr.log("world/dense_fused", rr.Points3D(dense_pc))

K = np.array([
    [954.398, 0, 628.246],
    [0, 954.398, 354.768],
    [0, 0, 1],
])

i = 37
img_pth = f"depth_map_dataset/img/{i}.png"
bin_path = f"depth_map_dataset/pc/{i}.bin"
depth_path = f'saved_depth/depth_map_{i}.bin'

visualize_rerun_from_paths(
    image_path=img_pth,
    bin_path =bin_path,
    depth_path=depth_path,
    K=K,
)
