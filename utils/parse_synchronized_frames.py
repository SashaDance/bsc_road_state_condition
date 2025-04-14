from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from cv_bridge import CvBridge
from typing import Generator, Tuple
import sqlite3
import numpy as np
import sensor_msgs_py.point_cloud2 as pc2

def find_closest_frames(cursor: sqlite3.Cursor,
                        img_topic_name: str = '/zed/zed_node/left/image_rect_color/compressed',
                        pc_topic_name: str = '/lidar/points',
                        sort_by_diff: bool = False) -> list[tuple[int, int, float]]:
    cursor.execute(f'''
        SELECT timestamp from messages 
        where topic_id = (select id from topics where name = '{img_topic_name}')
    ''')
    timestamps_img = cursor.fetchall()
    cursor.execute(f'''
        SELECT timestamp from messages 
        where topic_id = (select id from topics where name = '{pc_topic_name}')
    ''')
    timestamps_pcd = cursor.fetchall()
    pc_timestamps =  [tm[0] for tm in timestamps_pcd]
    img_timestamps = np.array([tm for tm in timestamps_img])
    results = []
    for pc_timestamp in pc_timestamps:
        closet_img_timestamp = img_timestamps[np.abs(img_timestamps - pc_timestamp).argmin()][0]
        diff = abs(pc_timestamp - closet_img_timestamp) / 1e9  # seconds
        results.append((closet_img_timestamp, pc_timestamp, diff))
    
    print(f'Found {len(results)} frames')
    
    if sort_by_diff:
        return sorted(results, key=lambda elem: elem[2])
    else:
        return results

def convert_pc(point_cloud: np.ndarray, 
               elev_range: list[float, float], 
               azim_range: list[float, float],
               num_elev: int, 
               num_azim: int = 480) -> np.ndarray:
    assert point_cloud.shape[1] == 3, f'Expected pc shape to be (n, 3), got {point_cloud.shape}'    
    point_cloud = point_cloud.reshape(num_azim, num_elev, 3)
    neg_mask = np.where(point_cloud[:, :, 0] >= 0, 1, -1)
    distance_matrix = (
        np.sqrt(point_cloud[:, :, 0] ** 2 + point_cloud[:, :, 1] ** 2 + point_cloud[:, :, 2] ** 2)
        * neg_mask  # keep negative values
    )
    elev_step = (elev_range[1] - elev_range[0]) / num_elev
    azim_step = (azim_range[1] - azim_range[0]) / num_azim
    elev = np.arange(elev_range[0], elev_range[1], elev_step)[::-1]
    azim = np.arange(azim_range[0], azim_range[1], azim_step)[::-1]
    elev_mat, azim_mat = np.meshgrid(np.deg2rad(elev), np.deg2rad(azim))
    x = (distance_matrix * np.cos(elev_mat) * np.cos(azim_mat)).flatten()
    y = (distance_matrix * np.cos(elev_mat) * np.sin(azim_mat)).flatten()
    z = (distance_matrix * np.sin(elev_mat)).flatten()
    
    return np.stack((x, y, z), axis=-1).reshape(-1, 3)

def get_image_pointcloud_pairs(bag_path: str,
                               max_frames: int = None,
                               sort_by_diff: bool = False
                               ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
    """
    Yields pairs of (image, point cloud) from the rosbag database.
    
    Args:
        bag_path: Path to the rosbag SQLite database file
        max_frames: Maximum number of frame pairs to yield (None for all)
        sort_by_diff: If True returns top max_frames best synchronized frames
    
    Yields:
        Tuple containing image and point cloud
    """
    conn = sqlite3.connect(bag_path)
    cursor = conn.cursor()
    all_frames = find_closest_frames(cursor, sort_by_diff=sort_by_diff)
    bridge = CvBridge()
    
    frames = all_frames[:max_frames] if max_frames is not None else all_frames
    
    for img_tm, pc_tm, _ in frames:
        cursor.execute(f'''
            SELECT data from messages 
            where topic_id = (select id from topics where name = '/zed/zed_node/left/image_rect_color/compressed')
            and timestamp = {img_tm}
        ''')
        img_row = cursor.fetchall()
        msg_type = get_message('sensor_msgs/msg/CompressedImage')
        image_msg = deserialize_message(img_row[0][0], msg_type)
        cv_image = bridge.compressed_imgmsg_to_cv2(image_msg, desired_encoding='bgr8')
        
        cursor.execute(f'''
            SELECT data from messages 
            where topic_id = (select id from topics where name = '/lidar/points') and timestamp = {pc_tm}
        ''')
        pc_row = cursor.fetchall()
        msg_type = get_message('sensor_msgs/msg/PointCloud2')
        point_cloud_msg = deserialize_message(pc_row[0][0], msg_type)
        pc = pc2.read_points(point_cloud_msg, field_names=('x', 'y', 'z'))
        xyz = convert_pc(np.stack((pc['x'], pc['y'], pc['z']), axis=-1), [-12.85, 7.85], [-60, 60], 96)
        
        yield cv_image, xyz
    
    conn.close()

    # TODO: rewrite this function so it reads rosbag with multiple .db3 files