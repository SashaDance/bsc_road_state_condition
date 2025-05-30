import numpy as np
from depth_map.depth_map_interpolation import DepthInterpolation

projection_matrix = np.array([
    [954.398, 0, 628.246, 0],
    [0, 954.398, 354.768, 0],
    [0, 0, 1, 0],
])
calibrator = DepthInterpolation(
    model_path='/home/sashadance/python_projects/bsc_road_state_condition/models/depth_anything_v2_metric_vkitti_vits.pth',
    device='cpu',
    camera_resolution=(1280, 720),
    proj_matr=projection_matrix,
)

if __name__ == '__main__':
    results = calibrator.evaluate_on_dataset(
        '/home/sashadance/python_projects/bsc_road_state_condition/depth_map_dataset',
        split_ratio=0.8
    )

    DepthInterpolation.print_metrics(results)