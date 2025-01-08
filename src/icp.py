import open3d as o3d
import numpy as np

def perform_icp(source, target, voxel_size, max_iterations=200):
    """
    Perform ICP alignment between two point clouds.
    Adapted from: https://medium.com/@BlanchR2/point-cloud-alignment-in-open3d-using-the-iterative-closest-point-icp-algorithm-22433693aa8a
    """
    # Downsample point clouds
    source_down = source.voxel_down_sample(voxel_size)
    target_down = target.voxel_down_sample(voxel_size)

    # Estimate normals (required for point-to-plane ICP)
    source_down.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30)
    )
    target_down.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30)
    )

    # Set ICP parameters
    threshold = voxel_size * 1.5  # Maximum correspondence distance
    trans_init = np.identity(4)   # Initial transformation (identity matrix)

    # Perform ICP registration
    reg_result = o3d.pipelines.registration.registration_icp(
        source_down, target_down, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iterations)
    )

    # Return the transformation matrix
    return reg_result.transformation