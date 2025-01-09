import open3d as o3d
import numpy as np

def calculate_centroid(pcd):
    """
    Calculate the centroid of a point cloud.
    
    Args:
        pcd (open3d.geometry.PointCloud): The input point cloud.
    
    Returns:
        numpy.ndarray: The centroid as a numpy array [x, y, z].
    """
    # Convert the point cloud to a numpy array
    points = np.asarray(pcd.points)
    
    # Calculate the mean (centroid) along each axis
    centroid = np.mean(points, axis=0)
    
    return centroid

def compute_fpfh_features(pcd, voxel_size):
    radius_normal = voxel_size * 2
    radius_feature = voxel_size * 5

    # Calculate the normal
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    # Calculate FPFH descriptors
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return fpfh

def perform_icp(source, target, voxel_size, max_iterations=200, rotation_threshold=0.02, voxel_size_threshold=1.5, apply_transformation=True):
    """
    Perform ICP alignment between two point clouds.
    Adapted from: https://medium.com/@BlanchR2/point-cloud-alignment-in-open3d-using-the-iterative-closest-point-icp-algorithm-22433693aa8a
    """

    # Resolve scaling
    source_points = np.asarray(source.points)
    source_centroid = calculate_centroid(source)
    source_distances = np.linalg.norm(source_points - source_centroid, axis=1)
    source_max_dist = np.max(source_distances)

    target_points = np.asarray(target.points)
    target_centroid = calculate_centroid(target)
    target_distances = np.linalg.norm(target_points - target_centroid, axis=1)
    target_max_dist = np.max(target_distances)
    
    result_scale = source_max_dist / target_max_dist
        
    # Apply the final scaling
    print("Scaling Factor:", result_scale)
    if apply_transformation:
        target.scale(result_scale, center=target_centroid)
    o3d.visualization.draw_geometries([source, target], window_name="Scaling Alignment")

    # Estimate normals (required for point-to-plane ICP)
    source.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
    )
    target.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
    )
    
    fpfh1 = compute_fpfh_features(source, voxel_size)
    fpfh2 = compute_fpfh_features(target, voxel_size)

    distance_threshold = voxel_size * voxel_size_threshold
    result = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
        source, target, fpfh1, fpfh2,
        o3d.pipelines.registration.FastGlobalRegistrationOption(maximum_correspondence_distance=distance_threshold)
    )

    result_rot = o3d.pipelines.registration.registration_icp(
        target, source, rotation_threshold,
        result.transformation,  # Initial Trasformation with FGR
        o3d.pipelines.registration.TransformationEstimationPointToPlane()
    )
        
    # Apply the final rotation
    print("Rotation Matrix:", result_rot.transformation)
    if apply_transformation:
        target.transform(result_rot.transformation)
    o3d.visualization.draw_geometries([source, target], window_name="Rotation Alignment")

    # Resolve traslation
    source_centroid = calculate_centroid(source)
    target_centroid = calculate_centroid(target)
    result_trl = source_centroid - target_centroid
        
    # Apply the final traslation
    print("Traslation Vector:", result_trl)
    if apply_transformation:
        target.translate(result_trl)
    o3d.visualization.draw_geometries([source, target], window_name="Translation Alignment")

    if apply_transformation:
        return target
    else:
        return result_scale, result_trl, result_rot
    