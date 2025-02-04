# merging the models
import open3d as o3d
import numpy as np
import plyfile
from plyfile import PlyData, PlyElement
from preprocessing import downsample_point_cloud, estimate_normals
from io_handler import load_ply_as_point_cloud, save_point_cloud_as_ply
from icp import perform_icp

# Function to apply a transformation (rotation, scaling, translation)
def apply_transformation(points, scale, translation, rotation_matrix):
    # Apply scaling
    points = points * scale
    # Apply rotation
    for point in points:
        position_homogeneous = np.array([*point, 1])
        point = np.dot(rotation_matrix, position_homogeneous)[:3]
    # Apply translation
    points = points + translation
    return points

# Function to compare points and change color
def compare_and_color(points1, points2, colors1, colors2, threshold=0.001):
    # Ensure points are in Open3D PointCloud format
    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(points1)
    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(points2)
    
    # Build KDTree for points2
    tree2 = o3d.geometry.KDTreeFlann(pcd2)
    # Find points in points1 without a match in points2
    mismatch_indices_p1 = []
    for i in range(len(points1)):
        [_, idx, dist] = tree2.search_knn_vector_3d(pcd1.points[i], 1)
        if dist[0] > threshold:
            mismatch_indices_p1.append(i)
    
    # Build KDTree for points1
    tree1 = o3d.geometry.KDTreeFlann(pcd1)
    # Find points in points2 without a match in points1
    mismatch_indices_p2 = []
    for i in range(len(points2)):
        [_, idx, dist] = tree1.search_knn_vector_3d(pcd2.points[i], 1)
        if dist[0] > threshold:
            mismatch_indices_p2.append(i)
    
    # Mark mismatched points in colors1 and colors2
    colors1[mismatch_indices_p1] = [255, 0, 0]  # Red for points in points1 without a match (missing/removed parts)
    colors2[mismatch_indices_p2] = [0, 255, 0]  # Green for points in points2 without a match (extra/added parts)
    
    return colors1, colors2

# Loading a PLY file features (coordinates and colors)
def load_plydata(file_path, default_color=[171, 171, 171]):
    ply = PlyData.read(file_path)
    points = np.vstack([ply['vertex']['x'], ply['vertex']['y'], ply['vertex']['z']]).T
    try:
        colors = np.vstack([ply['vertex']['red'], ply['vertex']['green'], ply['vertex']['blue']]).T
    except ValueError:
        colors = points.copy()
        for i in range(len(colors)):
            colors[i] = default_color
    return points, colors

# Compute the transformation (scaling , translation and rotation) needed to align two point clouds
def compute_transformation(source_path, target_path, voxel_size):
    source_point_cloud = load_ply_as_point_cloud(source_path)
    target_point_cloud = load_ply_as_point_cloud(target_path)

    source_downsample = downsample_point_cloud(source_point_cloud, voxel_size)
    target_downsample = downsample_point_cloud(target_point_cloud, voxel_size)

    result_scale, result_trl, result_rot = perform_icp(source_downsample, target_downsample, voxel_size, apply_transformation=False)
    return result_scale, result_trl, result_rot

