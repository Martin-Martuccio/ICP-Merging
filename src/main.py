# here we have to load PLY files , performing ICP, merging models, and highlighting differences.
from io_handler import load_ply_as_point_cloud, save_point_cloud_as_ply
from preprocessing import downsample_point_cloud, estimate_normals
from icp import perform_icp, calculate_centroid
from merging import merge_point_clouds, highlight_differences

import open3d as o3d
import numpy as np

if __name__ == "__main__":
    # Load point clouds
    #source = load_ply_as_point_cloud("data/input/Pie_v1.ply")
    #target = load_ply_as_point_cloud("data/input/Pie_v2.ply")

    bunny1 = o3d.data.BunnyMesh()
    bunny2 = o3d.data.BunnyMesh()
    source = o3d.io.read_point_cloud(bunny1.path)
    target = o3d.io.read_point_cloud(bunny2.path)

    # Preprocess point clouds
    voxel_size = 0.002
    source = downsample_point_cloud(source, voxel_size)
    target = downsample_point_cloud(target, voxel_size)
    estimate_normals(source, radius=0.1, max_nn=30)
    estimate_normals(target, radius=0.1, max_nn=30)

    #####
    rotation_matrix = o3d.geometry.get_rotation_matrix_from_xyz((np.pi / 2, np.pi / 2, np.pi / 2))
   
    # Apply the translation to the point cloud
    loaded_target = target.translate((1, 0, 0))
 
    # Apply the rotation to the point cloud
    loaded_target = loaded_target.rotate(rotation_matrix, center=calculate_centroid(loaded_target))

    # Apply random scaling
    loaded_target = loaded_target.scale(0.5, center=calculate_centroid(loaded_target))
 
    o3d.visualization.draw_geometries([source, loaded_target], window_name="Original Alignment")
    ######

    # Perform ICP alignment
    target = perform_icp(source, target, voxel_size)
    
    # Merge point clouds
    #merged_cloud = merge_point_clouds(source, target)

    # Highlight differences
    #highlight_differences(source, target, threshold=0.05)

    # Save merged point cloud
    #save_point_cloud_as_ply(merged_cloud, "data/output/merged_model.ply")
