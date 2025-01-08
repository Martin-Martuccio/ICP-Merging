# handle the input and output of the program

import open3d as o3d

def load_ply(file_path):
    """Load a PLY file into an Open3D point cloud."""
    return o3d.io.read_point_cloud(file_path)

def save_ply(point_cloud, file_path):
    """Save an Open3D point cloud to a PLY file."""
    o3d.io.write_point_cloud(file_path, point_cloud)