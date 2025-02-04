# preproessing and features extraction
import open3d as o3d

def downsample_point_cloud(point_cloud, voxel_size):
    """Downsample the point cloud using voxel grid filtering."""
    down_point_cloud = point_cloud.voxel_down_sample(voxel_size)
    down_point_cloud, _ = down_point_cloud.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0) # Statistical oulier removal
    return down_point_cloud

def estimate_normals(point_cloud, radius, max_nn):
    """Estimate normals for the point cloud."""
    point_cloud.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn)
    )