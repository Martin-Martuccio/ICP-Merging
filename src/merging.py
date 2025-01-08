# merging the models
import open3d as o3d
import numpy as np

def merge_point_clouds(source, target):
    """Merge two point clouds into one."""
    return source + target

def highlight_differences(source, target, threshold):
    """Highlight differences between two point clouds."""
    # Compute distances between corresponding points
    distances = source.compute_point_cloud_distance(target)
    distances = np.asarray(distances)

    # Highlight points exceeding the threshold
    colors = np.asarray(source.colors)
    colors[distances > threshold] = [1, 0, 0]  # Red for differences
    source.colors = o3d.utility.Vector3dVector(colors)