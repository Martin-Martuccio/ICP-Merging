# to validate the implementationss

import open3d as o3d
import numpy as np
import plotly.graph_objects as go
import sys
import os

# Add the 'src' directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

# Now you can import io_handler
from io_handler import load_ply_as_point_cloud


# reduce the point cloud size
def downsample_point_cloud(pcd, voxel_size):
    """
    Downsample a point cloud using a voxel grid.
    """
    return pcd.voxel_down_sample(voxel_size)

# Converting the point cloud to a Plotly 3D scatter plot.
def point_cloud_to_plotly(pcd, color):
    """
    Convert an Open3D point cloud to a Plotly 3D scatter plot.
    """
    points = np.asarray(pcd.points)
    return go.Scatter3d(
        x=points[:, 0],
        y=points[:, 1],
        z=points[:, 2],
        mode='markers',
        marker=dict(size=2, color=color),
        name="Point Cloud"
    )

# Visualizing point cloud in a 3D Plotly scatter plot.
def visualize_single_point_cloud(*plots, title="Point Cloud Visualization"):
    """
    Visualize multiple point clouds in a 3D Plotly scatter plot.
    """
    fig = go.Figure()

    for plot in plots:
        fig.add_trace(plot)

    fig.update_layout(
        scene=dict(
            xaxis=dict(title='X'),
            yaxis=dict(title='Y'),
            zaxis=dict(title='Z')
        ),
        title=title,
        margin=dict(l=0, r=0, b=0, t=30)
    )

    fig.show()

def visualize_multiple_point_clouds(source, target=None, transformed_source=None, differences=None, title="Point Cloud Visualization"):
    """
    Visualize source, target, and transformed point clouds using Plotly.
    
    Args:
        source (open3d.geometry.PointCloud): The source point cloud.
        target (open3d.geometry.PointCloud, optional): The target point cloud.
        transformed_source (open3d.geometry.PointCloud, optional): The transformed source point cloud.
        differences (open3d.geometry.PointCloud, optional): The differences between point clouds.
        title (str, optional): The title of the plot.
    """
    # Create a 3D scatter plot for the source point cloud (red)
    source_plot = point_cloud_to_plotly(source, 'red')

    # Create a 3D scatter plot for the target point cloud (green)
    target_plot = None
    if target is not None:
        target_plot = point_cloud_to_plotly(target, 'green')

    # Create a 3D scatter plot for the transformed source point cloud (blue)
    transformed_plot = None
    if transformed_source is not None:
        transformed_plot = point_cloud_to_plotly(transformed_source, 'blue')

    # Create a 3D scatter plot for the differences (yellow)
    differences_plot = None
    if differences is not None:
        differences_plot = point_cloud_to_plotly(differences, 'yellow')

    # Create the figure
    fig = go.Figure()

    # Add the plots to the figure
    fig.add_trace(source_plot)
    if target_plot is not None:
        fig.add_trace(target_plot)
    if transformed_plot is not None:
        fig.add_trace(transformed_plot)
    if differences_plot is not None:
        fig.add_trace(differences_plot)

    # Update layout for better visualization
    fig.update_layout(
        scene=dict(
            xaxis=dict(title='X'),
            yaxis=dict(title='Y'),
            zaxis=dict(title='Z')
        ),
        title=title,  # Add the title here
        margin=dict(l=0, r=0, b=0, t=30)  # Adjust top margin to accommodate the title
    )

    # Show the figure
    fig.show()

def apply_transformation(pcd, angle_degrees, axis='x'):
    """
    Apply a transformation (e.g., tilt) to a point cloud.
    
    Args:
        pcd (open3d.geometry.PointCloud): The input point cloud.
        angle_degrees (float): The angle of rotation in degrees.
        axis (str): The axis of rotation ('x', 'y', or 'z').
    
    Returns:
        open3d.geometry.PointCloud: The transformed point cloud.
    """
    # Convert the angle to radians
    angle_radians = np.radians(angle_degrees)
    
    # Create a rotation matrix based on the specified axis
    if axis == 'x':
        rotation_matrix = np.array([
            [1, 0, 0],
            [0, np.cos(angle_radians), -np.sin(angle_radians)],
            [0, np.sin(angle_radians), np.cos(angle_radians)]
        ])
    elif axis == 'y':
        rotation_matrix = np.array([
            [np.cos(angle_radians), 0, np.sin(angle_radians)],
            [0, 1, 0],
            [-np.sin(angle_radians), 0, np.cos(angle_radians)]
        ])
    elif axis == 'z':
        rotation_matrix = np.array([
            [np.cos(angle_radians), -np.sin(angle_radians), 0],
            [np.sin(angle_radians), np.cos(angle_radians), 0],
            [0, 0, 1]
        ])
    else:
        raise ValueError("Invalid axis. Choose 'x', 'y', or 'z'.")
    
    # Apply the rotation to the point cloud
    pcd.rotate(rotation_matrix, center=(0, 0, 0))
    
    return pcd


######### MAIN #########

if __name__ == "__main__":
    # Paths to your PLY files
    source_path = r"C:\Users\ACER\Desktop\MAGISTRALE\UNIGE MAGISTRALE 2 ANNO\AUGMENTED AND VIRTUAL REALITY\Project_Exam\Code\ICP-Merging\data\input\oct23.ply"
    target_path = r"C:\Users\ACER\Desktop\MAGISTRALE\UNIGE MAGISTRALE 2 ANNO\AUGMENTED AND VIRTUAL REALITY\Project_Exam\Code\ICP-Merging\data\input\oct24.ply"

    loaded_source = load_ply_as_point_cloud(source_path)
    down_loaded_source = downsample_point_cloud(loaded_source, 0.05)
    source_plot = point_cloud_to_plotly(down_loaded_source, 'red')

    # Downsample the source point cloud
    downsampled_source = downsample_point_cloud(loaded_source, voxel_size=0.25)

    # Apply a transformation (e.g., tilt by 15 degrees around the X-axis)
    transformed_source = apply_transformation(downsampled_source, angle_degrees=35, axis='y')

    # Visualize the original and transformed point clouds
    visualize_multiple_point_clouds(
        source=downsampled_source,
        transformed_source=transformed_source,
        title="Original and Transformed Point Clouds"
    )

    # visualize_single_point_cloud(source_plot)

    # transformed_source_path = "data/output/transformed_model.ply"  # Optional
    # differences_path = "data/output/differences.ply"  # Optional

    # Validate the ICP alignment
    # validate_icp(source_path, target_path, transformed_source_path, differences_path)