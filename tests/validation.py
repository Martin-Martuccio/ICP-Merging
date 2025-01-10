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
def point_cloud_to_plotly(pcd, color, name_fig="Point Cloud"):
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
        name=name_fig
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
    source_plot = point_cloud_to_plotly(source, 'red', "Source Point Cloud")

    # Create a 3D scatter plot for the target point cloud (green)
    target_plot = None
    if target is not None:
        target_plot = point_cloud_to_plotly(target, 'green', "Target Point Cloud")

    # Create a 3D scatter plot for the transformed source point cloud (blue)
    transformed_plot = None
    if transformed_source is not None:
        transformed_plot = point_cloud_to_plotly(transformed_source, 'blue', "Transformed Point Cloud")

    # Create a 3D scatter plot for the differences (yellow)
    differences_plot = None
    if differences is not None:
        differences_plot = point_cloud_to_plotly(differences, 'yellow', "Differences Point Cloud")

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

def apply_transformation(pcd, angle_degrees, axis='x', translation=None):
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
    print("Angle (rad): ", angle_radians)
    
    # Initialize rotation angles for x, y, z
    rotation_x, rotation_y, rotation_z = 0.0, 0.0, 0.0
    
    # Set the rotation angle for the specified axis
    if axis == 'x':
        rotation_x = angle_radians
    elif axis == 'y':
        rotation_y = angle_radians
    elif axis == 'z':
        rotation_z = angle_radians
    else:
        raise ValueError("Axis must be 'x', 'y', or 'z'.")
    
    # Get the rotation matrix
    rotation_matrix = o3d.geometry.get_rotation_matrix_from_xyz((rotation_x, rotation_y, rotation_z))

    centroid = calculate_centroid(pcd)
    print("Before rotation : ", centroid)
    
    # Apply the rotation to the point cloud
    pcd = pcd.rotate(rotation_matrix, center=centroid)

    if translation is not None:
        pcd = pcd.translate(translation)
        new_centroid = calculate_centroid(pcd)
        print("After rotation and translation : ", new_centroid)
    
    return pcd

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



######### MAIN #########

if __name__ == "__main__":
    # Paths to your PLY files
    source_path = "..\data\input\oct23.ply"
    target_path = "..\data\input\oct24.ply"

    # loading the point cloud and downsample it
    loaded_source = load_ply_as_point_cloud(source_path)
    downsampled_source = downsample_point_cloud(loaded_source, voxel_size=0.5)

    # loading the point cloud and downsample it (for translation and rotation)
    loaded_source_2 = load_ply_as_point_cloud(source_path)
    downsampled_source_2 = downsample_point_cloud(loaded_source_2, 0.5)

    # Apply a transformation (e.g., tilt by 65 degrees around the X-axis)
    transformed_source = apply_transformation(downsampled_source_2, angle_degrees=65, axis='z',translation=np.array([2, 2, 2]))

    # Debug: print the first 5 points of both point clouds to check if they are different
    # print("Original points (first 5):\n", np.asarray(downsampled_source.points)[:5])
    # print("Transformed points (first 5):\n", np.asarray(transformed_source.points)[:5])

    # Visualize the original and transformed point clouds
    """
    The transformation to Scatter3D is done inside the function visualize_multiple_point_clouds
    """
    visualize_multiple_point_clouds(
        source=downsampled_source,                          # point cloud
        target=None,
        transformed_source=transformed_source,              # point cloud
        differences=None,
        title="Original and Transformed Point Clouds"
    )