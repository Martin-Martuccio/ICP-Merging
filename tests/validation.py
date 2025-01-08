# to validate the implementationss

import open3d as o3d
import numpy as np
import plotly.graph_objects as go
from io import load_ply

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

def visualize_point_clouds(source, target, transformed_source=None, differences=None):
    """
    Visualize source, target, and transformed point clouds using Plotly.
    """
    # Create a 3D scatter plot for the source point cloud (red)
    source_plot = point_cloud_to_plotly(source, 'red')

    # Create a 3D scatter plot for the target point cloud (green)
    target_plot = point_cloud_to_plotly(target, 'green')

    # Create a 3D scatter plot for the transformed source point cloud (blue)
    if transformed_source is not None:
        transformed_plot = point_cloud_to_plotly(transformed_source, 'blue')
    else:
        transformed_plot = None

    # Create a 3D scatter plot for the differences (yellow)
    if differences is not None:
        differences_plot = point_cloud_to_plotly(differences, 'yellow')
    else:
        differences_plot = None

    # Create the figure
    fig = go.Figure()

    # Add the plots to the figure
    fig.add_trace(source_plot)
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
        margin=dict(l=0, r=0, b=0, t=0)
    )

    # Show the figure
    fig.show()

def validate_icp(source_path, target_path, transformed_source_path=None, differences_path=None):
    """
    Validate the ICP alignment by visualizing the point clouds.
    """
    # Load point clouds
    source = load_ply(source_path)
    target = load_ply(target_path)

    # Load transformed source and differences if provided
    transformed_source = load_ply(transformed_source_path) if transformed_source_path else None
    differences = load_ply(differences_path) if differences_path else None

    # Visualize the point clouds
    visualize_point_clouds(source, target, transformed_source, differences)

if __name__ == "__main__":
    # Paths to your PLY files
    source_path = "data/input/model1.ply"
    target_path = "data/input/model2.ply"
    transformed_source_path = "data/output/transformed_model.ply"  # Optional
    differences_path = "data/output/differences.ply"  # Optional

    # Validate the ICP alignment
    validate_icp(source_path, target_path, transformed_source_path, differences_path)