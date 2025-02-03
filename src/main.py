''' ### No more used ###
# here we have to load PLY files , performing ICP, merging models, and highlighting differences.
from io_handler import load_ply_as_point_cloud
from merging import compare_and_color, load_plydata, compute_transformation, apply_transformation
from plyfile import PlyData, PlyElement
import open3d as o3d
import numpy as np
'''
import SplatPLYHandler

if __name__ == "__main__":

    ''' ### No more used ###
    source_path = "../data/input/SatiroEBaccante_broken2.ply"
    target_path = "../data/input/SatiroEBaccante_broken.ply"
    voxel_parameter = 0.01 

    # Loading the first PLY file
    ply1_points, ply1_colors = load_plydata(source_path)

    # Loading the second PLY file
    ply2_points, ply2_colors = load_plydata(target_path)

    # Applica la trasformazione al secondo modello
    result_scale, result_trl, result_rot = compute_transformation(source_path, target_path, voxel_parameter)
    ply2_points = apply_transformation(ply2_points, result_scale, result_trl, result_rot)

    # Confronta i punti e cambia colore
    ply1_colors, ply2_colors = compare_and_color(ply1_points, ply2_points, ply1_colors, ply2_colors)

    # Unisci i punti e i colori dei due modelli
    merged_points = np.vstack([ply1_points, ply2_points])
    merged_colors = np.vstack([ply1_colors, ply2_colors])

    # Crea un nuovo file PLY
    merged_vertices = np.zeros(merged_points.shape[0], dtype=[
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')
    ])

    merged_vertices['x'] = merged_points[:, 0]
    merged_vertices['y'] = merged_points[:, 1]
    merged_vertices['z'] = merged_points[:, 2]
    merged_vertices['red'] = merged_colors[:, 0]
    merged_vertices['green'] = merged_colors[:, 1]
    merged_vertices['blue'] = merged_colors[:, 2]

    # Salva il file PLY risultante
    merged_ply = PlyData([PlyElement.describe(merged_vertices, 'vertex')], text=True)
    merged_ply.write("../data/output/merged_model.ply")

    pcd = load_ply_as_point_cloud("../data/output/merged_model.ply")
    o3d.visualization.draw_geometries([pcd], window_name="Merged Model")
    '''

    source_path = "../data/input/SatiroEBaccante_broken2.ply"
    target_path = "../data/input/SatiroEBaccante_broken.ply"

    # Create an instance of SplatPLYHandler
    handler1 = SplatPLYHandler()
    handler2 = SplatPLYHandler()

    # Load two PLY files
    handler1.load_ply(source_path)
    handler2.load_ply(target_path)
    
    # Calculate the alignment
    transformation = handler2.align_icp(handler1)
    
    # Align the second PLY
    handler2.apply_transformation(matrix4d = transformation)
    
    # Compare the two PLY files
    merged_handler = handler1.compare_and_merge(handler2, mode = 1)
    
    # Save the resulting PLY file
    merged_handler.save_ply("../data/output/merged_model.ply")
