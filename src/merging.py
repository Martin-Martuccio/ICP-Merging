# merging the models
import open3d as o3d
import numpy as np
import plyfile
from plyfile import PlyData, PlyElement
from preprocessing import downsample_point_cloud, estimate_normals
from io_handler import load_ply_as_point_cloud, save_point_cloud_as_ply
from icp import perform_icp

def apply2file(file_path, scale, traslation, rotation):
    
    pcd = load_ply_as_point_cloud(file_path)
    
    pcd.scale(scale)
    pcd.translate(traslation)
    pcd.transform(rotation)

    save_point_cloud_as_ply(pcd, file_path)

# Funzione per applicare una trasformazione (rotazione, scalamento, traslazione)
def apply_transformation(points, scale, translation, rotation_matrix):
    # Applica scalamento
    points = points * scale
    # Applica traslazione
    points = points + translation
    # Applica rotazione
    for point in points:
        position_homogeneous = np.array([*point, 1])
        point = np.dot(rotation_matrix, position_homogeneous)[:3]
    return points

# Function to compare points and change color
def compare_and_color(points1, points2, colors1, colors2, threshold=0.01):
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
    colors1[mismatch_indices_p1] = [255, 0, 0]  # Red for points in points1 without a match
    colors2[mismatch_indices_p2] = [0, 255, 0]  # Green for points in points2 without a match
    
    return colors1, colors2

##

def load_plydata(file_path):
    ply = PlyData.read(file_path)
    points = np.vstack([ply['vertex']['x'], ply['vertex']['y'], ply['vertex']['z']]).T
    colors = np.vstack([ply['vertex']['red'], ply['vertex']['green'], ply['vertex']['blue']]).T
    return points, colors

def compute_transformation(source_path, target_path, voxel_size):
    source_point_cloud = load_ply_as_point_cloud(source_path)
    target_point_cloud = load_ply_as_point_cloud(target_path)

    source_downsample = downsample_point_cloud(source_point_cloud, voxel_size)
    target_downsample = downsample_point_cloud(target_point_cloud, voxel_size)

    result_scale, result_trl, result_rot = perform_icp(source_downsample, target_downsample, voxel_size, apply_transformation=False)
    return result_scale, result_trl, result_rot


source_path = "../data/input/pie1_ASCII.ply"
target_path = "../data/input/pie1_ASCII2.ply"
voxel_parameter = 0.05 

# Carica il primo file PLY
ply1_points, ply1_colors = load_plydata(source_path)

# Carica il secondo file PLY
ply2_points, ply2_colors = load_plydata(target_path)

#result_scale, result_trl, result_rot = compute_transformation(source_path, target_path, voxel_parameter)

# Applica la trasformazione al secondo modello
#ply2_points = apply_transformation(ply2_points, result_scale, result_trl, result_rot)

# Confronta i punti e cambia colore
ply1_colors, ply2_colors = compare_and_color(ply1_points, ply2_points, ply1_colors, ply2_colors, threshold=0.00001)

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
merged_ply.write("merged_model_4.ply")

pcd = load_ply_as_point_cloud("merged_model_4.ply")
o3d.visualization.draw_geometries([pcd], window_name="Merged Model")