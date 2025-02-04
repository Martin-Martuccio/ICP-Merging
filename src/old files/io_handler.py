# handle the input and output of the program
 
import open3d as o3d
 
def load_ply_as_point_cloud(file_path):
    """
    Load a PLY file as a point cloud using Open3D.
    """
    try:
        # Load the PLY file
        pcd = o3d.io.read_point_cloud(file_path)
        if not pcd.has_points():
            raise ValueError("The file does not contain any points.")
        return pcd
    except Exception as e:
        print(f"Error loading PLY file: {e}")
        return None
 
def save_point_cloud_as_ply(pcd, file_path):
    """
    Save a point cloud to a PLY file using Open3D.
    """
    try:
        o3d.io.write_point_cloud(file_path, pcd)
        print(f"Point cloud saved to {file_path}")
    except Exception as e:
        print(f"Error saving PLY file: {e}")
 
def load_ply_as_mesh(file_path):
    """
    Load a PLY file as a mesh using Open3D.
    """
    try:
        mesh = o3d.io.read_triangle_mesh(file_path)
        if not mesh.has_vertices():
            raise ValueError("The file does not contain any vertices.")
        return mesh
    except Exception as e:
        print(f"Error loading PLY file as mesh: {e}")
        return None
 
def save_mesh_as_ply(mesh, file_path):
    """
    Save a mesh to a PLY file using Open3D.
    """
    try:
        o3d.io.write_triangle_mesh(file_path, mesh)
        print(f"Mesh saved to {file_path}")
    except Exception as e:
        print(f"Error saving mesh to PLY file: {e}")