# here we have to load PLY files , performing ICP, merging models, and highlighting differences.
from io import load_ply, save_ply
from preprocessing import downsample_point_cloud, estimate_normals
from icp import perform_icp
from merging import merge_point_clouds, highlight_differences

def main():
    # Load point clouds
    source = load_ply("data/input/model1.ply")
    target = load_ply("data/input/model2.ply")

    # Preprocess point clouds
    voxel_size = 0.02
    source = downsample_point_cloud(source, voxel_size)
    target = downsample_point_cloud(target, voxel_size)
    estimate_normals(source, radius=0.1, max_nn=30)
    estimate_normals(target, radius=0.1, max_nn=30)

    # Perform ICP alignment
    transformation = perform_icp(source, target, voxel_size)
    source.transform(transformation)

    # Merge point clouds
    merged_cloud = merge_point_clouds(source, target)

    # Highlight differences
    highlight_differences(source, target, threshold=0.05)

    # Save merged point cloud
    save_ply(merged_cloud, "data/output/merged_model.ply")

if __name__ == "__main__":
    main()