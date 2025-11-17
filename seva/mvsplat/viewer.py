import open3d as o3d

pc = o3d.io.read_point_cloud("gaussians.ply")
o3d.visualization.draw_geometries([pc])
