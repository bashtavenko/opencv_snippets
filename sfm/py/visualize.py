# Need
# export XDG_SESSION_TYPE=x11
# python3 -m visualize

import open3d as o3d

# Load the .ply file
pcd = o3d.io.read_point_cloud("/tmp/bottle.ply")

# Visualize the point cloud
o3d.visualization.draw_geometries([pcd])
