import numpy as np

import vedo as vd
import open3d as o3d


def visualize(mesh):
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(mesh)
    opt = vis.get_render_option()
    opt.show_coordinate_frame = True
    opt.background_color = np.asarray([0.5, 0.5, 0.5])
    vis.run()
    vis.destroy_window()

def main():
    mesh1 = o3d.io.read_triangle_mesh("../data/Tombstone1_p1.obj", enable_post_processing=True, print_progress=True)
    mesh2 = o3d.io.read_triangle_mesh("../data/Tombstone1_p2.obj", enable_post_processing=True, print_progress=True)
    # mesh = o3d.io.read_triangle_mesh("../data/Tombstone1.obj", enable_post_processing=True, print_progress=True)

    m1 = vd.Mesh("../data/Tombstone1_p1.obj").texture("../data/Tombstone1_low.jpg")
    texture_arr = m1.pointdata['material_0']
    pic = vd.Picture("../data/Tombstone1_low.jpg")
    color_arr = pic.tonumpy()

    x_y_coords = np.floor(texture_arr * color_arr.shape[0:2]).astype(int)
    color = color_arr[x_y_coords[:, 0], x_y_coords[:, 1], :]

    p1 = vd.Points(m1.points(), c=color)
    p1.normals = m1.normals()
    # p1.color(color)

    # texture_arr = m1.pointdata['material_0']
    #
    # pic = vd.Picture("../data/Tombstone1_low.jpg")
    # color_arr = pic.tonumpy()

    # print(pic.bounds())
    # print(color_arr.shape)

    # trimesh_m = trimesh.load("../data/Tombstone1_p1.obj", force='mesh')#, skip_texture=True)
    # vd.show(p1, __doc__, axes=1, viewup='z').close()
    vd.show(m1, "Mesh", at=0, N=2, axes=1)
    vd.show(p1, "PCD", at=1, interactive=1).close()

    # pcd1 = o3d.geometry.PointCloud(mesh1.vertices)
    # pcd1.normals = mesh1.vertex_normals
    # pcd1.colors = mesh1.vertex_colors
    # pcd2 = o3d.geometry.PointCloud(mesh2.vertices)
    # pcd2.normals = mesh2.vertex_normals
    # pcd2.colors = mesh2.vertex_colors
    #
    # o3d.visualization.draw_geometries([mesh1, mesh2])
    # # visualize([pcd1, pcd2])
    # visualize(mesh)

if __name__ == "__main__":
    main()