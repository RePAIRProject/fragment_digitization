import numpy as np
import open3d as o3d
from utils.mesh2pcd import PcdSampler


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
    mesh = o3d.io.read_triangle_mesh("../data/Tombstone1.obj", enable_post_processing=True, print_progress=True)
    o3d.visualization.draw_geometries([mesh])

    xyzs = np.asarray(mesh.vertices)
    nxyzs = np.asarray(mesh.vertex_normals)
    uvs = np.asarray(mesh.triangle_uvs)
    faces = np.asarray(mesh.triangles)

    # creating mesh_file and face_file self-defined
    mesh_file = "../data/mesh_file"
    face_file = "../data/face_file"

    with open(mesh_file, 'w') as f:
        for xyz, nxyz, uv in zip(xyzs, nxyzs, uvs):
            xyz_str = " ".join([str(x) for x in xyz.tolist()])
            nxyz_str = " ".join([str(x) for x in nxyz.tolist()])
            uv_str = " ".join([str(x) for x in uv.tolist()])
            data_str = [xyz_str, nxyz_str, uv_str]
            line = ','.join(data_str)
            f.write(line + '\n')

    with open("../data/face_file", 'w') as f:
        for face in faces:
            data = face.tolist()
            data_str = [str(x) for x in data]
            line = ' '.join(data_str)
            f.write(line + '\n')

    pcd_obj = PcdSampler(
        mesh_file=mesh_file,
        face_file=face_file,
        img_file="../data/Tombstone1_low.jpg"
    )
    # pcd, colors = pcd_obj.sample_surface_even(xyzs.shape[0])  # given sampling points you want
    pcd, colors = pcd_obj.sample_surface_even(1000000)  # given sampling points you want
    pcd_obj.show_points_cloud(pcd, colors)  # show pointcloud wiht color sampling

if __name__ == "__main__":
    main()