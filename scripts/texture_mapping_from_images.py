import os
import pymeshlab
import pdb 
import vedo as vd

import re
import fileinput

import natsort
from wcmatch import glob

import copy
from utils.helpers import *

VOXEL_SIZE = 0.1 # if alignment is not good try to play with this value, it seems to be playing a role on the alignment (lower number more correspondences)
NOISE_BOUND = VOXEL_SIZE

# Visualize the registration results
def visualize_registration(pcds_, transformations_):
    for i in range(1, len(pcds_), 1):
        A_pcd_T_teaser = copy.deepcopy(pcds_[0]).transform(transformations_[i-1])
        o3d.visualization.draw_geometries([A_pcd_T_teaser, pcds_[i]])

# Visualize the registration results
def visualize_registration_T(pcds_, transformations_):
    for i in range(1, len(pcds_), 1):
        A_pcd_T_teaser = copy.deepcopy(pcds_[i]).transform(np.linalg.inv(transformations_[i - 1]))
        o3d.visualization.draw_geometries([A_pcd_T_teaser, pcds_[0]])

def create_pcds(mesh_models, visualize=False):
    pcds_ = []
    for mesh_model in mesh_models:
        pcd = o3d.geometry.PointCloud(mesh_model.vertices)
        pcd.normals = mesh_model.vertex_normals
        pcd.paint_uniform_color([np.round(np.random.uniform(low=0, high=1), 1), np.round(np.random.uniform(low=0, high=1), 1), np.round(np.random.uniform(low=0, high=1), 1)]) # give each point cloud a random color
        pcds_.append(pcd)

    if visualize:
        o3d.visualization.draw_geometries(pcds_)  # plot A and B

    return pcds_

# voxel downsample both clouds
def downsample_pcds(pcds_, visualize=False):
    down_pcds = []
    for pcd_ in pcds_:
        down_pcds.append(pcd_.voxel_down_sample(voxel_size=VOXEL_SIZE))

    if visualize:
        o3d.visualization.draw_geometries(down_pcds)  # plot downsampled A and B

    return down_pcds

# extract FPFH features
def extract_FPFH_features(pcds_):
    pcd_fpfh_features = []
    for pcd_ in pcds_:
        feats = extract_fpfh(pcd_, VOXEL_SIZE)
        pcd_fpfh_features.append(feats)

    return pcd_fpfh_features

def pcds2xyz(pcds_):
    pcds_xyz_ = []
    for pcd_ in pcds_:
        pcds_xyz_.append(pcd2xyz(pcd_))  # np array of size 3 by N

    return pcds_xyz_

def visualize_corrs(A_pcd, B_pcd, A_corr, B_corr, num_corrs):
    # visualize the point clouds together with feature correspondences
    points = np.concatenate((A_corr.T, B_corr.T), axis=0)
    lines = []
    for i in range(num_corrs):
        lines.append([i, i + num_corrs])
    colors = [[0, 1, 0] for i in range(len(lines))]  # lines are shown in green
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines),
    )
    line_set.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([A_pcd, B_pcd, line_set])

# establish correspondences by nearest neighbour search in feature space
def get_corrs(pcds_, feats_, visualize=False):
    pcds_xyz = pcds2xyz(pcds_.copy())

    corrs_ = []
    for i in range(1, len(pcds_), 1):
        corrs_A, corrs_B  = find_correspondences(
            feats_[0], feats_[i], mutual_filter=True)
        A_corr = pcds_xyz[0][:, corrs_A]  # np array of size 3 by num_corrs
        B_corr = pcds_xyz[i][:, corrs_B]  # np array of size 3 by num_corrs

        num_corrs = A_corr.shape[1]
        print(f'FPFH generates {num_corrs} putative correspondences.')

        corrs_.append([A_corr, B_corr])

        if visualize:
            visualize_corrs(pcds_[0], pcds_[i], A_corr, B_corr, num_corrs)

    return corrs_

def register_pcds(corrs_):
    transformations_ = []
    for corr_ in corrs_:
        # robust global registration using TEASER++
        teaser_solver = get_teaser_solver(NOISE_BOUND)
        teaser_solver.solve(corr_[0], corr_[1])
        solution = teaser_solver.getSolution()
        R_teaser = solution.rotation
        t_teaser = solution.translation
        T_teaser = Rt2T(R_teaser, t_teaser)
        transformations_.append(T_teaser)

    return transformations_

# local refinement using ICP
def refine_registration(pcds_, t_transformations_):
    transformations_ = []

    for i in range(1, len(pcds_), 1):
        icp_sol = o3d.pipelines.registration.registration_icp(
            pcds_[0], pcds_[i], NOISE_BOUND, t_transformations_[i-1],
            o3d.pipelines.registration.TransformationEstimationPointToPoint(True),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100))
        T_icp = icp_sol.transformation

        transformations_.append(T_icp)

    return transformations_

def get_transformation_matrix(mesh_models, visualize=False):
    pcds = create_pcds(mesh_models)
    downsampled_pcds = downsample_pcds(pcds.copy())
    fpfh_features = extract_FPFH_features(downsampled_pcds.copy())
    correspondences = get_corrs(downsampled_pcds.copy(), fpfh_features)
    teaser_transformation = register_pcds(correspondences)

    np.set_printoptions(suppress=True)
    print("T_teaser transformations: \n{}\n".format(teaser_transformation))

    if visualize:
        visualize_registration(downsampled_pcds.copy(), teaser_transformation)

    icp_transformation = refine_registration(downsampled_pcds.copy(), teaser_transformation)

    print("ICP transformations: \n{}\n".format(icp_transformation))

    if visualize:
        visualize_registration_T(downsampled_pcds.copy(), np.copy(icp_transformation))

    # mesh_models[1].transform(np.linalg.inv(icp_transformation[-1]))

    return icp_transformation[-1]

def vedo2open3d(vd_mesh):
    """
    Return an `open3d.geometry.TriangleMesh` version of
    the current mesh.

    Returns
    ---------
    open3d : open3d.geometry.TriangleMesh
      Current mesh as an open3d object.
    """
    # create from numpy arrays
    o3d_mesh = o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(vd_mesh.points()),
        triangles=o3d.utility.Vector3iVector(vd_mesh.faces()))

    if isinstance(vd_mesh.pointdata["RGB"], np.ndarray):
        o3d_mesh.vertex_colors = o3d.utility.Vector3dVector(vd_mesh.pointdata["RGB"]/255)
    if isinstance(vd_mesh.pointdata["Normals"], np.ndarray):
        o3d_mesh.vertex_normals = o3d.utility.Vector3dVector(vd_mesh.pointdata["Normals"])

    return o3d_mesh

def o3d2pymesh(o3d_mesh):
    m = pymeshlab.Mesh(vertex_matrix=np.array(o3d_mesh.vertices), face_matrix=np.array(o3d_mesh.triangles),
                       v_normals_matrix=np.array(o3d_mesh.vertex_normals),
                       v_color_matrix=np.insert(np.array(o3d_mesh.vertex_colors), 3, 1, axis=1))

    # ms = pymeshlab.MeshSet()
    # ms.add_mesh(m)


    return m

def pymesh2o3d(pymesh_):
    # create from numpy arrays
    o3d_mesh = o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(pymesh_.vertex_matrix()),
        triangles=o3d.utility.Vector3iVector(pymesh_.face_matrix()))

    o3d_mesh.vertex_colors = o3d.utility.Vector3dVector(pymesh_.vertex_color_matrix()[:, 0:-1])
    o3d_mesh.vertex_normals = o3d.utility.Vector3dVector(pymesh_.vertex_normal_matrix())

    return o3d_mesh

def texture_mapping(mesh_, bundle_, scale_ratio, t_matrix, obj_filename, filename):
    # input_mesh = "/home/ttsesm/Development/repair/data/20211117_C0011/colmap_workspace/texture_map/RPf_00047.obj"

    # create a new MeshSet
    ms = pymeshlab.MeshSet()
    pdb.set_trace()
    img_list = bundle_.replace("cameras.out", "list.txt")
    ms.load_project([bundle_, img_list])
    # ms.load_project(
    #     ["/home/ttsesm/Development/repair/data/20211117_C0011/colmap_workspace/texture_map/images/bundle.rd.out",
    #      "/home/ttsesm/Development/repair/data/20211117_C0011/colmap_workspace/texture_map/images/texture_map.out.list.txt"])

    # ms.load_new_mesh(mesh_)
    ms.add_mesh(o3d2pymesh(mesh_))

    # filename = os.path.basename(input_mesh)
    texture_path = bundle_.replace("cameras.out", "model.png")
    ms.parameterization__texturing_from_registered_rasters(texturesize=6000, texturename=texture_path)

    # return back to original polyga space
    ms.matrix_set_copy_transformation(transformmatrix=np.linalg.inv(t_matrix), compose=True)
    ms.transform_scale_normalize(axisx=scale_ratio)

    # ms.set_current_mesh(0)
    # ms.save_current_mesh(filename.replace(".", "_out."))
    ms.save_current_mesh(obj_filename.replace("ply", "obj"))

def find_files(folder, types):

    if len(types) < 2:
        types = ",".join(types)
        return natsort.natsorted(glob.glob(folder + '**/**/*.' + types, flags=glob.BRACE | glob.GLOBSTAR))
    else:
        types = ",".join(types)
        return natsort.natsorted(glob.glob(folder + '**/**/*.{' + types + '}', flags=glob.BRACE | glob.GLOBSTAR))

def main():

    folder = "/media/lucap/big_data/datasets/repair/consolidated_fragments/group_1"
    # folder = "/run/media/ttsesm/external_data/data_for_testing/group_8/"
    
    scanned_meshes = find_files(os.path.join(folder, 'scanned'), ["ply"])
    # scanned_meshes = list(filter(lambda k: 'scanned' in k, scanned_meshes))

    filenames = []

    for scanned_mesh in scanned_meshes:
        dirname, basename = os.path.split(scanned_mesh)
        basename_without_ext, ext = basename.split('.', 1)
        path_without_ext = os.path.join(dirname, basename_without_ext)

        filenames.append(basename_without_ext)

    metashape_meshes = find_files(os.path.join(folder, 'photogrammetry'), ["obj"])

    bundlers = find_files(os.path.join(folder, 'photogrammetry'), ["out", "txt"])

    pattern = re.compile('|'.join(map(str, [sub for sub in filenames])))
    metashape_meshes = list(filter(pattern.search, metashape_meshes))
    bundlers = list(filter(pattern.search, bundlers))

    #bundlers = [bundlers[i:i + 2] for i in range(0, len(bundlers), 2)]
    
    # for j, bundler in enumerate(bundlers):
    #     # # ignore bundlers
    #     # if j < 2:
    #     #     continue
    #
    #     images_list = bundler[1]
    #
    #     for line in fileinput.input(images_list, inplace=1):
    #         print('{0}{1}'.format('./undistorted_images/', line.rstrip('\n')))
    

    for i, scanned_mesh in enumerate(scanned_meshes):
        # ignore meshes
        # if i < 7 or i > 7:
        # if i < 4:
        #     continue

        mesh1_ = vd.Mesh(scanned_mesh)
        mesh2_ = vd.Mesh(metashape_meshes[i])

        scale_ratio_down = mesh2_.diagonal_size() / mesh1_.diagonal_size()
        # scale_ratio_down = 0.12
        scale_ratio_up = mesh1_.diagonal_size() / mesh2_.diagonal_size()
        # scale_ratio_up = 8.4

        mesh1_.scale(s=scale_ratio_down)

        mesh_models = []
        mesh_models.append(vedo2open3d(mesh1_))
        mesh_models.append(vedo2open3d(mesh2_))

        # find transformation matrix between the camera model and the 3D scanner model
        t_matrix = get_transformation_matrix(mesh_models, visualize=False)

        # apply transformation
        mesh_models[0].transform(t_matrix)
        # mesh_models[1].transform(np.linalg.inv(t_matrix))

        texture_mapping(mesh_models[0], bundlers[i], scale_ratio_up, t_matrix, scanned_mesh, filenames[i])

    # mesh1_ = vd.Mesh("/home/ttsesm/Development/repair/data/20211117_C0011/colmap_workspace/texture_map/frag/RPf_00047.ply")
    # mesh2_ = vd.Mesh("/home/ttsesm/Development/repair/data/20211117_C0011/colmap_workspace/texture_map/frag/fragment_.obj")
    #
    # vd.show(mesh1_, mesh2_, axes=1).close()
    #
    # scale_ratio_down = mesh2_.diagonal_size() / mesh1_.diagonal_size()
    # scale_ratio_up = mesh1_.diagonal_size() / mesh2_.diagonal_size()
    #
    # mesh1_.scale(s=scale_ratio_down)
    #
    # vd.show(mesh1_, mesh2_, axes=1).close()
    #
    # mesh_models = []
    #
    # mesh_models.append(vedo2open3d(mesh1_))
    # mesh_models.append(vedo2open3d(mesh2_))
    #
    # # mesh1 = vedo2open3d(mesh1_)
    # # mesh2 = vedo2open3d(mesh2_)
    #
    # # find transformation matrix between the camera model and the 3D scanner model
    # t_matrix = get_transformation_matrix(mesh_models)
    #
    # # apply transformation
    # mesh_models[0].transform(t_matrix)
    # # mesh_models[1].transform(np.linalg.inv(t_matrix))
    #
    # # o3d.visualization.draw_geometries(mesh_models)
    #
    #
    # bundle = ["/home/ttsesm/Development/repair/data/20211117_C0011/colmap_workspace/texture_map/frag/camera_poses.out", "/home/ttsesm/Development/repair/data/20211117_C0011/colmap_workspace/texture_map/frag/list.txt"]
    # texture_mapping(mesh_models[0], bundle, scale_ratio_up, t_matrix)

    return 0


if __name__ == "__main__":
    main()