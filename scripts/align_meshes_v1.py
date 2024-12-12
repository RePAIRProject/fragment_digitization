import os
import sys
import natsort
# from glob import glob
from wcmatch import glob

import copy
from utils.helpers import *

import re, collections, itertools

import vedo as vd

import pymeshlab

VOXEL_SIZE = 1.2 # specifies the amount of points to be used, initial value was 0.05, other values used 0.3 / 0.5 but I was running out of memory in my local machine. Thus, 0.9 seems to be a good enough value
VISUALIZE = False
NOISE_BOUND = VOXEL_SIZE


def group(d, p=[]):
   c = collections.defaultdict(list)
   for a, *b in d:
      c[a].append(b)
   for a, b in c.items():
      if any(len(i) == 1 for i in b):
         v = [x for y in b for x in ([p+[a]+y] if len(y)==1 else group([y],p+[a]))]
         yield [j for k in v for j in ([k] if all(isinstance(i, str) for i in k) else k)]
      else:
         yield from group(b, p+[a])

def create_pcds(mesh_models):
    pcds_ = []
    for mesh_model in mesh_models:
        pcd = o3d.geometry.PointCloud(mesh_model.vertices)
        pcd.normals = mesh_model.vertex_normals
        pcd.paint_uniform_color([np.round(np.random.uniform(low=0, high=1), 1), np.round(np.random.uniform(low=0, high=1), 1), np.round(np.random.uniform(low=0, high=1), 1)]) # give each point cloud a random color
        pcds_.append(pcd)

    if VISUALIZE:
        o3d.visualization.draw_geometries(pcds_)  # plot A and B

    return pcds_

# voxel downsample both clouds
def downsample_pcds(pcds_):
    down_pcds = []
    for pcd_ in pcds_:
        down_pcds.append(pcd_.voxel_down_sample(voxel_size=VOXEL_SIZE))

    if VISUALIZE:
        o3d.visualization.draw_geometries(down_pcds)  # plot downsampled A and B

    return down_pcds

def pcds2xyz(pcds_):
    pcds_xyz_ = []
    for pcd_ in pcds_:
        pcds_xyz_.append(pcd2xyz(pcd_))  # np array of size 3 by N

    return pcds_xyz_

# extract FPFH features
def extract_FPFH_features(pcds_):
    pcd_fpfh_features = []
    for pcd_ in pcds_:
        feats = extract_fpfh(pcd_, VOXEL_SIZE)
        pcd_fpfh_features.append(feats)

    return pcd_fpfh_features

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
def get_corrs(pcds_, feats_):
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

        if VISUALIZE:
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
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100))
        T_icp = icp_sol.transformation

        transformations_.append(T_icp)

    return transformations_

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

def save_meshes(filename_, meshes_, transformations_):
    for i in range(1, len(meshes_), 1):
        meshes_[0] += copy.deepcopy(meshes_[i]).transform(np.linalg.inv(transformations_[i - 1]))

    meshes_[0].merge_close_vertices(0.000001)
    # o3d.io.write_triangle_mesh(filename_, remove_boundary_artifacts(fill_holes(meshes_[0])))
    o3d.io.write_triangle_mesh(filename_, post_processing(meshes_[0]))

def NestedDictValues(d):
  for v in d.values():
    if isinstance(v, dict):
      yield from NestedDictValues(v)
    else:
      yield v

def nested_dict_pairs_iterator(dict_obj):
    for key, value in dict_obj.items():
        # Check if value is of dict type
        if isinstance(value, dict):
            # If value is dict then iterate over all its values
            for pair in nested_dict_pairs_iterator(value):
                yield (key, *pair)
        else:
            # If value is not dict type then yield the value
            yield (key, value)

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

    o3d_mesh.vertex_colors = o3d.utility.Vector3dVector(vd_mesh.pointdata["RGB"]/255)
    o3d_mesh.vertex_normals = o3d.utility.Vector3dVector(vd_mesh.pointdata["Normals"])

    return o3d_mesh

def vedo2pymesh(vd_mesh):

    m = pymeshlab.Mesh(vertex_matrix=vd_mesh.points(), face_matrix=vd_mesh.faces(), v_normals_matrix=vd_mesh.pointdata["Normals"], v_color_matrix=np.insert(vd_mesh.pointdata["RGB"]/255, 3, 1, axis=1))

    ms = pymeshlab.MeshSet()
    ms.add_mesh(m)

    return ms

def pymesh2vedo(mlab_mesh):
    color = mlab_mesh.vertex_color_matrix()[:, 0:-1]
    reco_mesh = vd.Mesh(mlab_mesh)
    reco_mesh.pointdata["RGB"] = (color * 255).astype(np.uint8)
    reco_mesh.pointdata["Normals"] = mlab_mesh.vertex_normal_matrix().astype(np.float32)
    reco_mesh.pointdata.select("RGB")

    return reco_mesh

def o3d2pymesh(o3d_mesh):
    m = pymeshlab.Mesh(vertex_matrix=np.array(o3d_mesh.vertices), face_matrix=np.array(o3d_mesh.triangles),
                       v_normals_matrix=np.array(o3d_mesh.vertex_normals),
                       v_color_matrix=np.insert(np.array(o3d_mesh.vertex_colors), 3, 1, axis=1))

    ms = pymeshlab.MeshSet()
    ms.add_mesh(m)


    return ms

def pymesh2o3d(pymesh_):
    # create from numpy arrays
    o3d_mesh = o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(pymesh_.vertex_matrix()),
        triangles=o3d.utility.Vector3iVector(pymesh_.face_matrix()))

    o3d_mesh.vertex_colors = o3d.utility.Vector3dVector(pymesh_.vertex_color_matrix()[:, 0:-1])
    o3d_mesh.vertex_normals = o3d.utility.Vector3dVector(pymesh_.vertex_normal_matrix())

    return o3d_mesh

def trim_boundary(mesh_):
    # vd.settings.useDepthPeeling = True

    m1 = vd.Mesh(mesh_)
    cm = m1.centerOfMass()

    elli = vd.pcaEllipsoid(m1, pvalue=0.5)

    ax1 = vd.versor(elli.axis1 - cm)
    ax2 = vd.versor(elli.axis2 - cm)
    ax3 = vd.versor(elli.axis3 - cm)

    T = np.array([ax1, ax2, ax3])

    m1 = m1.applyTransform(T, reset=True)
    # m2 = m1.clone()

    bnd = m1.boundaries(nonManifoldEdges=False, featureAngle=180)

    modl = bnd.implicitModeller(distance=1, res=(100, 100, 20)).extractLargestRegion()

    m1.cutWithMesh(modl)

    return m1

def fill_holes(mesh_):
    m = o3d2pymesh(mesh_)
    m.close_holes(maxholesize=30, newfaceselected=False)

    mlab_mesh = m.current_mesh()

    reco_mesh = pymesh2o3d(mlab_mesh)

    return reco_mesh

def remove_boundary_artifacts(mesh_):
    m = o3d2pymesh(mesh_)
    m.surface_reconstruction_screened_poisson(depth=8, pointweight=1, preclean=True)

    mlab_mesh = m.current_mesh()

    reco_mesh = pymesh2o3d(mlab_mesh)

    return reco_mesh

def pre_processing(mesh_):

    step1 = trim_boundary(mesh_)

    return vedo2open3d(step1)

def post_processing(mesh_):

    # return remove_boundary_artifacts(fill_holes(mesh_))
    return remove_boundary_artifacts(mesh_)
    # return mesh_

def main():
    # folder = '/run/media/ttsesm/external_data/repair_dataset/dataset@server/group_19/raw/3D/'
    folder = "/run/media/ttsesm/51A6ECAF2B33CC01/groups/group_34/"
    # folder = '../data/test_box/box_7/frag_12/'
    mesh_files = natsort.natsorted(glob.glob(folder + '**/*.{ply,obj}', flags=glob.BRACE | glob.GLOBSTAR))

    d = collections.defaultdict(dict)
    for filepath in mesh_files:
        keys = filepath.split("/")
        folder_ = keys[-4]
        file_ = re.match("(.*?)"+re.findall(r"\d+", keys[-1].split(".")[0])[0], keys[-1]).group()
        if folder_ in d:
            if file_ in d[folder_]:
                d[folder_][file_].append(filepath)
            else:
                d[folder_][file_] = [filepath]
        else:
            d[folder_][file_] = [filepath]


    for pair in nested_dict_pairs_iterator(d):

        # ignore some folders and start from specific folder
        # if int(re.findall(r'\d+', pair[0])[0]) < 34:
        #     continue

        mesh_models = []
        if len(pair[-1]) < 2:
            continue
        filename = None
        dirname = None
        for mesh_file in pair[-1]:
            # print(mesh_file)
            mesh_models.append(o3d.io.read_triangle_mesh(mesh_file, enable_post_processing=True, print_progress=True))
            # mesh_models.append(pre_processing(mesh_file))
            if filename == None:
                filename = os.path.basename(mesh_file).split('.')[0]
            else:
                filename = filename + '-' + os.path.basename(mesh_file).split('.')[0]

            if dirname == None:
                dirname = os.path.dirname(mesh_file)

        pcds = create_pcds(mesh_models)
        downsampled_pcds = downsample_pcds(pcds.copy())
        fpfh_features = extract_FPFH_features(downsampled_pcds.copy())
        correspondences = get_corrs(downsampled_pcds.copy(), fpfh_features)
        teaser_transformations = register_pcds(correspondences)

        np.set_printoptions(suppress=True)
        print("T_teaser transformations: \n{}\n".format(teaser_transformations))

        if VISUALIZE:
            visualize_registration(downsampled_pcds.copy(), teaser_transformations)

        icp_transformations = refine_registration(downsampled_pcds.copy(), teaser_transformations)

        print("ICP transformations: \n{}\n".format(icp_transformations))

        if VISUALIZE:
            visualize_registration_T(downsampled_pcds.copy(), np.copy(icp_transformations))

        with open(dirname+"/transformations.txt", "a") as f:
            f.write(filename+"\n")
            for matrix in np.asarray(icp_transformations):
                np.savetxt(f, matrix, fmt='%1.10f', newline='\n')
                f.write("\n")
            f.write("\n")
            f.close()

        save_meshes(os.path.join(dirname, pair[-2]+".ply"), mesh_models, icp_transformations)

        mesh_models.clear()


    return 0

if __name__ == "__main__":
    main()