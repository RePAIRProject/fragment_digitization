
import os
import sys
import natsort
# from glob import glob
from wcmatch import glob

import numbers
from math import sqrt

import open3d as o3d
import teaserpp_python
import numpy as np
import copy
from utils.helpers import *

import re, collections, itertools

from vedo import *
import numpy as np
from scipy.interpolate import griddata
import pyshtools

import trimesh
import vedo as vd
from vedo.applications import Animation
from vedo import settings
settings.allowInteraction=True

import pymeshlab

VOXEL_SIZE = 0.3 # specifies the amount of points to be used, initial value was 0.05, other values used 0.3 / 0.5 but I was running out of memory in my local machine. Thus, 0.9 seems to be a good enough value
VISUALIZE = False
NOISE_BOUND = VOXEL_SIZE

def find_viz_shape(n):
    wIndex = np.ceil(np.sqrt(n)).astype(int)
    hIndex = np.ceil(n/wIndex).astype(int)

    return [hIndex, wIndex]

def transform_mesh(m):
    cm = m.centerOfMass()
    m.shift(-cm)
    elli = pcaEllipsoid(m, pvalue=0.5)

    ax1 = versor(elli.axis1)
    ax2 = versor(elli.axis2)
    ax3 = versor(elli.axis3)

    T = np.array([ax1, ax2, ax3])  # the transposed matrix is already the inverse
    # print(T)
    # print(T@ax1)

    return m.applyTransform(T, reset=True)

def update_pos(points, n_points):
    p1, p2 = points
    diff = p2 - p1
    t = np.linspace(0, 1, n_points+20)[1:-19]
    return (p1[np.newaxis, :] + t[:, np.newaxis] * diff[np.newaxis, :]).squeeze()

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

    # o3d_mesh.vertex_colors = o3d.utility.Vector3dVector(vd_mesh.pointdata["RGB"]/255)
    # o3d_mesh.vertex_normals = o3d.utility.Vector3dVector(vd_mesh.pointdata["Normals"])
    # o3d_mesh.vertex_normals = o3d.utility.Vector3dVector(vd_mesh.normals())

    return o3d_mesh

def vedo2pymesh(vd_mesh):

    m = pymeshlab.Mesh(vertex_matrix=vd_mesh.points(), face_matrix=vd_mesh.faces(), v_normals_matrix=vd_mesh.pointdata["Normals"], v_color_matrix=np.insert(vd_mesh.pointdata["RGB"]/255, 3, 1, axis=1))

    ms = pymeshlab.MeshSet()
    ms.add_mesh(m)

    # vd.show(ms, axes=True, interactive=True).close()

    return ms

def pymesh2vedo(mlab_mesh):
    color = mlab_mesh.vertex_color_matrix()[:, 0:-1]
    reco_mesh = vd.Mesh(mlab_mesh)
    reco_mesh.pointdata["RGB"] = (color * 255).astype(np.uint8)
    reco_mesh.pointdata["Normals"] = mlab_mesh.vertex_normal_matrix().astype(np.float32)
    reco_mesh.pointdata.select("RGB")

    # vd.show(reco_mesh, axes=True, interactive=True).close()
    return reco_mesh

# voxel downsample both clouds
def downsample_pcds(pcds_):
    down_pcds = []
    for pcd_ in pcds_:
        down_pcds.append(pcd_.voxel_down_sample(voxel_size=VOXEL_SIZE))

    if VISUALIZE:
        o3d.visualization.draw_geometries(down_pcds)  # plot downsampled A and B

    return down_pcds


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

def find_manual_correspondences(meshes):

    m1_pts = meshes[0].points()
    m2_pts = meshes[1].points()

    correspondences = []

    for i, p in enumerate(m1_pts):
        iclos = meshes[1].closestPoint(p, returnPointId=True)
        correspondences.append([i, iclos])

    dist_thres = 0.05
    correspondences = np.asarray(correspondences).reshape(-1, 2)
    correspondences = correspondences[np.where(vd.mag(m1_pts[correspondences[:,0]] - m2_pts[correspondences[:,1]]) < dist_thres), :]

    return correspondences.squeeze()


def main():
    folder = '/run/media/ttsesm/external_data/repair_dataset/tuwien/cake/'
    mesh_files = natsort.natsorted(glob.glob(folder + '**/*.xyz', flags=glob.BRACE | glob.GLOBSTAR))
    mesh_files = list(filter(lambda k: 'final' not in k, mesh_files))

    d = collections.defaultdict(dict)
    for filepath in mesh_files:
        keys = filepath.split("/")
        folder_ = keys[-3]
        file_ = re.match("(.*?)" + re.findall(r"\d+", keys[-1].split(".")[0])[0], keys[-1]).group()
        if folder_ in d:
            if file_ in d[folder_]:
                d[folder_][file_].append(filepath)
            else:
                d[folder_][file_] = [filepath]
        else:
            d[folder_][file_] = [filepath]

    mesh_models_original = []
    mesh_models = []
    mesh_models_o3d = []
    # mesh_boundaries = []
    # m = vd.Mesh("/home/ttsesm/Data/repair_dataset/presious/assembled_samples/Tombstone/Reassembled_Tombstone/Reassembled_Tombstone.obj").rotate(45)


    sx = 0
    sy = 0
    for i, mesh_file in enumerate(mesh_files):
        if i > 1:
            continue
        # color = [np.random.randint(0,255), np.random.randint(0,255), np.random.randint(0,255)]
        m = vd.Mesh(mesh_file).color(i)
        # m.closestPoint()
        mesh_models_original.append(m)
        # mesh = m
        mesh = transform_mesh(m.clone())
        # b = mesh.boundaries(featureAngle=250).extractLargestRegion().c('red')
        sx += mesh.xbounds()[1] -mesh.xbounds()[0]
        sy += mesh.ybounds()[1] - mesh.ybounds()[0]
        mesh_models.append(mesh)
        mesh_models_o3d.append(vedo2open3d(mesh))
        # mesh_boundaries.append(b)

    corrs = find_manual_correspondences(mesh_models_original)

    # pcds = create_pcds(mesh_models_o3d)
    # downsampled_pcds = downsample_pcds(pcds.copy())
    # fpfh_features = extract_FPFH_features(downsampled_pcds.copy())
    # correspondences = get_corrs(downsampled_pcds.copy(), fpfh_features, True)

    gridRes = find_viz_shape(len(mesh_models))
    grid = Grid(sx=sx, sy=sy, resx=gridRes[0], resy=gridRes[1])
    gpts = Points(grid.cellCenters())
    # vd.show(grid, axes=1, interactive=1).close()


    # pts1 = mesh_models_original[0].points()[corrs[:,0]]
    # pts2 = mesh_models_original[1].points()[corrs[:, 1]]
    pts1 = Points(mesh_models_original[0].points()[list(corrs[:,0])], r=5, c='blue')
    pts2 = Points(mesh_models_original[1].points()[list(corrs[:, 1])], r=5, c='red')

    pts1_ = Points(mesh_models[0].points()[list(corrs[:, 0])], r=5, c='blue')
    pts2_ = Points(mesh_models[1].points()[list(corrs[:, 1])], r=5, c='red')

    # vd.show(mesh_models_original, pts1, pts2, axes=1, interactive=1).close()
    vd.show(mesh_models_original, pts1, pts2, "Before", at=0, N=2, axes=1, sharecam=False)
    vd.show(mesh_models, pts1_, pts2_, "After", at=1, interactive=True, sharecam=False).close()

    # # Setup the scene
    # video = Video("anim_.gif", backend='ffmpeg')  # backend='opencv/ffmpeg'
    # video.options = "-b:v 8000k -filter_complex \"[0:v] split [a][b];[a] palettegen=stats_mode=single [p];[b][p] paletteuse=new=1\""
    # plt = vd.Plotter(axes=1, interactive=0)
    # plt.camera.SetPosition([4260.325, -2417.882, 2296.456])
    # plt.camera.SetFocalPoint([26.215, 31.243, 46.972])
    # plt.camera.SetViewUp([-0.389, 0.161, 0.907])
    # plt.camera.SetDistance(5383.872)
    # plt.camera.SetClippingRange([2782.322, 8671.674])
    # plt.show(mesh_models, grid, axes=dict(ztitle=""))
    #
    # for t in np.arange(0, 1, 0.005):
    #     for i, mesh_model in enumerate(mesh_models):
    #         mesh_model.pos(update_pos([mesh_model.pos(), grid.cellCenters()[i]], 1))
    #         plt.show(mesh_models, grid)
    #     video.addFrame()
    #     if plt.escaped:
    #         break  # if ESC button is hit during the loop
    #
    # video.close()
    # interactive().close()
    # #
    # #
    # # vp = Plotter(shape=gridRes, axes=0, interactive=0, sharecam=False)
    # #
    # # # for t in np.arange(0, 1, 0.005):
    # # for i, mesh_model in enumerate(mesh_models):
    # #     vp.show(mesh_model.lighting(style='default'), at=i)
    # #     cam = vp.renderer.GetActiveCamera()
    # #     # cam.Azimuth(2)
    # #
    # # vp.show(interactive=1)
    # # vp.clear()
    # # # vp.show(interactive=0)
    # # #
    # # # for i, mesh_model in enumerate(mesh_models):
    # # #     vp.show(mesh_model.lighting(style='default'), mesh_boundaries[i], at=i)
    # # #     cam = vp.renderer.GetActiveCamera()
    # # #     # cam.Azimuth(2)
    # # #
    # # # vp.show(interactive=1)


    return 0

if __name__ == "__main__":
    main()