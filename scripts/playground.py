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
    o3d_mesh.vertex_normals = o3d.utility.Vector3dVector(vd_mesh.pointdata["Normals"])
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




def main():

    # # mesh = o3d.io.read_triangle_mesh("/home/ttsesm/Downloads/hydria_apothecary_vase/scene.gltf", enable_post_processing=True, print_progress=True)
    # #
    # # o3d.visualization.draw_geometries([mesh])
    #
    # m1 = vd.Mesh("../data/box_8/frag_3_final.ply").c("green").alpha(0.8)
    # m2 = vd.Mesh("../data/box_8/frag_3__final.ply").c("red").alpha(0.8)
    #
    # # m1 = m1.decimate(0.1)
    # # m2 = m2.decimate(0.1)
    #
    # T = np.array([[-0.9969830354, -0.0742881466, -0.0224966296, 202.4966262662],
    #                [0.0534353633, -0.4466713800, -0.8931009687, 363.3441404664],
    #                [0.0562982151, -0.8916086303, 0.4492934024, 228.7605892738],
    #                [0.0000000000, 0.0000000000, 0.0000000000, 1.0000000000]])
    #
    # # m3 = m2.clone().applyTransform(T, reset=True)
    # #
    # # contour = m1.intersectWith(m3).lineWidth(4).c('black')
    # #
    # # vd.show(m1, m3, contour, __doc__, axes=7).close()
    #
    # m3 = m1.clone()
    # m4 = m2.clone().applyTransform(T, reset=True)
    # #
    # # a = 60
    # # m3.RotateX(a)
    # # m4.RotateX(a)
    # #
    # # m3.cutWithPlane((0, 0, 210), '-z')
    # # m4.cutWithPlane((0, 0, 203), 'z')
    # #
    # # b3 = m3.boundaries().extractLargestRegion().c('green6')
    # # b4 = m4.boundaries().extractLargestRegion().c('purple6')
    # #
    # # pts3 = m3.points()
    # # pts4 = m4.points()
    # # pcl3 = vd.Points(pts3)
    # # pcl4 = vd.Points(pts4)
    # #
    # # lines = []
    # # pb = vd.ProgressBar(0, b3.NPoints())
    # # for p in b4.points():
    # #     pb.print()
    # #     idx3 = pcl3.closestPoint(p, returnPointId=True)
    # #     idx4 = pcl4.closestPoint(p, returnPointId=True)
    # #     lines.append([pts3[idx3], pts4[idx4]])
    # #     pts3[idx3] = pts4[idx4]
    # #
    # # m3.points(pts3)  # update mesh vertices to seal the gap
    # # m = vd.merge(m3, m4).clean().c('indigo5')
    # # m.write("test.ply")
    # #
    # # lns = vd.Lines(lines, lw=2)
    # # vd.show(m3, m4, b3, b4, lns, axes=1)
    # # vd.show(m, b3, b4, axes=1).close()
    # #
    # # # vd.show(m1, m2, "Before", at=0, N=2, axes=1)
    # # # vd.show(m1, m3, "After", at=1, interactive=True).close()






    # mesh = vd.Mesh("../data/box_8/frag_3.ply")#.c("gray")
    #
    # m = vedo2pymesh(mesh)
    # m.surface_reconstruction_screened_poisson(depth=8, pointweight=1, preclean=True)
    # # m.close_holes(maxholesize=30, newfaceselected=False)
    #
    # mlab_mesh = m.current_mesh()
    #
    # reco_mesh = pymesh2vedo(mlab_mesh)
    #
    # vd.show(mesh.lighting(style='default'), "Before", at=0, N=2, axes=1)
    # vd.show(reco_mesh.lighting(style='ambient'), "After", at=1, interactive=True).close()
    #
    # # # Set the camera position
    # # plt = Animation()
    # # # plt.camera.SetPosition([512921.567, 6407793.637, 8217.335])
    # # # plt.camera.SetFocalPoint([505099.133, 6415752.321, -907.462])
    # # # plt.camera.SetViewUp([-0.494, 0.4, 0.772])
    # # # plt.camera.SetDistance(14415.028)
    # # # plt.camera.SetClippingRange([7367.387, 23203.319])
    # #
    # # # Trying to play with the Animation class
    # # plt.showProgressBar = True
    # # plt.timeResolution = 0.025  # secs
    # #
    # # # It would be okay if I only plot one mesh object
    # # plt.fadeIn(mesh, t=0, duration=0.5)
    # # plt.changeLighting(style='plastic', t=0)
    # # plt.rotate(mesh, axis="y", angle=180, t=1, duration=2)
    # # # plt.rotate([mesh, reco_mesh], axis="y", angle=180, t=1, duration=2)
    # #
    # # # # If I try to plot multiple objects, then things quickly go wrong: the mesh objects become a small point
    # # # # in the middle of the mesh and then disappear
    # # # plt.fadeIn([cond_msh, msh, bm1_msh, bm2_msh], t=0, duration=0.5)
    # # # plt.rotate([cond_msh, msh, bm1_msh, bm2_msh], axis="z", angle=180, t=1, duration=3)
    # # # plt.totalDuration = 4  # can shrink/expand total duration
    # #
    # # plt.play()







    # ###########################################################################
    # lmax = 8  # maximum degree of the spherical harm. expansion
    # N = 50  # number of grid intervals on the unit sphere
    # rmax = 500  # line length
    # x0 = [250, 250, 250]  # set SPH sphere at this position
    # ###########################################################################
    #
    # x0 = np.array(x0)
    # surface = Box(pos=x0 + [10, 20, 30], size=(300, 150, 100)).color('grey').alpha(0.2)
    #
    # ############################################################
    # # cast rays from the sphere center and find intersections
    # agrid, pts = [], []
    # for th in np.linspace(0, np.pi, N, endpoint=True):
    #     longs = []
    #     for ph in np.linspace(0, 2 * np.pi, N, endpoint=False):
    #         p = spher2cart(rmax, th, ph)
    #         intersections = surface.intersectWithLine(x0, x0 + p)
    #         if len(intersections):
    #             value = mag(intersections[0] - x0)
    #             longs.append(value)
    #             pts.append(intersections[0])
    #         else:
    #             printc('No hit for theta, phi =', th, ph, c='r')
    #             longs.append(rmax)
    #             pts.append(p)
    #     agrid.append(longs)
    # agrid = np.array(agrid)
    #
    # hits = Points(pts).cmap('jet', agrid.ravel()).addScalarBar3D(title='scalar distance to x_0')
    # show([surface, hits, Point(x0), __doc__], at=0, N=2, axes=1)
    #
    # #############################################################
    # grid = pyshtools.SHGrid.from_array(agrid)
    # clm = grid.expand()
    # grid_reco = clm.expand(lmax=lmax).to_array()  # cut "high frequency" components
    #
    # #############################################################
    # # interpolate to a finer grid
    # ll = []
    # for i, long in enumerate(np.linspace(0, 360, num=grid_reco.shape[1], endpoint=False)):
    #     for j, lat in enumerate(np.linspace(90, -90, num=grid_reco.shape[0], endpoint=True)):
    #         th = np.deg2rad(90 - lat)
    #         ph = np.deg2rad(long)
    #         p = spher2cart(grid_reco[j][i], th, ph)
    #         ll.append((lat, long))
    #
    # radii = grid_reco.T.ravel()
    # n = 200j
    # lnmin, lnmax = np.array(ll).min(axis=0), np.array(ll).max(axis=0)
    # grid = np.mgrid[lnmax[0]:lnmin[0]:n, lnmin[1]:lnmax[1]:n]
    # grid_x, grid_y = grid
    # grid_reco_finer = griddata(ll, radii, (grid_x, grid_y), method='cubic')
    #
    # pts2 = []
    # for i, long in enumerate(np.linspace(0, 360, num=grid_reco_finer.shape[1], endpoint=False)):
    #     for j, lat in enumerate(np.linspace(90, -90, num=grid_reco_finer.shape[0], endpoint=True)):
    #         th = np.deg2rad(90 - lat)
    #         ph = np.deg2rad(long)
    #         p = spher2cart(grid_reco_finer[j][i], th, ph)
    #         pts2.append(p + x0)
    #
    # show(Points(pts2, c="r", alpha=0.5), surface,
    #      'Spherical harmonics\nexpansion of order ' + str(lmax),
    #      at=1, interactive=True)






    # ##########################################################
    # N = 100  # number of sample points on the unit sphere
    # lmax = 15  # maximum degree of the sph. harm. expansion
    # rbias = 0.5  # subtract a constant average value
    # x0 = [0, 0, 0]  # set object at this position
    #
    # ##########################################################
    #
    # def makeGrid(shape, N):
    #     rmax = 2.0  # line length
    #     agrid, pts = [], []
    #     for th in np.linspace(0, np.pi, N, endpoint=True):
    #         lats = []
    #         for ph in np.linspace(0, 2 * np.pi, N, endpoint=True):
    #             p = np.array([sin(th) * cos(ph), sin(th) * sin(ph), cos(th)]) * rmax
    #             intersections = shape.intersectWithLine([0, 0, 0], p)
    #             if len(intersections):
    #                 value = mag(intersections[0])
    #                 lats.append(value - rbias)
    #                 pts.append(intersections[0])
    #             else:
    #                 lats.append(rmax - rbias)
    #                 pts.append(p)
    #         agrid.append(lats)
    #     agrid = np.array(agrid)
    #     actor = Points(pts, c="k", alpha=0.4, r=1)
    #     return agrid, actor
    #
    # def morph(clm1, clm2, t, lmax):
    #     """Interpolate linearly the two sets of sph harm. coeeficients."""
    #     clm = (1 - t) * clm1 + t * clm2
    #     grid_reco = clm.expand(lmax=lmax)  # cut "high frequency" components
    #     agrid_reco = grid_reco.to_array()
    #     pts = []
    #     for i, longs in enumerate(agrid_reco):
    #         ilat = grid_reco.lats()[i]
    #         for j, value in enumerate(longs):
    #             ilong = grid_reco.lons()[j]
    #             th = np.deg2rad(90 - ilat)
    #             ph = np.deg2rad(ilong)
    #             r = value + rbias
    #             p = np.array([sin(th) * cos(ph), sin(th) * sin(ph), cos(th)]) * r
    #             pts.append(p)
    #     return pts
    #
    # vp = Plotter(shape=[2, 2], axes=3, interactive=0)
    #
    # shape1 = Sphere(alpha=0.2)
    # shape2 = vp.load(dataurl + "icosahedron.vtk").normalize().lineWidth(1)
    #
    # agrid1, actorpts1 = makeGrid(shape1, N)
    #
    # vp.show(shape1, actorpts1, at=0)
    #
    # agrid2, actorpts2 = makeGrid(shape2, N)
    # vp.show(shape2, actorpts2, at=1)
    #
    # vp.camera.Zoom(1.2)
    # vp.interactive = False
    #
    # clm1 = pyshtools.SHGrid.from_array(agrid1).expand()
    # clm2 = pyshtools.SHGrid.from_array(agrid2).expand()
    # clm1.plot_spectrum2d()  # plot the value of the sph harm. coefficients
    # clm2.plot_spectrum2d()
    #
    # for t in np.arange(0, 1, 0.005):
    #     act21 = Points(morph(clm2, clm1, t, lmax), c="r", r=4)
    #     act12 = Points(morph(clm1, clm2, t, lmax), c="g", r=4)
    #
    #     vp.show(act21, at=2, resetcam=0)
    #     vp.show(act12, at=3)
    #     vp.camera.Azimuth(2)
    #
    # vp.show(interactive=1)






    # # folder = '../data/tombstone/'
    # folder = '../data/box_8/'
    # mesh_files = natsort.natsorted(glob.glob(folder + '**/*.{ply,obj}', flags=glob.BRACE | glob.GLOBSTAR))
    # mesh_files = list(filter(lambda k: 'final' in k, mesh_files))
    #
    # d = collections.defaultdict(dict)
    # for filepath in mesh_files:
    #     keys = filepath.split("/")
    #     folder_ = keys[-2]
    #     file_ = re.match("(.*?)" + re.findall(r"\d+", keys[-1].split(".")[0])[0], keys[-1]).group()
    #     if folder_ in d:
    #         if file_ in d[folder_]:
    #             d[folder_][file_].append(filepath)
    #         else:
    #             d[folder_][file_] = [filepath]
    #     else:
    #         d[folder_][file_] = [filepath]
    #
    # mesh_models = []
    # mesh_boundaries = []
    # m = vd.Mesh("/home/ttsesm/Data/repair_dataset/presious/assembled_samples/Tombstone/Reassembled_Tombstone/Reassembled_Tombstone.obj").rotate(45)
    #
    # for mesh_file in mesh_files:
    #     mesh = vd.Mesh(mesh_file)
    #     # b = mesh.boundaries().c('red')
    #     mesh_models.append(mesh)
    #     # mesh_boundaries.append(b)
    #
    # vp = Plotter(shape=find_viz_shape(len(mesh_files)), axes=0, interactive=0, sharecam=False)
    # video = Video("box_8.gif", backend='ffmpeg')  # backend='opencv/ffmpeg'
    # video.options = "-b:v 8000k -filter_complex \"[0:v] split [a][b];[a] palettegen=stats_mode=single [p];[b][p] paletteuse=new=1\""
    # # vp = Plotter(axes=0, interactive=0)
    #
    # # vp.camera.Zoom(1.2)
    # # vp.camera.SetClippingRange([0.01, 1])
    # # vp.interactive = False
    #
    # for t in np.arange(0, 1, 0.005):
    #     for i, mesh_model in enumerate(mesh_models):
    #         vp.show(mesh_model.lighting(style='ambient'), at=i)
    #         cam = vp.renderer.GetActiveCamera()
    #         cam.Azimuth(2)
    #         # vp.show(mesh_models[0].c('gray').lighting(style='ambient'), at=0)
    #         # cam = vp.renderer.GetActiveCamera()
    #         # cam.Azimuth(2)
    #         # vp.show(mesh_models[1].c('gray').lighting(style='ambient'), at=1)
    #         # cam = vp.renderer.GetActiveCamera()
    #         # cam.Azimuth(2)
    #         # vp.show(mesh_models[2].c('gray').lighting(style='ambient'), at=2)
    #         # cam = vp.renderer.GetActiveCamera()
    #         # cam.Azimuth(2)
    #         # # vp.show(mesh_models[3].c('gray'), mesh_boundaries[3], at=3)
    #         # # vp.show(mesh_models[4].c('gray'), mesh_boundaries[4], at=4)
    #         # # vp.show(m.c('gray'), at=5)
    #         # # vp.camera.Azimuth(2)
    #     video.addFrame()
    #
    # video.close()
    # # vp.show(interactive=1)




    #
    # # folder = '../data/tombstone/'
    # # folder = '/home/ttsesm/Data/repair_dataset/presious/assembled_samples/Tombstone/Reassembled_Tombstone/'
    # folder = '../data/box_8/'
    # mesh_files = natsort.natsorted(glob.glob(folder + '**/*.obj', flags=glob.BRACE | glob.GLOBSTAR))
    # mesh_files = list(filter(lambda k: 'final' not in k, mesh_files))
    #
    # d = collections.defaultdict(dict)
    # for filepath in mesh_files:
    #     keys = filepath.split("/")
    #     folder_ = keys[-2]
    #     file_ = re.match("(.*?)" + re.findall(r"\d+", keys[-1].split(".")[0])[0], keys[-1]).group()
    #     if folder_ in d:
    #         if file_ in d[folder_]:
    #             d[folder_][file_].append(filepath)
    #         else:
    #             d[folder_][file_] = [filepath]
    #     else:
    #         d[folder_][file_] = [filepath]
    #
    # mesh_models = []
    # mesh_boundaries = []
    # # m = vd.Mesh("/home/ttsesm/Data/repair_dataset/presious/assembled_samples/Tombstone/Reassembled_Tombstone/Reassembled_Tombstone.obj").rotate(45)
    #
    #
    # sx = 0
    # sy = 0
    # for i, mesh_file in enumerate(mesh_files):
    #     # color = [np.random.randint(0,255), np.random.randint(0,255), np.random.randint(0,255)]
    #     m = vd.Mesh(mesh_file).color(i)
    #     mesh = m
    #     # mesh = transform_mesh(m.clone())
    #     b = mesh.boundaries(featureAngle=250).extractLargestRegion().c('red')
    #     sx += mesh.xbounds()[1] -mesh.xbounds()[0]
    #     sy += mesh.ybounds()[1] - mesh.ybounds()[0]
    #     mesh_models.append(mesh)
    #     mesh_boundaries.append(b)
    #
    # gridRes = find_viz_shape(len(mesh_models))
    # grid = Grid(sx=sx, sy=sy, resx=gridRes[0], resy=gridRes[1])
    # gpts = Points(grid.cellCenters())
    # # vd.show(grid, axes=1, interactive=1).close()
    #
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
    #
    # # anim = Animation()  # a vedo.Plotter object
    # # anim.timeResolution = 0.01  # secs
    # # anim.totalDuration = 5  # can now shrink/expand total duration
    # # axes = Axes(mesh_models[0])  # build axes manually
    # #
    # # anim.fadeIn(mesh_models, t=0, duration=0.2)
    # # anim.fadeIn(grid, t=0, duration=0.2)
    # # # anim.fadeIn(axes, t=0, duration=0.2) # will not work right now!
    # # anim.fadeIn(gpts, t=0, duration=0.2)
    # #
    # # for i, mm in enumerate(mesh_models):
    # #     anim.move(mm, grid.cellCenters()[i])#, style="quadratic")
    # #
    # # anim.play()
    #
    # vp = Plotter(shape=gridRes, axes=0, interactive=0, sharecam=False)
    #
    # # for t in np.arange(0, 1, 0.005):
    # for i, mesh_model in enumerate(mesh_models):
    #     vp.show(mesh_model.lighting(style='default'), at=i)
    #     cam = vp.renderer.GetActiveCamera()
    #     # cam.Azimuth(2)
    #
    # vp.show(interactive=1)
    # vp.clear()
    # vp.show(interactive=0)
    #
    # for i, mesh_model in enumerate(mesh_models):
    #     vp.show(mesh_model.lighting(style='default'), mesh_boundaries[i], at=i)
    #     cam = vp.renderer.GetActiveCamera()
    #     # cam.Azimuth(2)
    #
    # vp.show(interactive=1)






    # # folder = '../data/tombstone/'
    # folder = '../data/box_8/'
    # mesh_files = natsort.natsorted(glob.glob(folder + '**/*.{ply,obj}', flags=glob.BRACE | glob.GLOBSTAR))
    # mesh_files = list(filter(lambda k: 'final' not in k, mesh_files))
    #
    # d = collections.defaultdict(dict)
    # for filepath in mesh_files:
    #     keys = filepath.split("/")
    #     folder_ = keys[-2]
    #     file_ = re.match("(.*?)" + re.findall(r"\d+", keys[-1].split(".")[0])[0], keys[-1]).group()
    #     if folder_ in d:
    #         if file_ in d[folder_]:
    #             d[folder_][file_].append(filepath)
    #         else:
    #             d[folder_][file_] = [filepath]
    #     else:
    #         d[folder_][file_] = [filepath]
    #
    # mesh_models = []
    # # mesh_boundaries = []
    #
    # for mesh_file in mesh_files:
    #     s = vd.Mesh(mesh_file)
    #     # b = mesh.boundaries().c('red')
    #     mesh_models.append(mesh)
    #     # mesh_boundaries.append(b)
    #
    # # s = Mesh(dataurl + 'shark.ply').c('gray', 0.1).lw(0.1).lc('k')
    #
    # # this call creates the camera object needed by silhouette()
    #     show(s, bg='db', bg2='lb', interactive=False, axes=5)
    #
    #     sil = s.silhouette().c('darkred', 0.9).lw(3)
    #
    #     show(s, sil, __doc__, interactive=True)


    poses = np.load("../data/20211117_C0011/poses_bounds.npy")

    return 0

if __name__ == "__main__":
    main()