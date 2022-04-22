import os
import sys
import copy

import numpy as np

import open3d as o3d
import probreg as pr
import transforms3d as t3d
from probreg import filterreg
from probreg import features
from probreg import callbacks



def main():
    # mesh1 = o3d.io.read_triangle_mesh("../data/Tombstone1_p1.obj", enable_post_processing=True, print_progress=True)
    # mesh2 = o3d.io.read_triangle_mesh("../data/Tombstone1_p2.obj", enable_post_processing=True, print_progress=True)

    mesh1 = o3d.io.read_triangle_mesh("../data/frag_1_final.ply", enable_post_processing=True, print_progress=True)
    mesh2 = o3d.io.read_triangle_mesh("../data/frag_1__final.ply", enable_post_processing=True, print_progress=True)

    # o3d.visualization.draw_geometries([mesh1, mesh2])

    pcd1 = o3d.geometry.PointCloud(mesh1.vertices)
    pcd1.normals = mesh1.vertex_normals
    pcd2 = o3d.geometry.PointCloud(mesh2.vertices)
    pcd2.normals = mesh2.vertex_normals

    # cbs = [pr.callbacks.Open3dVisualizerCallback(pcd1, pcd2)]
    # tf_param, _, _ = pr.cpd.registration_cpd(pcd1, pcd2)
    # tf_param = pr.l2dist_regs.registration_svr(pcd1,pcd2)
    # tf_param = pr.bcpd.registration_bcpd(pcd1, pcd2)
    # tf_param, _, _ = pr.filterreg.registration_filterreg(pcd1, pcd2)
    objective_type = 'pt2pt'
    tf_param, _, _ = filterreg.registration_filterreg(pcd1, pcd2, objective_type=objective_type, sigma2=1000, feature_fn=features.FPFH())#, callbacks=cbs)

    result = copy.deepcopy(pcd1)
    result.points = tf_param.transform(result.points)

    print("Matrix: \n{}\n".format(np.column_stack([tf_param.rot, tf_param.t])))

    # draw result
    pcd1.paint_uniform_color([1, 0, 0])
    pcd2.paint_uniform_color([0, 1, 0])
    result.paint_uniform_color([0, 0, 1])
    o3d.visualization.draw_geometries([pcd1, pcd2, result])

    # print("result: ", np.rad2deg(t3d.euler.mat2euler(tf_param.rot)),
    #       tf_param.scale, tf_param.t)

if __name__ == "__main__":
    main()