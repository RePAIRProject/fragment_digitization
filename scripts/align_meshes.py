import os
import sys
import natsort
from glob import glob
from subprocess import call, run

import copy

import numpy as np

import vedo as vd
import trimesh
import pymeshlab
import open3d as o3d
import cv2 as cv
import probreg as pr
import transforms3d as t3d

# def pairwise_registration(source, target):
#     print("Apply point-to-plane ICP")
#     icp_coarse = o3d.pipelines.registration.registration_icp(
#         source, target, max_correspondence_distance_coarse, np.identity(4),
#         o3d.pipelines.registration.TransformationEstimationPointToPlane())
#     icp_fine = o3d.pipelines.registration.registration_icp(
#         source, target, max_correspondence_distance_fine,
#         icp_coarse.transformation,
#         o3d.pipelines.registration.TransformationEstimationPointToPlane())
#     transformation_icp = icp_fine.transformation
#     information_icp = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
#         source, target, max_correspondence_distance_fine,
#         icp_fine.transformation)
#     return transformation_icp, information_icp
#
#
# def full_registration(pcds, max_correspondence_distance_coarse,
#                       max_correspondence_distance_fine):
#     pose_graph = o3d.pipelines.registration.PoseGraph()
#     odometry = np.identity(4)
#     pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(odometry))
#     n_pcds = len(pcds)
#     for source_id in range(n_pcds):
#         for target_id in range(source_id + 1, n_pcds):
#             transformation_icp, information_icp = pairwise_registration(
#                 pcds[source_id], pcds[target_id])
#             print("Build o3d.pipelines.registration.PoseGraph")
#             if target_id == source_id + 1:  # odometry case
#                 odometry = np.dot(transformation_icp, odometry)
#                 pose_graph.nodes.append(
#                     o3d.pipelines.registration.PoseGraphNode(
#                         np.linalg.inv(odometry)))
#                 pose_graph.edges.append(
#                     o3d.pipelines.registration.PoseGraphEdge(source_id,
#                                                              target_id,
#                                                              transformation_icp,
#                                                              information_icp,
#                                                              uncertain=False))
#             else:  # loop closure case
#                 pose_graph.edges.append(
#                     o3d.pipelines.registration.PoseGraphEdge(source_id,
#                                                              target_id,
#                                                              transformation_icp,
#                                                              information_icp,
#                                                              uncertain=True))
#     return pose_graph
#
# def visualize(mesh):
#     vis = o3d.visualization.Visualizer()
#     vis.create_window()
#     vis.add_geometry(mesh)
#     opt = vis.get_render_option()
#     opt.show_coordinate_frame = True
#     opt.background_color = np.asarray([0.5, 0.5, 0.5])
#     vis.run()
#     vis.destroy_window()

if __name__ == "__main__":
    # print('Start!!!!')

    # m = vd.Mesh("../data/Tombstone1.obj")
    mesh1 = o3d.io.read_triangle_mesh("../data/Tombstone1_p1.obj", enable_post_processing=True, print_progress=True)
    mesh2 = o3d.io.read_triangle_mesh("../data/Tombstone1_p2.obj", enable_post_processing=True, print_progress=True)

    # # pcds = []

    pcd1 = o3d.geometry.PointCloud(mesh1.vertices)
    pcd1.normals = mesh1.vertex_normals
    # pcds.append(pcd1)
    pcd2 = o3d.geometry.PointCloud(mesh2.vertices)
    pcd2.normals = mesh2.vertex_normals
    # pcds.append(pcd2)

    # # # o3d.visualization.draw_geometries([mesh1])
    # # # visualize(pcd1)
    # # # vd.show(m, __doc__, axes=1, viewup='z').close()
    # #
    # # voxel_size = 0.02
    # #
    # # print("Full registration ...")
    # # max_correspondence_distance_coarse = voxel_size * 15
    # # max_correspondence_distance_fine = voxel_size * 1.5
    # # with o3d.utility.VerbosityContextManager(
    # #         o3d.utility.VerbosityLevel.Debug) as cm:
    # #     pose_graph = full_registration(pcds,
    # #                                    max_correspondence_distance_coarse,
    # #                                    max_correspondence_distance_fine)
    # #
    # # print("Optimizing PoseGraph ...")
    # # option = o3d.pipelines.registration.GlobalOptimizationOption(
    # #     max_correspondence_distance=max_correspondence_distance_fine,
    # #     edge_prune_threshold=0.25,
    # #     reference_node=0)
    # # with o3d.utility.VerbosityContextManager(
    # #         o3d.utility.VerbosityLevel.Debug) as cm:
    # #     o3d.pipelines.registration.global_optimization(
    # #         pose_graph,
    # #         o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
    # #         o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
    # #         option)
    # #
    # # print("Transform points and display")
    # # for point_id in range(len(pcds)):
    # #     print(pose_graph.nodes[point_id].pose)
    # #     pcds[point_id].transform(pose_graph.nodes[point_id].pose)
    # # o3d.visualization.draw_geometries(pcds,
    # #                                   zoom=0.3412,
    # #                                   front=[0.4257, -0.2125, -0.8795],
    # #                                   lookat=[2.6172, 2.0475, 1.532],
    # #                                   up=[-0.0694, -0.9768, 0.2024])

    m1 = vd.Mesh("../data/Tombstone1_p1.obj").texture("../data/Tombstone1_low.jpg")#.rotate(90)
    m2 = vd.Mesh("../data/Tombstone1_p2.obj").texture("../data/Tombstone1_low.jpg")

    idx = np.arange(0, len(m1.points()))
    sampled_points_idx = np.random.choice(idx, len(m2.points()), replace=False)
    p2_ = m1.points()[sampled_points_idx]

    # # vd.show(m1, "Part 1", at=0, N=2, axes=1)
    # # vd.show(m2, "Part 2", at=1, interactive=True).close()
    #
    # matrix = trimesh.registration.mesh_other(vd.vedo2trimesh(m1), vd.vedo2trimesh(m2), samples=500, scale=False, icp_first=10, icp_final=50)[0]
    np.set_printoptions(suppress=True)
    # print("Matrix1: {}\n".format(matrix))

    p1 = vd.Points(m1.points()[sampled_points_idx]).c("green").alpha(0.3)
    p1.normals = m1.normals()[sampled_points_idx]
    p2 = vd.Points(m2.points()).c("red").alpha(0.3)
    p2.normals = m2.normals()

    # t1 = trimesh.points.PointCloud(m1.points())
    # t2 = trimesh.points.PointCloud(m2.points())


    # matrix3 = trimesh.registration.procrustes(m1.points()[sampled_points_idx], m2.points(), reflection=False, scale=False)[0]
    # print("Matrix3: \n{}\n".format(matrix3))
    #
    # matrix2 = trimesh.registration.icp(m1.points(), m2.points(), matrix3)[0]
    # print("Matrix2: \n{}\n".format(matrix2))

    aligned = vd.procrustesAlignment([p1, p2], rigid=False)

    # aligned_pts1 = p2.clone().alignTo(p1, rigid=True, invert=True, useCentroids=False)

    # vd.show(m1, m2, "Before", at=0, N=2, axes=1)
    # vd.show(m2, m1.clone().applyTransform(matrix2), "After", at=1, interactive=True).close()
    vd.show(p1, p2, "Before", at=0, N=2, axes=1)
    vd.show(aligned, at=1, interactive=1).close()

    # detector = cv.ppf_match_3d_PPF3DDetector(0.025, 0.05)
    #
    # print('Loading model...')
    # # pc = cv.ppf_match_3d.loadPLYSimple("../data/parasaurolophus_6700.ply", 1)
    # pc = np.column_stack([m1.points(), m1.normals()])
    #
    # print('Training...')
    # detector.trainModel(pc)
    #
    # print('Loading scene...')
    # # pcTest = cv.ppf_match_3d.loadPLYSimple("data/%s.ply" % scenename, 1)
    # pcTest = np.column_stack([m2.points(), m2.normals()])
    #
    # print('Matching...')
    # results = detector.match(pcTest, 1.0 / 40.0, 0.05)

    # # cbs = [pr.callbacks.Open3dVisualizerCallback(pcd1, pcd2)]
    # # tf_param, _, _ = pr.cpd.registration_cpd(pcd1, pcd2)
    # # tf_param = pr.l2dist_regs.registration_svr(pcd1,pcd2)
    # tf_param = pr.bcpd.registration_bcpd(pcd1, pcd2)
    # # tf_param, _, _ = pr.filterreg.registration_filterreg(pcd1, pcd2)
    #
    # result = copy.deepcopy(pcd1)
    # result.points = tf_param.transform(result.points)
    #
    # print("Matrix: \n{}\n".format(np.column_stack([tf_param.rot, tf_param.t])))
    #
    # # draw result
    # pcd1.paint_uniform_color([1, 0, 0])
    # pcd2.paint_uniform_color([0, 1, 0])
    # result.paint_uniform_color([0, 0, 1])
    # o3d.visualization.draw_geometries([pcd1, pcd2, result])
    #
    # # print("result: ", np.rad2deg(t3d.euler.mat2euler(tf_param.rot)),
    # #       tf_param.scale, tf_param.t)

    # print('End!!!!')
    os._exit(0)