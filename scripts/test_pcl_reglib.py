import os
import time

import numpy as np

from utils.pcl_registration_module import reglib

import vedo as vd


def main():
    # Only needed if you want to use manually compiled library code
    # reglib.load_library(os.path.join(os.curdir, "cmake-build-debug"))

    # # Load you data
    # source_points = reglib.load_data(os.path.join(os.curdir, "../pcl_registration_module/files", "model_points.csv"))
    # target_points = reglib.load_data(os.path.join(os.curdir, "../pcl_registration_module/files", "scene_points.csv"))

    # m1 = vd.Mesh("../data/Tombstone1_p1.obj").texture("../data/Tombstone1_low.jpg")  # .rotate(90)
    # m2 = vd.Mesh("../data/Tombstone1_p2.obj").texture("../data/Tombstone1_low.jpg")

    m1 = vd.Mesh("../data/frag_1_final.ply")  # .rotate(90)
    m2 = vd.Mesh("../data/frag_1__final.ply")

    # pc = np.column_stack([m1.points(), m1.normals()])
    source_points = np.array(m1.points()).astype(float)
    # pcTest = np.column_stack([m2.points(), m2.normals()])
    target_points = np.array(m2.points()).astype(float)

    # Run the registration algorithm
    start = time.time()
    trans = reglib.icp(source=source_points, target=target_points, nr_iterations=1, epsilon=0.01,
                       inlier_threshold=0.05, distance_threshold=500.0, downsample=0, visualize=True)
                       #resolution=12.0, step_size=0.5, voxelize=0)

    # trans = reglib.ndt(source=source_points, target=target_points, nr_iterations=1, epsilon=0.01,
    #                    inlier_threshold=0.05, distance_threshold=500.0, downsample=0, visualize=True)

    print("Runtime:", time.time() - start)
    print(trans)


if __name__ == "__main__":
    main()