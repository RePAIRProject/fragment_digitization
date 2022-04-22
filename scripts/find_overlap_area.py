import copy
import os
import sys
import natsort
from wcmatch import glob

import re, collections

from vedo import *
import numpy as np

import vedo as vd
# from vedo import settings
# settings.allowInteraction=True

from tqdm import tqdm

def get_overlap_ratio(pcd1_, pcd2_, visualize=False):
    # pcd2_ = Points(pcd2_)
    pcd2_.distanceTo(pcd1_).addScalarBar()

    # print(pcd2_.pointdata.keys(), pcd2_.pointdata['Distance'].shape)

    if visualize:
        vd.show(pcd1_, pcd2_, axes=1).close()

    overlapping_points = np.where(pcd2_.pointdata['Distance'] < 0.1)

    match_points = (pcd2_.pointdata['Distance'] < 0.1).sum() # or len(overlapping_points)

    overlap_ratio = match_points / len(pcd2_.points())

    return overlap_ratio * 100, overlapping_points

def main():
    folder = '/home/ttsesm/Development/OverlapPredator/data/3dmatch/test/7-scenes-redkitchen/'
    pcd_files = natsort.natsorted(glob.glob(folder + '**/*.{ply,obj}', flags=glob.BRACE | glob.GLOBSTAR))

    tranf_files = natsort.natsorted(glob.glob(folder + '**/*.txt', flags=glob.BRACE | glob.GLOBSTAR))

    # pc1, pc2, pc3 = load('/home/ttsesm/Development/OverlapPredator/data/3dmatch/test/7-scenes-redkitchen/cloud_bin_?.ply')

    # pc1 = Points('/home/ttsesm/Development/OverlapPredator/data/3dmatch/test/7-scenes-redkitchen/cloud_bin_0.ply').applyTransform(np.loadtxt(tranf_files[0], skiprows=1), reset=True)
    # pc2 = Points('/home/ttsesm/Development/OverlapPredator/data/3dmatch/test/7-scenes-redkitchen/cloud_bin_1.ply').applyTransform(np.loadtxt(tranf_files[1], skiprows=1), reset=True)
    #
    # get_overlap_ratio(pc1, pc2, True)

    # pc1a = Points(pc1)
    # pc1a.distanceTo(pc2).addScalarBar()
    # print(pc1a.pointdata.keys(), pc1a.pointdata['Distance'].shape)
    #
    # vd.show(pc1a, pc2, axes=1).close()

    #
    # pc1 = pc1.applyTransform(np.loadtxt(tranf_files[0], skiprows=1), reset=True)
    # pc1 = pc2.applyTransform(np.loadtxt(tranf_files[1], skiprows=1), reset=True)
    # pc1 = pc3.applyTransform(np.loadtxt(tranf_files[2], skiprows=1), reset=True)
    #
    # pc1.distanceTo(pc2)

    pcds = []
    for i, input_path in enumerate(pcd_files):
        t = np.loadtxt(tranf_files[i], skiprows=1)
        pcd = vd.Points(input_path).color(i).applyTransform(t, reset=True)
        pcds.append(pcd)

        # pcd.delete()
        if len(pcds) > 1:
            overlap_ratio, _ = get_overlap_ratio(pcds[0], pcd)
            print("Pcd0 --> Pcd{}: {}%".format(i, overlap_ratio))


        # vd.show(pcds, axes=1).close()
        #
        # print(input_path)


    return 0

if __name__ == "__main__":
    main()