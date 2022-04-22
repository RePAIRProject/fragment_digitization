import copy
import os
import sys
import natsort
from wcmatch import glob

import re, collections

from vedo import *
import numpy as np

import vedo as vd
from vedo import settings
settings.allowInteraction=True

import shutil

from string import ascii_lowercase

import open3d as o3d
import trimesh

import json
import jsbeautifier

import pymeshlab

def o3d2pymesh(o3d_mesh):
    m = pymeshlab.Mesh(vertex_matrix=np.array(o3d_mesh.vertices), face_matrix=np.array(o3d_mesh.triangles),
                       v_normals_matrix=np.array(o3d_mesh.vertex_normals),
                       v_color_matrix=np.insert(np.array(o3d_mesh.vertex_colors), 3, 1, axis=1))

    ms = pymeshlab.MeshSet()
    ms.add_mesh(m)


    return ms

def extract_texture_maps(mesh_):
    m = o3d2pymesh(mesh_)
    # m.surface_reconstruction_screened_poisson(depth=8, pointweight=1, preclean=True)

    m.parametrization_voronoi_atlas(overlapflag=True)
    m.transfer_vertex_attributes_to_texture_1_or_2_meshes(sourcemesh=0, targetmesh=1, textname="test.png")

    m.set_current_mesh(1)
    mlab_mesh = m.current_mesh()
    m.save_current_mesh("bone_saved.obj")

    # reco_mesh = pymesh2o3d(mlab_mesh)

    return #reco_mesh


# def main():
#     # folder = '../data/json_test/'
#     # # folder = '/home/ttsesm/Data/repair_dataset/pompei_17_11_2021/3d_models/'
#     # # dest_folder = '/home/ttsesm/Data/repair_dataset/dataset@server/'
#     # mesh_files = natsort.natsorted(glob.glob(folder + '**/*.{ply,obj}', flags=glob.BRACE | glob.GLOBSTAR))
#     # mesh_files = list(filter(lambda k: 'final' not in k, mesh_files))
#     #
#     # d = collections.defaultdict(dict)
#     # for filepath in mesh_files:
#     #     keys = filepath.split("/")
#     #     folder_ = keys[-2]
#     #     file_ = re.match("(.*?)" + re.findall(r"\d+", keys[-1].split(".")[0])[0], keys[-1]).group()
#     #     if folder_ in d:
#     #         if file_ in d[folder_]:
#     #             d[folder_][file_].append(filepath)
#     #         else:
#     #             d[folder_][file_] = [filepath]
#     #     else:
#     #         d[folder_][file_] = [filepath]
#
#
#     id = 1
#     for box, frags in d.items():
#         print(box)
#
#         for frag, mesh_files in frags.items():
#             for idx, mesh_file in enumerate(mesh_files, start=1):
#                 mesh = vd.Mesh(mesh_file)
#
#                 o3d_mesh = o3d.io.read_triangle_mesh("/home/ttsesm/Development/repair/data/box_8/frag_2.ply", enable_post_processing=True, print_progress=True)
#                 # o3d_mesh = o3d.io.read_triangle_mesh(mesh_file, enable_post_processing=True, print_progress=True)
#                 extract_texture_maps(o3d_mesh)
#
#     return 0


def main():

    # create a new MeshSet
    ms = pymeshlab.MeshSet()

    ms.load_new_mesh("/home/ttsesm/Development/repair/data/box_8/frag_2.ply")

    # ms.parametrization_voronoi_atlas(overlapflag=True)
    # ms.transfer_vertex_attributes_to_texture_1_or_2_meshes(sourcemesh=0, targetmesh=1, textname="frag_2.png")

    # ms.parametrization_trivial_per_triangle(border=0.5)
    # ms.transfer_vertex_color_to_texture(textname="frag_2.png")

    ms.set_current_mesh(0)
    ms.save_current_mesh("frag_2.obj")


if __name__ == "__main__":
    main()