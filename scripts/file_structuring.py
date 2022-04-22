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

LETTERS = {index: str(letter) for index, letter in enumerate(ascii_lowercase, start=1)}

# def update_metadata():
#     metadata = {'Filename(s)': None, 'ID': None, 'Link': None,
#               'Texture': {'Low_res_texture': None, 'High_res_texture': None},
#               'Raw File(s)': None, 'RGB File(s)': None, 'Hyperspectral File(s)': None, 'Acquisition Date': None,
#               'Artistic Style': None, 'Fresco Family': None,
#               'Geometric Data': {'Points': None, 'Faces': None, 'Polygons': None, 'Position': None, 'Scale': None,
#                                  'CenterOfMass': None, 'Avg. Size': None, 'Diag. Size': None, 'Bounds': None,
#                                  'Transformation': None}}
#
#     return metadata

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

def getParentDir(path, level=1):
    return os.path.sep.join(path.split(os.path.sep)[level:])


def main():
    # folder = '../data/json_test/'
    folder = '/run/media/ttsesm/51A6ECAF2B33CC01/groups/'
    dest_folder = '/home/ttsesm/Data/repair_dataset/dataset@server/'
    mesh_files = natsort.natsorted(glob.glob(folder + '**/*.{ply,obj}', flags=glob.BRACE | glob.GLOBSTAR))
    mesh_files = list(filter(lambda k: 'final' not in k, mesh_files))

    tranf_files =  natsort.natsorted(glob.glob(folder + '**/*.txt', flags=glob.BRACE | glob.GLOBSTAR))

    d = collections.defaultdict(dict)
    for filepath in mesh_files:
        keys = filepath.split("/")
        folder_ = keys[-2]
        file_ = re.match("(.*?)" + re.findall(r"\d+", keys[-1].split(".")[0])[0], keys[-1]).group()
        if folder_ in d:
            if file_ in d[folder_]:
                d[folder_][file_].append(filepath)
            else:
                d[folder_][file_] = [filepath]
        else:
            d[folder_][file_] = [filepath]


    id = 1
    for box, frags in d.items():
        print(box)
        # if box in ["box_1", "box_2", "box_3"]:
        #     continue
        dirname = None

        transf_d = {}
        with open(list(filter(lambda k: box in k, tranf_files))[0]) as f:
            for group in f.read().split('\n\n\n'):
                key, *val = group.split('\n')
                transf_d[key] = val

        mesh_models = []
        kk = 0
        for frag, mesh_files in frags.items():
            for idx, mesh_file in enumerate(mesh_files, start=1):
                mesh = vd.Mesh(mesh_file)

                # o3d_mesh = o3d.io.read_triangle_mesh("/home/ttsesm/Development/repair/data/box_8/frag_1.ply", enable_post_processing=True, print_progress=True)
                # extract_texture_maps(o3d_mesh)

                # # export the mesh including data
                # mm = trimesh.load(mesh_file, process=True, maintain_order=True, force='mesh')
                # m = copy.deepcopy(mm)
                # m.visual = mm.visual.to_texture()
                # # test_color = [0, 255, 0, 255]
                # # m.visual.face_colors = test_color
                # obj, data = trimesh.exchange.obj.export_obj(m, include_texture=True, return_texture=True)
                #
                # # with "/home/ttsesm/Development/repair/data/box_8/" as path:
                # path = "/home/ttsesm/Development/repair/data/box_8/"
                # # where is the OBJ file going to be saved
                # obj_path = os.path.join(path, 'test.obj')
                # with open(obj_path, 'w') as f:
                #     f.write(obj)
                # # save the MTL and images
                # for k, v in data.items():
                #     with open(os.path.join(path, k), 'wb') as f:
                #         f.write(v)

                # mesh_models.append(mesh)
                # dest_path = os.path.join(dest_folder+box.replace("box", "group"), "raw", "3D")
                dest_path = os.path.join(dest_folder + box.replace("box", "group"), "processed")
                # dest_filename = frag.replace(frag, "RPf_")+str(id).zfill(5)+LETTERS[idx]+".ply"
                dest_filename = frag.replace(frag, "RPf_") + str(id).zfill(5) + ".ply"
                os.makedirs(dest_path, exist_ok=True)
                shutil.copy2(mesh_file, os.path.join(dest_path, dest_filename))

                raw_files = natsort.natsorted(glob.glob(os.path.join(dest_folder+box.replace("box", "group"), "raw", "3D") + '**/'+ os.path.splitext(dest_filename)[0]+'*', flags=glob.BRACE | glob.GLOBSTAR))
                raw_files = [getParentDir(x, level=-4) for x in raw_files]
                raw_files = {k: [v, {'Transformation': ""}] for k, v in dict(enumerate(raw_files)).items()}

                if len(raw_files) < 3:
                    for k, v in (j for i, j in enumerate(raw_files.items()) if not i==list(raw_files.keys())[-1]):
                        raw_files[k][1]["Transformation"] = [list(map(float, item.split())) for item in list(transf_d.values())[kk]]
                        kk += 1
                else:
                    print(os.path.join(dest_path, dest_filename))

                rgb_files = natsort.natsorted(glob.glob(
                    os.path.join(dest_folder + box.replace("box", "group"), "raw", "RGB") + '**/' +
                    os.path.splitext(dest_filename)[0] + '*', flags=glob.BRACE | glob.GLOBSTAR))
                rgb_files = [getParentDir(x, level=-4) for x in rgb_files]

                metadata = {'Filename(s)': dict(enumerate([dest_filename])), 'ID': os.path.splitext(dest_filename)[0], 'Link': "",
                            'Texture': {'Low_res_texture': "", 'High_res_texture': ""},
                            'Raw 3D File(s)': raw_files, 'RGB File(s)': dict(enumerate(rgb_files)), 'Hyperspectral File(s)': "",
                            'Acquisition Date': "17-11-2021",
                            'Artistic Style': "", 'Fresco Family': "", 'Weight': "",
                            'Geometric Data': {'Points': len(mesh.points()), 'Faces': len(mesh.faces()), 'Polygons': len(mesh.cells()), 'Position': list(mesh.centerOfMass()),
                                               'Scale': list(mesh.scale()),
                                               'CenterOfMass': list(mesh.centerOfMass()), 'Avg. Size': mesh.averageSize(), 'Diag. Size': mesh.diagonalSize(),
                                               'Bounds': {'x': mesh.bounds()[0:2], 'y': mesh.bounds()[2:4], 'z': mesh.bounds()[4:]},
                                               'Transformation': ""}}

                opts = jsbeautifier.default_options()
                opts.indent_size = 4
                data = jsbeautifier.beautify(json.dumps(metadata, sort_keys=True), opts)
                # data = jsbeautifier.beautify(json.dumps(metadata), opts)


                with open(os.path.join(dest_path, dest_filename.replace('ply','json')), 'w') as fp:
                    # json.dump(dict, fp, indent=2, sort_keys=True)
                    fp.write(data)

            id += 1


    return 0

if __name__ == "__main__":
    main()