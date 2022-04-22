import os
import sys
import natsort
from glob import glob
from subprocess import call, run

import vedo as vd
import trimesh
import pymeshlab

if __name__ == "__main__":
    # print('Start!!!!')

    # ms = pymeshlab.MeshSet()
    # ms.load_new_mesh('/home/ttsesm/Data/repair_dataset/presious/assembled_samples/DoraColumn/DoraColumn_Reassembled/DoraColumn_Reassembled.obj')
    # em = vd.Volume(vd.dataurl + "embryo.tif").isosurface(80)
    m = vd.Mesh("/home/ttsesm/Data/repair_dataset/presious/assembled_samples/DoraColumn/DoraColumn_Reassembled/DoraColumn_Reassembled.obj", c='gray')
    # m = vd.Mesh("/home/ttsesm/Data/repair_dataset/presious/assembled_samples/DoraColumnFoundation/DoraColumnFoundation_Reassembled/DoraColumnFoundation_Reassembled.obj")
    # m = vd.Mesh("/home/ttsesm/Data/repair_dataset/presious/assembled_samples/Tombstone/Reassembled_Tombstone/Reassembled_Tombstone.obj", c='gray')

    # # trimesh_m = trimesh.load("/home/ttsesm/Data/repair_dataset/presious/assembled_samples/DoraColumn/DoraColumn_Reassembled/DoraColumn_Reassembled.obj", force='mesh', skip_texture=True)
    # trimesh_m = trimesh.load("/home/ttsesm/Data/repair_dataset/presious/assembled_samples/DoraColumn/DoraColumn_Reassembled/DoraColumn_Reassembled.ply", force='mesh', skip_texture=True)
    #
    trimesh_m = vd.vedo2trimesh(m)
    # m = vd.trimesh2vedo(trimesh_m)
    # # test = trimesh.graph.split(trimesh_m)
    split_all = trimesh_m.split(only_watertight=False)

    # return the list of the largest 10 connected meshes:
    splitem = m.splitByConnectivity(maxdepth=5)#[0:9]

    # vd.show(em, __doc__, axes=1, viewup='z').close()

    vd.show(m, "Before", at=0, N=2, axes=1)
    vd.show(splitem, "After", at=1, interactive=True).close()

    # print('End!!!!')
    os._exit(0)