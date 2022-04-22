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

def find_viz_shape(n):
    wIndex = np.ceil(np.sqrt(n)).astype(int)
    hIndex = np.ceil(n/wIndex).astype(int)

    return [hIndex, wIndex]

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


def main():
    folder = '/home/ttsesm/Data/repair_dataset/pompei_17_11_2021/3d_models/'
    # folder = '../data/box_8/'
    mesh_files = natsort.natsorted(glob.glob(folder + '**/*.{ply,obj}', flags=glob.BRACE | glob.GLOBSTAR))

    mesh_files = list(filter(lambda k: 'final' not in k, mesh_files))

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


    for box, frags in d.items():
        print(box)
        dirname = None

        mesh_models = []
        for frag, mesh_file in frags.items():
            mesh = vd.Mesh(mesh_file[-1])
            mesh_models.append(mesh)


            if dirname == None:
                dirname = os.path.dirname(mesh_file[-1])



        vp = Plotter(shape=find_viz_shape(len(mesh_models)), axes=0, interactive=0, sharecam=False)
        video = Video(os.path.join(dirname, box+".gif"), backend='ffmpeg')  # backend='opencv/ffmpeg'
        video.options = "-b:v 8000k -filter_complex \"[0:v] split [a][b];[a] palettegen=stats_mode=single [p];[b][p] paletteuse=new=1\""
        # vp = Plotter(axes=0, interactive=0)


        for t in np.arange(0, 1, 0.005):
            for i, mesh_model in enumerate(mesh_models):
                vp.show(mesh_model.lighting(style='ambient'), at=i)
                cam = vp.renderer.GetActiveCamera()
                cam.Azimuth(2)
            video.addFrame()

        video.close()
        vp.show().close()

    return 0

if __name__ == "__main__":
    main()