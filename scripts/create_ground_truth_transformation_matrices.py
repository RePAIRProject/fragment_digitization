import copy

from rembg.bg import remove
import numpy as np
import io
from PIL import Image
import os

import natsort
from wcmatch import glob
from tqdm import tqdm

import rembg.bg


import re, collections
import vedo as vd

import cv2

def center_mesh(m):
    cm = m.centerOfMass()
    m.shift(-cm)
    elli = vd.pcaEllipsoid(m)

    ax1 = vd.versor(elli.axis1)
    ax2 = vd.versor(elli.axis2)
    ax3 = vd.versor(elli.axis3)

    T = np.array([ax1, ax2, ax3])  # the transposed matrix is already the inverse
    # print(T)
    # print(T@ax1)

    # return m.applyTransform(T, reset=True) # <-- I had to enable reset
    return cm, T

def main():
    folder = '/run/media/ttsesm/external_data/repair_dataset/tuwien/'
    # folder = '/home/ttsesm/Desktop/sony_a7c/images'
    # folder = '/run/media/ttsesm/external_data/repair_dataset/dataset@server/'
    # last_folder = os.path.basename(os.path.normpath(folder))
    # dest_folder = folder.replace(last_folder, 'no_bg_images')
    # dest_folder = folder.replace(last_folder, 'mask_images')
    point_clouds = natsort.natsorted(glob.glob(folder + '**/*.xyz', flags=glob.BRACE | glob.GLOBSTAR))
    # mesh_files = list(filter(lambda k: 'final' not in k, mesh_files))

    # Uncomment the following line if working with trucated image formats (ex. JPEG / JPG)
    # ImageFile.LOAD_TRUNCATED_IMAGES = True

    d = collections.defaultdict(dict)
    for filepath in point_clouds:
        keys = filepath.split("/")
        folder_ = keys[-3]
        # file_ = re.match("(.*?)" + re.findall(r"\d+", keys[-2].split(".")[0])[0], keys[-1]).group()
        if folder_ in d:
            if folder_ in filepath:
                d[folder_].append(filepath)
            # else:
            #     d[folder_][file_] = [filepath]
        else:
            d[folder_] = [filepath]

    for group, pieces in tqdm(d.items()):

        dirname, basename = os.path.split(os.path.commonpath(pieces))

        pcds = vd.load(pieces)

        m = vd.merge(pcds)
        cm, t = center_mesh(m)
        # # m.shift(-cm)



        for i, pcd in enumerate(pcds):
            pcd.color(i).shift(-cm).applyTransform(t, reset=True)
            # pcd = center_mesh(pcd)
            print("Transform: {}".format(pcd.getTransform(invert=True)))

        vd.show(pcds, __doc__, axes=1).close()
    #
    #     dest_folder = os.path.join(dirname, "masks/")
    #
    #     os.makedirs(dest_folder, exist_ok=True)
    #
    #     for input_path in tqdm(imgs):
    #         filename = os.path.basename(input_path)
    #         src = cv2.imread(input_path)
    #         f = np.fromfile(input_path)
    #         # result = remove(f, alpha_matting=False, only_mask=True)
    #         # img = Image.open(io.BytesIO(result)).convert("RGB")
    #         img = segment_obj(f)
    #
    #         refined_mask = refine_mask(src, img)
    #
    #         byte_img = image_to_byte_array(refined_mask)
    #
    #         img = segment_obj(byte_img)
    #
    #         # os.makedirs(dest_folder, exist_ok=True)
    #         # print(input_path)
    #         img.save(input_path.replace("/images", "/masks"))

    return 0

if __name__ == "__main__":
    main()