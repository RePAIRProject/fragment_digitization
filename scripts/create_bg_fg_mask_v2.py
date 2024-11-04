import torch, torchvision
#print(torch.__version__, torch.cuda.is_available())
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
import cv2
import torch

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog
from detectron2.projects import point_rend # import PointRend project
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer
import cv2 as cv
from skimage.io._plugins.pil_plugin import ndarray_to_pil, pil_to_ndarray
import os
import shutil
import re

import copy

# from rembg.bg import remove
import numpy as np
import io
from PIL import Image
import os

import natsort
from wcmatch import glob
from tqdm import tqdm


import re, collections


# #to sort file in a folder
# def sorted_alphanumeric(data):
#     convert = lambda text: int(text) if text.isdigit() else text.lower()
#     alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
#     return sorted(data, key=alphanum_key)
#
#
# MODEL_PATH = "../"
# INPUT_PATH = "/run/media/ttsesm/external_data/repair_dataset/pompeii/Pompeii_29_08_2022/3d_models/group_87/raw/RGB/RPf_00959a/images/"
# OUTPUT_PATH = "/run/media/ttsesm/external_data/repair_dataset/pompeii/Pompeii_29_08_2022/3d_models/group_87/raw/RGB/RPf_00959a/masks/"

# #define predictor
# cfg = get_cfg()
# #backbone
# cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
# cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
# cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class
# cfg.MODEL.WEIGHTS = os.path.join(MODEL_PATH, "model_final.pth")  #model we just trained
# cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  #threshold
# predictor = DefaultPredictor(cfg)


# inputdir = os.listdir(INPUT_PATH)
# inputdir = sorted_alphanumeric(inputdir)
# #for each file in the input folder
# for j, dirname in enumerate(inputdir):
#     os.makedirs(OUTPUT_PATH+dirname+"/rcnn", exist_ok=True)
#     for i, filename in enumerate(os.listdir(INPUT_PATH+dirname+"/")):
#         path_frame = INPUT_PATH + dirname+"/"+filename
#         img = cv.cvtColor(cv.imread(path_frame),cv.COLOR_BGR2RGB)
#         outputs = predictor(img)
#
        # mask = np.array(outputs['instances'].get_fields()["pred_masks"][0].to("cpu"))
        # greyscale = ndarray_to_pil(mask).convert("1")
        # #sava output in the rcnn folder
        # greyscale.save(OUTPUT_PATH+dirname+"/rcnn/"+filename[0:-3]+"png")
        # print("RCNN completed for ",dirname+"/"+filename)
        # print()

def main():
    # folder = '/run/media/ttsesm/external_data/repair_dataset/pompeii/Pompeii_17_06_2022/hi_res_images/'
    folder = '/media/lucap/big_data/datasets/repair/consolidated_fragments/group_1'
    # folder = '/run/media/ttsesm/external_data/repair_dataset/pompeii/fake_frescoes/polyga_raw/group_91/raw/RGB/'
    # last_folder = os.path.basename(os.path.normpath(folder))
    # dest_folder = folder.replace(last_folder, 'no_bg_images')
    # dest_folder = folder.replace(last_folder, 'mask_images')
    image_files = natsort.natsorted(glob.glob(folder + '**/**/*.{png,PNG,jpg,JPG}', flags=glob.BRACE | glob.GLOBSTAR))
    print(image_files)# mesh_files = list(filter(lambda k: 'final' not in k, mesh_files))

    # define predictor
    cfg = get_cfg()
    # backbone
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class
    cfg.MODEL.WEIGHTS = "model_final.pth"  # model we just trained
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # threshold
    predictor = DefaultPredictor(cfg)

    # Uncomment the following line if working with trucated image formats (ex. JPEG / JPG)
    # ImageFile.LOAD_TRUNCATED_IMAGES = True

    d = collections.defaultdict(dict)
    for filepath in image_files:
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

    for frag, imgs in tqdm(d.items()):
        # # check again on pieces RPf_00579a/b - 00594a/b - 00595a/b - 00644a/b - 00707b - 00713a/b - 00715b - 00893a/b -
        # if int(re.findall(r'\d+', frag)[0]) < 1001:
        #     continue

        dirname, basename = os.path.split(os.path.commonpath(imgs))

        dest_folder = os.path.join(dirname, "masks/")

        os.makedirs(dest_folder, exist_ok=True)

        for input_path in tqdm(imgs):

            if os.path.exists(input_path.replace("/images", "/masks")):
                continue

            filename = os.path.basename(input_path)
            src = cv2.imread(input_path,0)
            img = cv.cvtColor(src, cv.COLOR_BGR2RGB)
            outputs = predictor(img)

            mask = np.array(outputs['instances'].get_fields()["pred_masks"][0].to("cpu"))
            # mask = cv.threshold(src, 254, 255,cv.THRESH_BINARY_INV)
            # cv.imwrite(input_path.replace("/images", "/masks"), mask[1])
            greyscale = ndarray_to_pil(mask).convert("1")
            # sava output in the rcnn folder
            # greyscale.save(OUTPUT_PATH + dirname + "/rcnn/" + filename[0:-3] + "png")
            # print("RCNN completed for ", dirname + "/" + filename)
            # print()


            # # os.makedirs(dest_folder, exist_ok=True)
            # # print(input_path)
            greyscale.save(input_path.replace("/images", "/masks"))

    return 0

if __name__ == "__main__":
    main()