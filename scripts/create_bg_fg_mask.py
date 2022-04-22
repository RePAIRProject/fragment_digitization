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

import cv2

def toImgOpenCV(imgPIL): # Conver imgPIL to imgOpenCV
    i = np.array(imgPIL) # After mapping from PIL to numpy : [R,G,B,A]
                         # numpy Image Channel system: [B,G,R,A]
    red = i[:,:,0].copy(); i[:,:,0] = i[:,:,2].copy(); i[:,:,2] = red;
    return i;

def toImgPIL(imgOpenCV): return Image.fromarray(cv2.cvtColor(imgOpenCV, cv2.COLOR_BGR2RGB));

def image_to_byte_array(image:Image):
    if image.format == None:
        image.format = 'PNG'

    imgByteArr = io.BytesIO()
    image.save(imgByteArr, format=image.format)
    imgByteArr = imgByteArr.getvalue()
    return imgByteArr

def refine_mask(src, img):
    img_cv = toImgOpenCV(img)
    # cv2.namedWindow('img_cv', cv2.WINDOW_NORMAL)
    # cv2.imshow("img_cv", img_cv)
    # img_cv_gray = copy.deepcopy(img_cv)

    img_cv_gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    # cv2.namedWindow('img_cv_gray', cv2.WINDOW_NORMAL)
    # cv2.imshow("img_cv_gray", img_cv_gray)

    # Otsu's thresholding after Gaussian filtering
    blur = cv2.GaussianBlur(img_cv_gray, (5, 5), 0)
    ret3, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # th3 = cv2.adaptiveThreshold(img_cv_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    # Find largest contour in intermediate image
    cnts, _ = cv2.findContours(th3, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnt = max(cnts, key=cv2.contourArea)

    # Output
    mask = np.zeros(th3.shape, np.uint8)
    cv2.drawContours(mask, [cnt], -1, 255, cv2.FILLED)
    # out = cv2.bitwise_and(img, out)

    masked = cv2.bitwise_and(src, img_cv, mask=mask)

    # # show the output images
    # cv2.namedWindow('th3', cv2.WINDOW_NORMAL)
    # cv2.imshow("th3", th3)
    # cv2.namedWindow('mask', cv2.WINDOW_NORMAL)
    # cv2.imshow("mask", mask)
    # cv2.namedWindow('masked', cv2.WINDOW_NORMAL)
    # cv2.imshow("masked", masked)
    # cv2.waitKey(0)

    return toImgPIL(masked)

def segment_obj(f):
    result = remove(f, alpha_matting=False, only_mask=True)
    img = Image.open(io.BytesIO(result)).convert("RGB")

    return img

def main():
    folder = '/run/media/ttsesm/external_data/repair_dataset/pompeii/Pompeii_14_03_2022/sony_images/'
    # folder = '/home/ttsesm/Desktop/sony_a7c/images'
    # folder = '/run/media/ttsesm/external_data/repair_dataset/dataset@server/'
    # last_folder = os.path.basename(os.path.normpath(folder))
    # dest_folder = folder.replace(last_folder, 'no_bg_images')
    # dest_folder = folder.replace(last_folder, 'mask_images')
    image_files = natsort.natsorted(glob.glob(folder + '**/*.{png,PNG,jpg,JPG}', flags=glob.BRACE | glob.GLOBSTAR))
    # mesh_files = list(filter(lambda k: 'final' not in k, mesh_files))

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

        dirname, basename = os.path.split(os.path.commonpath(imgs))

        dest_folder = os.path.join(dirname, "masks/")

        os.makedirs(dest_folder, exist_ok=True)

        for input_path in tqdm(imgs):
            filename = os.path.basename(input_path)
            src = cv2.imread(input_path)
            f = np.fromfile(input_path)
            # result = remove(f, alpha_matting=False, only_mask=True)
            # img = Image.open(io.BytesIO(result)).convert("RGB")
            img = segment_obj(f)

            refined_mask = refine_mask(src, img)

            byte_img = image_to_byte_array(refined_mask)

            img = segment_obj(byte_img)

            # os.makedirs(dest_folder, exist_ok=True)
            # print(input_path)
            img.save(input_path.replace("/images", "/masks"))

    return 0

if __name__ == "__main__":
    main()