import cv2
import time
import matplotlib.pyplot as plt
import segmentation_refinement as refine

import natsort
from wcmatch import glob
from tqdm import tqdm

def main():
    folder = '/run/media/ttsesm/external_data/data_for_testing/test_rembg_segmentation/images/'
    # folder = '/home/ttsesm/Desktop/sony_a7c/images'
    # folder = '/run/media/ttsesm/external_data/repair_dataset/dataset@server/'
    # last_folder = os.path.basename(os.path.normpath(folder))
    # dest_folder = folder.replace(last_folder, 'no_bg_images')
    # dest_folder = folder.replace(last_folder, 'mask_images')
    image_files = natsort.natsorted(glob.glob(folder + '**/*.{png,PNG,jpg,JPG}', flags=glob.BRACE | glob.GLOBSTAR))

    for img in tqdm(image_files):
        image = cv2.imread(img)
        mask = cv2.imread(img.replace('/images', '/masks'), cv2.IMREAD_GRAYSCALE)

        # model_path can also be specified here
        # This step takes some time to load the model
        refiner = refine.Refiner(device='cuda:0', model_folder='/home/ttsesm/Development/repair/') # device can also be 'cpu'

        # Fast - Global step only.
        # Smaller L -> Less memory usage; faster in fast mode.
        output = refiner.refine(image, mask, fast=False, L=900)

        # this line to save output
        cv2.imwrite(img.replace('/images', '/refined_masks'), output)

        # plt.imshow(output)
        # plt.show()

    return 0

if __name__ == "__main__":
    main()