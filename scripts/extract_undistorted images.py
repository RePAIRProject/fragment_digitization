### Info retrieved from https://github.com/agisoft-llc/metashape-scripts/blob/master/src/samples/general_workflow.py

import Metashape.Metashape as Metashape
import os, sys, time
import natsort
from wcmatch import glob

def find_files(folder, types):
    # return [entry.path for entry in os.scandir(folder) if (entry.is_file() and os.path.splitext(entry.name)[1].lower() in types)]
    return natsort.natsorted(glob.glob(folder + '**/*.{' + types + '}', flags=glob.BRACE | glob.GLOBSTAR))

def main():
    # Checking compatibility
    compatible_major_version = "1.8"
    found_major_version = ".".join(Metashape.app.version.split('.')[:2])
    if found_major_version != compatible_major_version:
        raise Exception(
            "Incompatible Metashape version: {} != {}".format(found_major_version, compatible_major_version))

    # if len(sys.argv) < 3:
    #     print("Usage: general_workflow.py <image_folder> <output_folder>")
    #     sys.exit(1)

    ########## Manual Mode ################
    output_folder = "/run/media/ttsesm/external_data/data_for_testing/testing_metashape_alignment/RPf_00266b/metashape/"

    # load existing project
    doc = Metashape.Document()
    doc.open(output_folder + "project.psx")
    chunk = doc.chunk

    # chunk.buildDepthMaps(downscale=1, filter_mode=Metashape.MildFiltering)
    # doc.save()
    #
    # chunk.buildDenseCloud()
    # doc.save()

    for camera in chunk.cameras:
        image = camera.photo.image()
        calibration = camera.sensor.calibration
        undist = image.undistort(calibration, True, True)
        undist.save(os.path.join(output_folder, "imgs", os.path.split(camera.photo.path) [-1]))  # path should be defined

    # ########## Scripting Mode ################
    # # input_folder = "/run/media/ttsesm/external_data/repair_dataset/dataset@server/"
    # input_folder = "/run/media/ttsesm/external_data/data_for_testing/group_8/"
    # image_folders = list(filter(lambda k: '/images' in k, natsort.natsorted([x[0] for x in os.walk(input_folder)])))
    #
    # # image_folder = "./RPf_00001a/images"
    # # output_folder = "/run/media/ttsesm/external_data/data_for_testing/group_8/raw/RGB/RPf_00059b/metashape/"
    # # output_folder = "/home/ttsesm/Desktop/"
    #
    # for i, image_folder in enumerate(image_folders):
    #     # if i < 149:
    #     #     continue
    #
    #     output_folder = image_folder.replace('images', 'metashape')
    #
    #     dirname, basename = os.path.split(os.path.dirname(image_folder))
    #
    #     os.makedirs(os.path.join(output_folder, 'undistorted_images'), exist_ok=True)
    #
    #     # load existing project
    #     doc = Metashape.Document()
    #     doc.open(output_folder + "/project.psx")
    #     chunk = doc.chunk
    #
        # for camera in chunk.cameras:
        #     image = camera.photo.image()
        #     calibration = camera.sensor.calibration
        #     undist = image.undistort(calibration, True, True)
        #     undist.save(os.path.join(output_folder, 'undistorted_images', os.path.split(camera.photo.path) [-1]))  # path should be defined


    return 0

if __name__ == "__main__":
    main()