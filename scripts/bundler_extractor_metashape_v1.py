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

    # input_folder = "/run/media/ttsesm/external_data/repair_dataset/dataset@server/"
    input_folder = "/run/media/ttsesm/external_data/repair_dataset/pompeii/Pompeii_14_03_2022/sony_images/"
    # input_folder = "/run/media/ttsesm/external_data/data_for_testing/group_8/raw/RGB/RPf_00047b/"
    image_folders = list(filter(lambda k: '/images' in k, natsort.natsorted([x[0] for x in os.walk(input_folder)])))

    # image_folder = "./RPf_00001a/images"
    # output_folder = "/run/media/ttsesm/external_data/data_for_testing/group_8/raw/RGB/RPf_00059b/metashape/"

    for i, image_folder in enumerate(image_folders):
        if i < 220:
            continue

        output_folder = image_folder.replace('/images', '/metashape')

        dirname, basename = os.path.split(os.path.dirname(image_folder))

        os.makedirs(output_folder, exist_ok=True)

        photos = find_files(image_folder, "png,jpg,jpeg,tif,tiff,PNG,JPG")
        masks = find_files(image_folder.replace("/images", "/masks"), "png,jpg,jpeg,tif,tiff,PNG,JPG")

        # Metashape.app.settings.setValue("main/gpu_enable_cuda", "0")
        doc = Metashape.Document()
        doc.save(output_folder + '/project.psx')

        chunk = doc.addChunk()

        chunk.addPhotos(photos)
        doc.save()
        print(str(len(chunk.cameras)) + " images loaded")

        for j, f in enumerate(chunk.cameras):
            chunk.generateMasks(path=masks[j], masking_mode=Metashape.MaskingModeFile, mask_operation=Metashape.MaskOperationReplacement, cameras=chunk.cameras[j])
        doc.save()
        print(str(len(chunk.cameras)) + " mask images loaded")

        chunk.matchPhotos()
        doc.save()

        chunk.alignCameras()
        doc.save()

        chunk.optimizeCameras(adaptive_fitting=True, tiepoint_covariance=True)

        chunk.buildDepthMaps(downscale=2, filter_mode=Metashape.MildFiltering)
        doc.save()

        chunk.buildDenseCloud()
        doc.save()

        chunk.buildModel(source_data=Metashape.DenseCloudData, face_count=Metashape.HighFaceCount)
        doc.save()

        chunk.buildUV(mapping_mode=Metashape.GenericMapping, texture_size=4096)
        doc.save()

        chunk.buildTexture(blending_mode=Metashape.MosaicBlending, texture_size=4096, ghosting_filter=True)
        doc.save()

        if chunk.model:
            chunk.exportModel(output_folder + '/model.obj', texture_format=Metashape.ImageFormatPNG)

            chunk.exportCameras(output_folder + '/cameras.out', format=Metashape.CamerasFormatBundler)

        os.makedirs(os.path.join(output_folder, 'undistorted_images'), exist_ok=True)
        for camera in chunk.cameras:
            image = camera.photo.image()
            calibration = camera.sensor.calibration
            undist = image.undistort(calibration, True, True)
            undist.save(os.path.join(output_folder, 'undistorted_images', os.path.split(camera.photo.path) [-1]))  # path should be defined

    return 0

if __name__ == "__main__":
    main()