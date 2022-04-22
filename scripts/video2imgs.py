import copy

import cv2
import time
import os
import sys
import natsort
from wcmatch import glob

import shutil

import re, collections

from skimage.metrics import structural_similarity as compare_ssim

def video_to_frames(input_loc, output_loc):
    """Function to extract frames from input video file
    and save them as separate frames in an output directory.
    Args:
        input_loc: Input video file.
        output_loc: Output directory to save the frames.
    Returns:
        None
    """
    try:
        os.mkdir(output_loc)
    except OSError:
        pass
    # Log the time
    time_start = time.time()
    # Start capturing the feed
    cap = cv2.VideoCapture(input_loc)
    # Find the number of frames
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
    print ("Number of frames: ", video_length)
    count = 0
    print ("Converting video..\n")

    # Start converting the video
    while cap.isOpened():
        # Extract the frame
        ret, frame = cap.read()
        if not ret:
            continue

        if 'prev_frame' in dir():
            grayA = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            grayB = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

            # compute the Structural Similarity Index (SSIM) between the two
            # images, ensuring that the difference image is returned
            (score, diff) = compare_ssim(grayA, grayB, full=True)
            diff = (diff * 255).astype("uint8")
            # print("SSIM: {}".format(score))

        # Write the results back to output location.
        if 'score' in dir() and score < 0.95:
            print("SSIM: {}".format(score))
            cv2.imwrite(output_loc + "/%#05d.png" % (count+1), frame)

        prev_frame = copy.deepcopy(frame)
        count = count + 1
        # If there are no more frames left
        if (count > (video_length-1)):
            # Log the time again
            time_end = time.time()
            # Release the feed
            cap.release()
            # Print stats
            print ("Done extracting frames.\n%d frames extracted" % count)
            print ("It took %d seconds forconversion." % (time_end-time_start))
            break

if __name__=="__main__":

    # input_loc = '../data/20211117_C0012.MP4'
    # output_loc = '../data/20211117_C0012/'

    folder = '/run/media/ttsesm/external_data/repair_dataset/dataset@server/'
    # folder = '../data/box_8/'
    video_files = natsort.natsorted(glob.glob(folder + '**/*.{MP4,mp4}', flags=glob.BRACE | glob.GLOBSTAR))
    video_files = list(filter(lambda k: 'RPf' in k, video_files))

    for video_file in video_files:
        dirname, basename = os.path.split(video_file)
        basename_without_ext, ext = basename.split('.', 1)
        path_without_ext = os.path.join(dirname, basename_without_ext)

        dest_folder = os.path.join(dirname, basename_without_ext, "images/")

        os.makedirs(dest_folder, exist_ok=True)
        video_to_frames(video_file, dest_folder)
        # shutil.copy2(video_file, os.path.join(dirname, basename_without_ext))
        shutil.move(video_file, os.path.join(dirname, basename_without_ext))

    # d = collections.defaultdict(dict)
    # for filepath in mesh_files:
    #     keys = filepath.split("/")
    #     folder_ = keys[-2]
    #     file_ = re.match("(.*?)" + re.findall(r"\d+", keys[-1].split(".")[0])[0], keys[-1]).group()
    #     if folder_ in d:
    #         if file_ in d[folder_]:
    #             d[folder_][file_].append(filepath)
    #         else:
    #             d[folder_][file_] = [filepath]
    #     else:
    #         d[folder_][file_] = [filepath]

    print("End!!!")
    # video_to_frames(input_loc, output_loc)