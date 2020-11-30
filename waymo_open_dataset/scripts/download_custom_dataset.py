'''
Ahmed Bahnasy
script to download the dataset in form of single frames to train the detector
choose the no of frames to be extracted from every segment
this script assumes that it will redownload data from scratch
Arguments: datasplit / no.of frames to be extracted
'''

import os
from pathlib import Path
import subprocess
import argparse
import sys
import numpy as np
import pickle

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# sys.path.append(BASE_DIR)

# sys.path.append(os.path.join(BASE_DIR, '..' , '..', 'utils')) # import Box class

import tensorflow.compat.v1 as tf
sys.path.append(os.path.join(BASE_DIR, '..'))
import dataset_pb2 as open_dataset
import frame_utils

parser = argparse.ArgumentParser()
parser.add_argument('--verbose', action='store_true')
parser.add_argument('type', choices=['training', 'validation', 'testing'])
parser.add_argument('--data-dir', default='dataset')
parser.add_argument('--num-segments', type=int, default = 0, help="Number of downloaded segments")
parser.add_argument('--num-frames', type=int, default = 0, help="Number of frames to be extracted from the segments")
args = parser.parse_args()
print("Python Version is {}.{}.{}".format(sys.version_info[0], sys.version_info[1], sys.version_info[2]))
if sys.version_info[1] <= 6:
    from subprocess import PIPE
    p1 = subprocess.run("gsutil ls -r gs://waymo_open_dataset_v_1_2_0_individual_files/{type}".format(type=args.type), shell=True, stdout=PIPE, stderr=PIPE, universal_newlines=True)
else:
    p1 = subprocess.run("gsutil ls -r gs://waymo_open_dataset_v_1_2_0_individual_files/{type}".format(type=args.type), shell=True, capture_output=True, text=True)
if p1.returncode != 0:
    raise Exception("gsutil error, check that gsutil is installed and configured correctly according to README file !", p1.stderr)
download_list = p1.stdout.split()

print("Creating/accessing dataset folder...")

dir_name_dict = {'training':'train', 'validation': 'val', 'testing':'test'}

DATA_DIR = os.path.join(BASE_DIR, '..', args.data_dir, dir_name_dict[args.type])
# create the main dataset and data split directories if not found
Path(DATA_DIR).mkdir(parents=True, exist_ok=True)
# get the list of already downloaded segments if there is any !
# get tensorflow record files


# list of dictionaries for every segment
segments_dict_list = []


# set number of segmetns to download
no_downloads = args.num_segments+1 if args.num_segments != 0 else len(download_list)
#
for i in range(1, no_downloads):
    segment_name = download_list[i].split('/{}/'.format(args.type))[-1]
    
    
    print("Downloading segment {}".format(i))
    p2 = subprocess.run(" ".join(["gsutil cp", download_list[i], DATA_DIR]), shell=True)

    segment_dict = {}
    # extract the no. of frames needed
    segment_id = '_'.join(segment_name.split('_')[:5]) # will get the ID example: 'segment-10072140764565668044_4060_000_4080_000'
    segment_dict['id'] = segment_id
    # create folder for the current segment
    segment_dir = os.path.join(DATA_DIR, segment_id)
    Path(segment_dir).mkdir(parents=True, exist_ok=True)
    
    segment_path = os.path.join(DATA_DIR, segment_name)
    print(segment_path)
    if not os.path.exists(segment_path):
        raise ValueError("Path doesn't exist !")
    # Read TFRecord
    recorded_segment = tf.data.TFRecordDataset(segment_path, compression_type='')
    # Loop over every frame
    frame_count = 0
    for data in recorded_segment:
        # Read the first frame only
        frame = open_dataset.Frame()
        frame.ParseFromString(bytearray(data.numpy()))
        
        if args.verbose: print("processing frame no. {}".format(frame_count))
        
        # extract the camera images, camera projection points and range images
        (range_images, 
        camera_projections, 
        range_image_top_pose) = frame_utils.parse_range_image_and_camera_projection(frame)

        # First return of Lidar data
        points, cp_points = frame_utils.convert_range_image_to_point_cloud(
            frame,
            range_images,
            camera_projections,
            range_image_top_pose)

        # Second return of Lidar data
        points_ri2, cp_points_ri2 = frame_utils.convert_range_image_to_point_cloud(
            frame,
            range_images,
            camera_projections,
            range_image_top_pose,
            ri_index=1)

        # concatenate all LIDAR points from the 5 radars.
        points_all = np.concatenate(points, axis=0)
        points_all_ri2 = np.concatenate(points_ri2, axis=0)

        bboxes = []
        for laser_label in frame.laser_labels:
            label = laser_label.type
            length = laser_label.box.length
            width = laser_label.box.width
            height = laser_label.box.height
            x, y, z = laser_label.box.center_x, laser_label.box.center_y, laser_label.box.center_z
            heading = laser_label.box.heading
            box = [label, length, width, height, x, y, z, heading]
            bboxes.append(box)

        labels_arr = np.array(bboxes, dtype=np.float32)
        file_name = '_'.join([segment_id, str(frame_count)])
        np.savez_compressed(os.path.join(segment_dir, '{}.npz'.format(file_name)),pc=points_all, pc_ri2 = points_all_ri2, labels=labels_arr)
        
    
        frame_count += 1
        if frame_count == args.num_frames:
            break
    # after every frame extracted, save the metadata for it    
    segment_dict['frame_count'] = frame_count
    segments_dict_list.append(segment_dict)
    # TODO: delete the original tfrecord file to save space

# save segmetns dictioanry list on desk, this would be used later to count the size of the dataset and navigate through the dataset
pickle_out = open(os.path.join(DATA_DIR, 'segments_dict_list'), 'wb')
pickle.dump(segments_dict_list, pickle_out)
pickle_out.close()


    
    
    


    
    

      
