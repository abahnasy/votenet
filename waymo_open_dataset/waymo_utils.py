# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

''' Provides Python helper function to read My Waymo Open Dataset dataset.

Author: Ahmed Bahnasy
Date: 

'''
import numpy as np
import pickle
import gzip
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../utils/')) # import Box class

import tensorflow.compat.v1 as tf

import dataset_pb2 as open_dataset
import frame_utils
from box_util import Box



def rotx(t):
    """Rotation about the x-axis."""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[1,  0,  0],
                     [0,  c, -s],
                     [0,  s,  c]])


def roty(t):
    """Rotation about the y-axis."""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c,  0,  s],
                     [0,  1,  0],
                     [-s, 0,  c]])


def rotz(t):
    """Rotation about the z-axis."""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, -s,  0],
                     [s,  c,  0],
                     [0,  0,  1]])


def preprocess_waymo_data(root_dir, split='training', verbose: bool =False):
    """
    Function will read the TFRecords and extract data for every first first frame in every segment and save it as text file. the data could be easily loaded afterwards by the data class for every frame instead of loading the whole segment
    """
    # TODO: checks to see if the function have been already exectuted before and successfully extracted the data or not
    
    # list all the segments in the folder
    split_dir = os.path.join(BASE_DIR, root_dir, split)
    if not os.path.exists(split_dir):
        raise Exception("Path is not found")
    segments_list = os.listdir(split_dir)
    
    # create Dict idx to segment_id
    idx2segment_id = {}

    # create sub folders to extract the data
    CAMERA_IMAGES_DIR = os.path.join(split_dir, 'camera_images')
    if not os.path.exists(CAMERA_IMAGES_DIR):
        os.mkdir(CAMERA_IMAGES_DIR)
    RANGE_IMAGES_DIR = os.path.join(split_dir, 'range_images')
    if not os.path.exists(RANGE_IMAGES_DIR):
        os.mkdir(RANGE_IMAGES_DIR)
    POINT_CLOUD_DIR = os.path.join(split_dir, 'point_cloud')
    if not os.path.exists(POINT_CLOUD_DIR):
        os.mkdir(POINT_CLOUD_DIR)
    LABEL_DIR = os.path.join(split_dir, 'label')
    if not os.path.exists(LABEL_DIR):
        os.mkdir(LABEL_DIR)
    
    for idx in range(len(segments_list)):
        # get segment id
        segment_id = segments_list[idx].split('-')[1].split('_')[0] # will get the ID number
        if verbose:
            print("Reading Segment, index: {}, ID: {}".format(idx, segment_id))
        # add segment id with the respective idx to dict 
        idx2segment_id[idx] = segment_id
        print("Processing segment ID: {}".format(segment_id))
        # get full path of TFRecord file
        FILENAME = os.path.join(split_dir, segments_list[idx])
        if not os.path.exists(FILENAME):
            raise Exception("File cannot be found")
        # Read TFRecord
        recorded_segment = tf.data.TFRecordDataset(FILENAME, compression_type='')
        for data in recorded_segment:
            # Read the first frame only
            frame = open_dataset.Frame()
            frame.ParseFromString(bytearray(data.numpy()))
            break
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

        # save file on the desk
        pickle_out = open(os.path.join(POINT_CLOUD_DIR, 'point_cloud_{}'.format(segment_id)), 'wb')
        pickle.dump(points_all, pickle_out)
        pickle_out.close()
        pickle_out_ri2 = open(os.path.join(POINT_CLOUD_DIR, 'point_cloud_{}_ri2'.format(segment_id)), 'wb')
        pickle.dump(points_all_ri2, pickle_out_ri2)
        pickle_out_ri2.close()

        # extracting labels and save them
        bboxes = []
        for laser_label in frame.laser_labels:
            label = laser_label.type
            length = laser_label.box.length
            width = laser_label.box.width
            height = laser_label.box.height
            x, y, z = laser_label.box.center_x, laser_label.box.center_y, laser_label.box.center_z
            box = Box([x,y,z], [length, width, height], laser_label.box.heading, label)
            bboxes.append(box)

        # save the bounding boxes in their respective directory
        pickle_out = open(os.path.join(LABEL_DIR, 'label_{}'.format(segment_id)), 'wb')
        pickle.dump(bboxes, pickle_out)
        pickle_out.close()

    # save idx2segment_id dict on desk
    pickle_out = open(os.path.join(split_dir, 'idx2segment_id_dict'), 'wb')
    pickle.dump(idx2segment_id, pickle_out)
    pickle_out.close()


def in_hull(p, hull):
    from scipy.spatial import Delaunay
    if not isinstance(hull,Delaunay):
        hull = Delaunay(hull)
    return hull.find_simplex(p)>=0

def extract_pc_in_box3d(pc, box3d):
    ''' pc: (N,3), box3d: (8,3) '''
    box3d_roi_inds = in_hull(pc[:,0:3], box3d)
    return pc[box3d_roi_inds,:], box3d_roi_inds





def load_image(img_filename):
    pass
def load_range_images():
    pass

def read_frame_bboxes(label_file_name):
    bboxes = []
    pickle_in = open(label_file_name, 'rb')
    bboxes = pickle.load(pickle_in)
    pickle_in.close()
    print("Loaded file type is ", type(bboxes), "length of the list is ", len(bboxes))
    return bboxes

def load_point_cloud(point_cloud_filename):
    pickle_in = open(point_cloud_filename, 'rb')
    point_cloud = pickle.load(pickle_in)
    pickle_in.close()
    return point_cloud


def save_zipped_pickle(obj, filename, protocol=-1):
    with gzip.open(filename, 'wb') as f:
        pickle.dump(obj, f, protocol)

def load_zipped_pickle(filename):
    with gzip.open(filename, 'rb') as f:
        loaded_object = pickle.load(f)
        return loaded_object
