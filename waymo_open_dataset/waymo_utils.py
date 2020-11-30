''' Provides Python helper function to read Waymo Open Dataset dataset.

Author: Ahmed Bahnasy
Date: 

'''
import numpy as np
from pathlib import Path
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


def preprocess_waymo_data(dataset_dir, split='train', verbose: bool =False): #TODO: Obselete function should be handled inside the download scripts
    """ Function will read the TFRecords and extract data for every first first frame in every segment and save it as text file. the data could be easily loaded afterwards by the data class for every frame instead of loading the whole segment
    Args:
        root_dir: data split directory
        split: type of data to preprocess options:[training, val, test]
        verbose: flag to print debugging messages
    Returns:
    """
    # TODO: checks to see if the function have been already exectuted before and successfully extracted the data or not
    
    # list all the segments in the folder
    split_dir = os.path.join(BASE_DIR, dataset_dir, split)
    if not os.path.exists(split_dir):
        raise Exception("Path is not found")
    segments_list = os.listdir(split_dir)
    
    # list of dictionaries for every segment
    segments_dict_list = []
    
    # Loop over every segment in the dataset
    for idx in range(len(segments_list)):
        segment_dict = {}
        # get segment id
        segment_id = '_'.join(segments_list[idx].split('_')[:5]) # will get the ID example: 'segment-10072140764565668044_4060_000_4080_000'
        segment_dict['id'] = segment_id
        if verbose: print("processing segment id {}".format(segment_id))

        segment_dir = os.path.join(split_dir, segment_id)
        # create folder for the current segment
        Path(segment_dir).mkdir(parents=True, exist_ok=True)
        FILENAME = os.path.join(split_dir, segments_list[idx])
        if not os.path.exists(FILENAME):
            raise Exception("File cannot be found")
        # Read TFRecord
        recorded_segment = tf.data.TFRecordDataset(FILENAME, compression_type='')
        # Loop over every frame
        frame_count = 0
        for data in recorded_segment:
            # Read the first frame only
            frame = open_dataset.Frame()
            frame.ParseFromString(bytearray(data.numpy()))
            
            if verbose: print("processing frame no. {}".format(frame_count))
            
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
        # after every frame extracted, save the metadata for it    
        segment_dict['frame_count'] = frame_count
        segments_dict_list.append(segment_dict)

    # save segmetns dictioanry list on desk, this would be used later to count the size of the dataset and navigate through the dataset
    pickle_out = open(os.path.join(split_dir, 'segments_dict_list'), 'wb')
    pickle.dump(segments_dict_list, pickle_out)
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
    raise NotImplementedError("Not implemented !")
def load_range_images():
    raise NotImplementedError("Not implemented !")

def read_frame_bboxes(frame_data_path):
    ''' Return array of bounding boxes
    '''
    with np.load(frame_data_path) as frame_data:
        point_cloud = frame_data['labels']
    return point_cloud

def read_frame_bboxes_as_objects(label_file_name):
    bboxes = []
    pickle_in = open(label_file_name, 'rb')
    bboxes = pickle.load(pickle_in)
    pickle_in.close()
    print("Loaded file type is ", type(bboxes), "length of the list is ", len(bboxes))
    return bboxes


def load_point_cloud(point_cloud_filename):
    with np.load(point_cloud_filename) as frame_data:
        point_cloud = frame_data['pc']
    return point_cloud


def save_zipped_pickle(obj, filename, protocol=-1):
    with gzip.open(filename, 'wb') as f:
        pickle.dump(obj, f, protocol)

def load_zipped_pickle(filename):
    with gzip.open(filename, 'rb') as f:
        loaded_object = pickle.load(f)
        return loaded_object
