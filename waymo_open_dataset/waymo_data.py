# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

''' Helper class and functions for loading SUN RGB-D objects

Author: Charles R. Qi
Date: December, 2018

Note: removed unused code for frustum preparation.
Changed a way for data visualization (removed depdency on mayavi).
Load depth with scipy.io
'''

import os
from os.path import split
import pickle
import sys
import numpy as np
import sys
import argparse

import plotly.graph_objects as go

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../utils/'))
import pc_util
import waymo_utils
from box_util import view_points

# Object types 
DEFAULT_TYPE_WHITELIST = ['TYPE_UNKNOWN','TYPE_VEHICLE','TYPE_PEDESTRIAN','TYPE_SIGN','TYPE_CYCLIST']



class waymo_object(object):
    ''' Load and parse object data '''
    def __init__(self, root_dir, split='training'):
        self.root_dir = root_dir
        self.split = split
        self.split_dir = os.path.join(BASE_DIR, root_dir, split)
        if not os.path.exists(self.split_dir):
            raise Exception('Make sure to run preprocessing function to prepare the data')
        # load idx2segment_id dictionary
        pickle_in = open(os.path.join(self.split_dir, 'idx2segment_id_dict'), 'rb')
        self.idx2segment_id = pickle.load(pickle_in)
        print(self.idx2segment_id)
        pickle_in.close()

        self.type2class = {'TYPE_UNKNOWN':0,'TYPE_VEHICLE':1,'TYPE_PEDESTRIAN':2,'TYPE_SIGN':3,'TYPE_CYCLIST':4}
        self.class2type = {self.type2class[t]:t for t in self.type2class}
        # get the count of the dataset
        if split == 'training':
            # TODO: set it later to the total number of training segments (or if you are going to use multiple frames from the same segment !)
            # assumption: we will use only the first frames from each segment to train on detection
            self.num_samples = len([file for file in os.listdir(os.path.join(root_dir, split)) if file.split('-')[0] == 'segment'])
            
        elif split == 'validation':
            self.num_samples = 150 # total number of validation segments

        elif split == 'testing':
            # TODO: change it later when you start to configure the testing
            self.num_samples = 1 # dummy value, set it later to the correct testing segments 
        else:
            raise Exception('Unknown split: {}'.format(split))

        print('No of samples are {} and No. of indices in segment dict is {}'.format(self.num_samples, len(self.idx2segment_id)))
        
        assert(self.num_samples == len(self.idx2segment_id)) #otherwise something wrong with data preprocessing

        # TODO: path check excpetions
        self.camera_images_dir = os.path.join(self.split_dir, 'camera_images')
        self.range_images_dir = os.path.join(self.split_dir, 'range_images')
        self.point_cloud_dir = os.path.join(self.split_dir, 'point_cloud')
        self.label_dir = os.path.join(self.split_dir, 'label')

    def __len__(self):
        return self.num_samples

    def get_camera_images(self, idx):
        # img_filename = os.path.join(self.image_dir, '%06d.jpg'%(idx))
        # return waymo_utils.load_images(img_filename)
        raise NotImplementedError("Camera images are not extracted. currently !!")

    def get_range_images(self, idx):
        # img_filename = os.path.join(self.image_dir, '%06d.jpg'%(idx))
        # return waymo_utils.load_images(img_filename)
        raise NotImplementedError("Range images are not extracted. currently !!")

    def get_point_cloud(self, idx):
        point_cloud_filename = os.path.join(self.point_cloud_dir, 'point_cloud_{}'.format(self.idx2segment_id[idx]))
        return waymo_utils.load_point_cloud(point_cloud_filename)
        
    def get_label_objects(self, idx):
        label_filename = os.path.join(self.label_dir, 'label_{}'.format(self.idx2segment_id[idx]))
        if not os.path.exists(label_filename):
            raise Exception("Couldn't find objects file for idx {}".format(idx))
        return waymo_utils.read_frame_bboxes(label_filename)

def data_viz(data_dir, idx: int = np.nan, verbose: bool = False):  
    ''' Visualize frame from Waymo data '''
    waymo_objects = waymo_object(data_dir)
    idx = int (idx) if not np.isnan(idx) else np.random.choice(np.range(len(waymo_objects)))
    if verbose: print('Visualizing frame with idx {}'.format(idx))
    pc = waymo_objects.get_point_cloud(idx)
    if verbose: print("No of points recorded in LIDAR return is {}".format(pc.shape))
    
    bboxes = waymo_objects.get_label_objects(idx)
    if verbose: print("No. of BBoxes for this frame is {}".format(len(bboxes)))

    pc_norm = np.sqrt(np.power(pc, 2).sum(axis=1))
    # add LIDAR return to the plot
    scatter = go.Scatter3d(
        x=pc[:,0],
        y=pc[:,1],
        z=pc[:,2],
        mode="markers",
        marker=dict(size=1, color=pc_norm, opacity=0.8),
    )

    label_colors = {0: 'cyan', 1: 'green', 2: 'orange', 3: 'red', 4: 'blue'}
    bboxes_lines = {0:{'x_lines':[], 'y_lines':[], 'z_lines':[], 'color': label_colors[0], 'type': waymo_objects.class2type[0]},
                1:{'x_lines':[], 'y_lines':[], 'z_lines':[], 'color': label_colors[1], 'type': waymo_objects.class2type[1]},
                2:{'x_lines':[], 'y_lines':[], 'z_lines':[], 'color': label_colors[2], 'type': waymo_objects.class2type[2]},
                3:{'x_lines':[], 'y_lines':[], 'z_lines':[], 'color': label_colors[3], 'type': waymo_objects.class2type[3]},
                4:{'x_lines':[], 'y_lines':[], 'z_lines':[], 'color': label_colors[4], 'type': waymo_objects.class2type[4]}}

    def f_lines_add_nones(label_type):
        bboxes_lines[label_type]['x_lines'].append(None)
        bboxes_lines[label_type]['y_lines'].append(None)
        bboxes_lines[label_type]['z_lines'].append(None)
    
    
    
    
    ixs_box_0 = [0, 1, 2, 3, 0]
    ixs_box_1 = [4, 5, 6, 7, 4]

    for bbox in bboxes:
        # get label type
        label_type = bbox.label 
        points = view_points(bbox.corners(), view=np.eye(3), normalize=False)
        bboxes_lines[label_type]['x_lines'].extend(points[0, ixs_box_0])
        bboxes_lines[label_type]['y_lines'].extend(points[1, ixs_box_0])
        bboxes_lines[label_type]['z_lines'].extend(points[2, ixs_box_0])
        f_lines_add_nones(label_type)
        bboxes_lines[label_type]['x_lines'].extend(points[0, ixs_box_1])
        bboxes_lines[label_type]['y_lines'].extend(points[1, ixs_box_1])
        bboxes_lines[label_type]['z_lines'].extend(points[2, ixs_box_1])
        f_lines_add_nones(label_type)
        for i in range(4):
            bboxes_lines[label_type]['x_lines'].extend(points[0, [ixs_box_0[i], ixs_box_1[i]]])
            bboxes_lines[label_type]['y_lines'].extend(points[1, [ixs_box_0[i], ixs_box_1[i]]])
            bboxes_lines[label_type]['z_lines'].extend(points[2, [ixs_box_0[i], ixs_box_1[i]]])
            f_lines_add_nones(label_type)


    #add lines to the plot
    all_lines = []
    for type_idx, bboxes_dict in bboxes_lines.items():
        if len(bboxes_dict['x_lines']) != 0:
            lines = go.Scatter3d(x=bboxes_dict['x_lines'], y=bboxes_dict['y_lines'], z=bboxes_dict['z_lines'], mode="lines", name=bboxes_dict['type'])
            all_lines.append(lines)

    fig = go.Figure(data=[scatter, *all_lines])
    fig.update_layout(scene_aspectmode="data")
    fig.show()

def extract_waymo_data(data_dir, split, output_folder, num_point=40000,
    type_whitelist=DEFAULT_TYPE_WHITELIST,
    save_votes=False, verbose: bool = False):
    """ Extract scene point clouds and 
    bounding boxes (centroids, box sizes, heading angles, semantic classes).
    Dumped point clouds and boxes are in upright depth coord.

    Args:
        split: training or testing
        save_votes: whether to compute and save Ground truth votes.
        use_v1: use the SUN RGB-D V1 data
        skip_empty_scene: if True, skip scenes that contain no object (no objet in whitelist)

    Dumps:
        <id>_pc.npz of (N,6) where N is for number of subsampled points and 6 is
            for XYZ and RGB (in 0~1) in upright depth coord
        <id>_bbox.npy of (K,8) where K is the number of objects, 8 is for
            centroids (cx,cy,cz), dimension (l,w,h), heanding_angle and semantic_class
        <id>_votes.npz of (N,10) with 0/1 indicating whether the point belongs to an object,
            then three sets of GT votes for up to three objects. If the point is only in one
            object's OBB, then the three GT votes are the same.
    """
    dataset = waymo_object(data_dir)
    if verbose: print("Length of the loaded dataset is {}".format(len(dataset)))

    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    for data_idx in range(len(dataset)):
        if verbose: print('Extracting information from index {}'.format(data_idx))
        objects = dataset.get_label_objects(data_idx)
        print("objects type", type(objects))
        if verbose: print("Number of loaded objects are {}".format(len(objects)))

        # Skip scenes with 0 object
        if (len(objects) == 0 or \
            len([obj for obj in objects if dataset.class2type[obj.label] in type_whitelist])== 0):
                print("+++++++++++++++++ Skipping Empty Scene, check that ++++++++++++++++")
                continue

        pc_upright_depth = dataset.get_point_cloud(data_idx)
        pc_upright_depth_subsampled = pc_util.random_sampling(pc_upright_depth, num_point)

        # save th subsampled point cloud
        np.savez_compressed(os.path.join(output_folder,'pc_{}.npz'.format(dataset.idx2segment_id[data_idx])),
            pc=pc_upright_depth_subsampled)
        # np.save(os.path.join(output_folder, '%06d_bbox.npy'%(data_idx)), obbs)
       
        if save_votes:
            N = pc_upright_depth_subsampled.shape[0]
            if verbose: print("No. of subsamples points are {}".format(N))
            point_votes = np.zeros((N,10)) # 3 votes and 1 vote mask 
            point_vote_idx = np.zeros((N)).astype(np.int32) # in the range of [0,2]
            indices = np.arange(N)
            for obj in objects:
                if dataset.class2type[obj.label] not in type_whitelist:
                    if verbose: print("Skipping this object, not in the white list")
                    continue
                try:
                    # Find all points in this object's OBB
                    box3d_pts_3d = np.transpose(obj.corners())
                    # if verbose: print(" 3d box data shape {}".format(box3d_pts_3d.shape))
                    # if verbose: print("Box coordinates of the current object are {}".format(box3d_pts_3d))
                    pc_in_box3d,inds = waymo_utils.extract_pc_in_box3d(\
                        pc_upright_depth_subsampled, box3d_pts_3d)
                    # if verbose: print("list of indices inside the box {}".format(inds))
                    # if verbose: print("No. of points inside the box are {}".format(len(pc_in_box3d)))
                    # Assign first dimension to indicate it is in an object box
                    point_votes[inds,0] = 1
                    # Add the votes (all 0 if the point is not in any object's OBB)
                    votes = np.expand_dims(obj.center,0) - pc_in_box3d[:,0:3]
                    sparse_inds = indices[inds] # turn dense True,False inds to sparse number-wise inds
                    for i in range(len(sparse_inds)):
                        j = sparse_inds[i]
                        point_votes[j, int(point_vote_idx[j]*3+1):int((point_vote_idx[j]+1)*3+1)] = votes[i,:]
                        # Populate votes with the fisrt vote
                        if point_vote_idx[j] == 0:
                            point_votes[j,4:7] = votes[i,:]
                            point_votes[j,7:10] = votes[i,:]
                    point_vote_idx[inds] = np.minimum(2, point_vote_idx[inds]+1)
                except Exception as e:
                    print(e)
                    # print('ERROR, idx {}, classlabel {} and not found in whitelist'.format(data_idx, obj.label))
                    raise
            # TODO: replace pickle with savez_compressed
            # with open(os.path.join(output_folder, 'votes_{}'.format(dataset.idx2segment_id[data_idx])), 'wb') as fp:
            #     if verbose: print("size of votes for index {} is {}".format(data_idx, point_votes.shape))
            #     pickle.dump(point_votes, fp) 
                
            np.savez_compressed(os.path.join(output_folder, 'votes_{}.npz'.format(dataset.idx2segment_id[data_idx])),
                point_votes = point_votes)

    
def get_box3d_dim_statistics(data_dir, type_whitelist=DEFAULT_TYPE_WHITELIST, save_path=None, verbose: bool = False):
    """ Collect 3D bounding box statistics.
    Used for computing mean box sizes. """
    dataset = waymo_object(data_dir)
    dimension_list = []
    type_list = []
    ry_list = []
    for data_idx in range(len(dataset)):
        if verbose: print('Collecting Box statistics from idx {} '.format(data_idx))
        bboxes = dataset.get_label_objects(data_idx)
        for obj_idx in range(len(bboxes)):
            obj = bboxes[obj_idx]
            print(obj.label)
            if dataset.class2type[obj.label] not in type_whitelist:
                print("++++++++++++++ WARNING: UNKNOWN BOX TYPE +++++++++++++++")
                continue
            dimension_list.append(obj.lwh) 
            type_list.append(obj.label) 
            ry_list.append(obj.heading_angle)


    # Get average box size for different catgories
    # box3d_pts = np.vstack(dimension_list)
    median_statistics = {}
    for class_type in sorted(set(type_list)):
        cnt = 0
        box3d_list = []
        for i in range(len(dimension_list)):
            if type_list[i]==class_type:
                cnt += 1
                box3d_list.append(dimension_list[i])
        median_box3d = np.median(box3d_list,0)
        print("\'{}\': np.array([{:f},{:f},{:f}]),".format(class_type, median_box3d[0], median_box3d[1], median_box3d[2]))
        median_statistics[class_type] = median_box3d[0], median_box3d[1], median_box3d[2]
    
    if save_path is not None:
        with open(save_path,'wb') as fp:
            pickle.dump(median_statistics, fp)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--preprocessing', action='store_true', help='Extract the data into readable foramt to be used for viz and training')
    parser.add_argument('--viz', action='store_true', help='Run data visualization.')
    parser.add_argument('--compute_median_size', action='store_true', help='Compute median 3D bounding box sizes for each class.')
    parser.add_argument('--extract_votes', action='store_true')
    args = parser.parse_args()

    print("check reachable point")
    if args.preprocessing:
        waymo_utils.preprocess_waymo_data('./dataset', 'training', args.verbose)
        exit()
    
    if args.viz:
        data_viz(os.path.join(BASE_DIR, 'dataset'), idx = 0, verbose = args.verbose)
        exit()

    if args.compute_median_size:
        get_box3d_dim_statistics(os.path.join(BASE_DIR, 'dataset'), verbose=args.verbose)
        exit()
    
    if(args.extract_votes):
        extract_waymo_data(os.path.join(BASE_DIR, 'dataset'),
        split = 'training', 
        output_folder= os.path.join(BASE_DIR, 'dataset', 'training', 'votes'),
        save_votes = True,
        num_point = 60000,
        verbose = args.verbose
        )
        exit()

