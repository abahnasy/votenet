''' Provides Python helper function to read Waymo Open Dataset dataset.

Author: Ahmed Bahnasy
Date: 

'''

import os
from os.path import split
import pickle
import sys
import numpy as np
import sys
import argparse
from pathlib import Path
#import plotly.graph_objects as go

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '..', 'utils'))
import pc_util
import waymo_utils
from box_util import view_points, get_corners_from_labels_array

# Object types 
DEFAULT_TYPE_WHITELIST = ['TYPE_UNKNOWN','TYPE_VEHICLE','TYPE_PEDESTRIAN','TYPE_SIGN','TYPE_CYCLIST']



class waymo_object(object):
    ''' Load and parse object data '''
    def __init__(self, root_dir, split='train', verbose:bool = False):
        self.root_dir = root_dir
        self.split = split
        self.split_dir = os.path.join(BASE_DIR, root_dir, split)
        if not os.path.exists(self.split_dir):
            raise Exception('Make sure to run preprocessing function to prepare the data')
        # load segments_dict_list dictionary
        segments_dict_list_path = os.path.join(self.split_dir, 'segments_dict_list')
        if not os.path.exists(segments_dict_list_path):
            raise ValueError('segments Dictionary list is not found, make sure to preprocess the data first')
        with open(segments_dict_list_path, 'rb') as f:
            self.segments_dict_list = pickle.load(f)
        self.num_segments = len(self.segments_dict_list)
        if verbose: print("No of segments in the dataset is {}".format(len(self.segments_dict_list)))



        self.type2class = {'TYPE_UNKNOWN':0,'TYPE_VEHICLE':1,'TYPE_PEDESTRIAN':2,'TYPE_SIGN':3,'TYPE_CYCLIST':4}
        self.class2type = {self.type2class[t]:t for t in self.type2class}
        # get the count of the dataset
        
        self.num_frames = 0
        for segment_dict in self.segments_dict_list:
            # add total number of frames in every segment
            self.num_frames += segment_dict['frame_count']

        print('No of frames are {} and No. of indices in segment dict is {}'.format(self.num_frames, len(self.segments_dict_list)))
        
        #TODO check no of folders equals no of segments and no of noz files in each folder equals no of frames for each segment
        # assert(self.num_frames == len(self.idx2segment_id)) #otherwise something wrong with data preprocessing

        # # TODO: path check excpetions
        # self.camera_images_dir = os.path.join(self.split_dir, 'camera_images')
        # self.range_images_dir = os.path.join(self.split_dir, 'range_images')
        # self.point_cloud_dir = os.path.join(self.split_dir, 'point_cloud')
        # self.label_dir = os.path.join(self.split_dir, 'label')

    def __len__(self):
        return self.num_frames
    def resolve_idx_to_frame(self, idx):
        ''' Get Global idx and transorm into segment frame idx
        '''
        frame_idx = idx
        for segment_dict in self.segments_dict_list:
            if frame_idx >= segment_dict['frame_count']:
                frame_idx -= segment_dict['frame_count']
            else:
                return (segment_dict['id'], frame_idx)


    def get_camera_images(self, idx):
        # img_filename = os.path.join(self.image_dir, '%06d.jpg'%(idx))
        # return waymo_utils.load_images(img_filename)
        raise NotImplementedError("Camera images are not extracted. currently !!")

    def get_range_images(self, idx):
        # img_filename = os.path.join(self.image_dir, '%06d.jpg'%(idx))
        # return waymo_utils.load_images(img_filename)
        raise NotImplementedError("Range images are not extracted. currently !!")

    def get_point_cloud(self, idx):
        segment_id, frame_id = self.resolve_idx_to_frame(idx)
        frame_data_path = os.path.join(self.split_dir, segment_id, '{}_{}.npz'.format(segment_id, frame_id))
        if not os.path.exists(frame_data_path):
            raise ValueError('frame data is not found !')
        return waymo_utils.load_point_cloud(frame_data_path)
        
    def get_labels(self, idx):
        segment_id, frame_id = self.resolve_idx_to_frame(idx)
        frame_data_path = os.path.join(self.split_dir, segment_id, '{}_{}.npz'.format(segment_id, frame_id))
        if not os.path.exists(frame_data_path):
            print(frame_data_path)
            raise ValueError('frame data is not found !')
        return waymo_utils.read_frame_bboxes(frame_data_path)

def data_viz(data_dir, idx: int = np.nan, verbose: bool = False):  
    ''' Visualize frame from Waymo data '''
    raise NotImplementedError("Not implemented !")
    waymo_objects = waymo_object(data_dir)
    idx = int (idx) if not np.isnan(idx) else np.random.choice(np.range(len(waymo_objects)))
    if verbose: print('Visualizing frame with idx {}'.format(idx))
    pc = waymo_objects.get_point_cloud(idx)
    if verbose: print("No of points recorded in LIDAR return is {}".format(pc.shape))
    
    bboxes = waymo_objects.get_labels(idx)
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

def extract_waymo_data(data_dir, split, output_folder, num_point=126000,
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
    dataset = waymo_object(data_dir, split)
    if verbose: print("Length of the loaded dataset is {}".format(len(dataset)))

    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    for data_idx in range(len(dataset)):
        # Get segment id and frame id
        segment_id, frame_id = dataset.resolve_idx_to_frame(data_idx)
        if verbose: print('Extracting information from index {}'.format(data_idx))
        objects = dataset.get_labels(data_idx)
        print("objects type", type(objects))
        if verbose: print("Number of loaded objects for idx {} are {}".format(data_idx, objects.shape[0]))

        # Skip scenes with 0 object
        if (objects.shape[0] == 0):
                print("+++++++++++++++++ Skipping Empty Scene, check that ++++++++++++++++")
                continue

        pc_upright_depth = dataset.get_point_cloud(data_idx)
        pc_upright_depth_subsampled = pc_util.random_sampling(pc_upright_depth, num_point)

        # save th subsampled point cloud
        segment_votes_path = os.path.join(output_folder, segment_id)
        # create path if not found
        Path(segment_votes_path).mkdir(parents=True, exist_ok=True)
        
        np.savez_compressed(os.path.join(segment_votes_path,'{}_{}_pc.npz'.format(segment_id, frame_id)),
            pc=pc_upright_depth_subsampled)
        # np.save(os.path.join(output_folder, '%06d_bbox.npy'%(data_idx)), obbs)
       
        if save_votes:
            N = pc_upright_depth_subsampled.shape[0]
            if verbose: print("No. of subsamples points are {}".format(N))
            point_votes = np.zeros((N,10)) # 3 votes and 1 vote mask 
            point_vote_idx = np.zeros((N)).astype(np.int32) # in the range of [0,2]
            indices = np.arange(N)
            
            for i in range(objects.shape[0]):
                if dataset.class2type[objects[i][0]] not in type_whitelist:
                    if verbose: print("Skipping this object, not in the white list")
                    continue
                try:
                    # Find all points in this object's OBB
                    box3d_pts_3d = np.transpose(get_corners_from_labels_array(objects[i,:]))
                    # if verbose: print(" 3d box data shape {}".format(box3d_pts_3d.shape))
                    # if verbose: print("Box coordinates of the current object are {}".format(box3d_pts_3d))
                    pc_in_box3d,inds = waymo_utils.extract_pc_in_box3d(\
                        pc_upright_depth_subsampled, box3d_pts_3d)
                    # if verbose: print("list of indices inside the box {}".format(inds))
                    # if verbose: print("No. of points inside the box are {}".format(len(pc_in_box3d)))
                    # Assign first dimension to indicate it is in an object box
                    point_votes[inds,0] = 1
                    # Add the votes (all 0 if the point is not in any object's OBB)
                    votes = np.expand_dims(objects[i,4:7],0) - pc_in_box3d[:,0:3]
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
                
            np.savez_compressed(os.path.join(segment_votes_path, '{}_{}_votes.npz'.format(segment_id, frame_id)),
                point_votes = point_votes)

    
def get_box3d_dim_statistics(data_dir, type_whitelist=DEFAULT_TYPE_WHITELIST, save_path=None, verbose: bool = False):
    """ Collect 3D bounding box statistics.
    Used for computing mean box sizes. """
    dataset = waymo_object(data_dir)
    dimension_list = []
    # type_list = []
    # ry_list = []
    for data_idx in range(len(dataset)):
        
        bboxes = dataset.get_labels(data_idx)
        dimension_list.append(bboxes[:,0:4])
    
    all_boxes = np.concatenate(dimension_list, axis = 0)
    median_statistics = {}
    
    for class_type in range(5):
        mask = (all_boxes[:,0] == class_type)
        class_boxes = all_boxes[mask]
        if class_boxes.shape[0] == 0:
            continue
        class_boxes = class_boxes[:,1:]
        median_statistics[class_type] = np.median(class_boxes, axis=0)
        print("\'{}\': np.array([{:f},{:f},{:f}]),".format(class_type, median_statistics[class_type][0], median_statistics[class_type][1], median_statistics[class_type][2]))

    if save_path is not None:
        with open(save_path,'wb') as fp:
            pickle.dump(median_statistics, fp)



if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--preprocessing-train', action='store_true', help='Extract the data into readable foramt to be used for viz and training')
    parser.add_argument('--preprocessing-val', action='store_true', help='Extract the data into readable foramt to be used for viz and training')
    parser.add_argument('--viz', action='store_true', help='Run data visualization.')
    parser.add_argument('--compute_median_size', action='store_true', help='Compute median 3D bounding box sizes for each class.')
    parser.add_argument('--extract-votes-train', action='store_true')
    parser.add_argument('--extract-votes-val', action='store_true')
    parser.add_argument('--num_point', type=int, default=60000, help='Point Number [default: 60000]')
    args = parser.parse_args()

    # step 1
    if args.preprocessing_train:
        waymo_utils.preprocess_waymo_data('./dataset', 'train', args.verbose)
        exit()

    if args.preprocessing_val:
        waymo_utils.preprocess_waymo_data('./dataset', 'val', args.verbose)
        exit()
    # step 2
    if(args.extract_votes_train): # extract votes for both splits train and validation
        extract_waymo_data(os.path.join(BASE_DIR, 'dataset'),
        split = 'train', 
        output_folder= os.path.join(BASE_DIR, 'dataset', 'train', 'votes'),
        save_votes = True,
        num_point = args.num_point,
        verbose = args.verbose
        )
        exit()
    if(args.extract_votes_val): # extract votes for both splits train and validation
        extract_waymo_data(os.path.join(BASE_DIR, 'dataset'),
        split = 'val', 
        output_folder= os.path.join(BASE_DIR, 'dataset', 'val', 'votes'),
        save_votes = True,
        num_point = args.num_point,
        verbose = args.verbose
        )
        exit()
    # step 3
    if args.compute_median_size:
        get_box3d_dim_statistics(os.path.join(BASE_DIR, 'dataset'), verbose=args.verbose)
        exit()
    
    if args.viz:
        data_viz(os.path.join(BASE_DIR, 'dataset'), idx = 0, verbose = args.verbose)
        exit()

    
    
    

