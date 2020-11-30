# Adopted From: https://github.com/facebookresearch/votenet/

""" Training routine for Waymo dataset

Sample usage:
python3 train.py --model=votenet --dataset=waymo --log_dir=log --dump_dir=dump --num_point=100000 --max_epoch=100 --batch_size=1 --overwrite --dump_results --verbose

To use Tensorboard:
At server:
    python -m tensorboard.main --logdir=<log_dir_name> --port=6006
At local machine:
    ssh -L 1237:localhost:6006 <server_name>
Then go to local browser and type:
    localhost:1237
"""

import os
import sys
import numpy as np
from datetime import datetime
import argparse
import importlib

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'pointnet2'))
sys.path.append(os.path.join(ROOT_DIR, 'models'))
from pytorch_utils import BNMomentumScheduler
from tf_visualizer import Visualizer as TfVisualizer
from ap_helper import APCalculator, parse_predictions, parse_groundtruths

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='votenet', help='Model file name [default: votenet]')
parser.add_argument('--dataset', default='waymo', help='Dataset name. [default: waymo]')
parser.add_argument('--checkpoint_path', default=None, help='Model checkpoint path [default: None]')
parser.add_argument('--log_dir', default='log', help='Dump dir to save model checkpoint [default: log]')
parser.add_argument('--dump_dir', default=None, help='Dump dir to save sample outputs [default: None]')
parser.add_argument('--num_point', type=int, default=60000, help='Point Number [default: 60000]')
parser.add_argument('--num_target', type=int, default=128, help='Proposal number [default: 128]')
parser.add_argument('--vote_factor', type=int, default=1, help='Vote factor [default: 1]')
parser.add_argument('--cluster_sampling', default='vote_fps', help='Sampling strategy for vote clusters: vote_fps, seed_fps, random [default: vote_fps]')
parser.add_argument('--ap_iou_thresh', type=float, default=0.25, help='AP IoU threshold [default: 0.25]')
parser.add_argument('--max_epoch', type=int, default=180, help='Epoch to run [default: 180]')
parser.add_argument('--batch_size', type=int, default=8, help='Batch Size during training [default: 8]')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--weight_decay', type=float, default=0, help='Optimization L2 weight decay [default: 0]')
parser.add_argument('--bn_decay_step', type=int, default=20, help='Period of BN decay (in epochs) [default: 20]')
parser.add_argument('--bn_decay_rate', type=float, default=0.5, help='Decay rate for BN decay [default: 0.5]')
parser.add_argument('--lr_decay_steps', default='65,500,500', help='When to decay the learning rate (in epochs) [default: 80,120,160]')
parser.add_argument('--lr_decay_rates', default='0.1,0.1,0.1', help='Decay rates for lr decay [default: 0.1,0.1,0.1]')
parser.add_argument('--no_height', action='store_true', help='Do NOT use height signal in input.')
parser.add_argument('--overwrite', action='store_true', help='Overwrite existing log and dump folders.')
parser.add_argument('--dump_results', action='store_true', help='Dump results.')
parser.add_argument('--verbose', action='store_true', help='Print debugging messages')

parser.add_argument('--load_saved_model', action='store_true')
FLAGS = parser.parse_args()

# ------------------------------------------------------------------------- GLOBAL CONFIG BEG
BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
MAX_EPOCH = FLAGS.max_epoch
BASE_LEARNING_RATE = FLAGS.learning_rate
BN_DECAY_STEP = FLAGS.bn_decay_step
BN_DECAY_RATE = FLAGS.bn_decay_rate
LR_DECAY_STEPS = [int(x) for x in FLAGS.lr_decay_steps.split(',')]
LR_DECAY_RATES = [float(x) for x in FLAGS.lr_decay_rates.split(',')]
assert(len(LR_DECAY_STEPS)==len(LR_DECAY_RATES))
LOG_DIR = FLAGS.log_dir
DEFAULT_DUMP_DIR = os.path.join(BASE_DIR, os.path.basename(LOG_DIR))
DUMP_DIR = FLAGS.dump_dir if FLAGS.dump_dir is not None else DEFAULT_DUMP_DIR
DEFAULT_CHECKPOINT_PATH = os.path.join(LOG_DIR, 'checkpoint.tar')
CHECKPOINT_PATH = FLAGS.checkpoint_path if FLAGS.checkpoint_path is not None \
    else DEFAULT_CHECKPOINT_PATH
FLAGS.DUMP_DIR = DUMP_DIR

# Prepare LOG_DIR and DUMP_DIR
if os.path.exists(LOG_DIR) and FLAGS.overwrite:
    print('Log folder %s already exists. Are you sure to overwrite? (Y/N)'%(LOG_DIR))
    c = input()
    if c == 'n' or c == 'N':
        print('Exiting..')
        exit()
    elif c == 'y' or c == 'Y':
        print('Overwrite the files in the log and dump folers...')
        os.system('rm -r %s %s'%(LOG_DIR, DUMP_DIR))

if not os.path.exists(LOG_DIR):
    os.mkdir(LOG_DIR)

LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'a')
LOG_FOUT.write(str(FLAGS)+'\n')
def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)
if not os.path.exists(DUMP_DIR): os.mkdir(DUMP_DIR)

# Init datasets and dataloaders 
def my_worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)

# Create Dataset and Dataloader
if FLAGS.dataset == 'waymo':
    sys.path.append(os.path.join(ROOT_DIR, 'waymo_open_dataset'))
    from waymo_detection_dataset import WaymoDetectionVotesDataset, MAX_NUM_OBJ
    from model_util_waymo import WaymoDatasetConfig
    DATASET_CONFIG = WaymoDatasetConfig()
    TRAIN_DATASET = WaymoDetectionVotesDataset('train', num_points=NUM_POINT,
        use_height = (not FLAGS.no_height), augment=False)
    VAL_DATASET = WaymoDetectionVotesDataset('val', num_points=NUM_POINT,
        use_height = (not FLAGS.no_height), augment=False)

else:
    print('Unknown dataset %s. Exiting...'%(FLAGS.dataset))
    exit(-1)
# print(len(TRAIN_DATASET), len(VAL_DATASET))
print("Length of train dataset is ", len(TRAIN_DATASET))
TRAIN_DATALOADER = DataLoader(TRAIN_DATASET, batch_size=BATCH_SIZE,
    shuffle=True, num_workers=4, worker_init_fn=my_worker_init_fn)
VAL_DATALOADER = DataLoader(VAL_DATASET, batch_size=BATCH_SIZE,
    shuffle=True, num_workers=4, worker_init_fn=my_worker_init_fn)
# print(len(TRAIN_DATALOADER), len(VAL_DATALOADER))
print("Length of val dataset is ", len(VAL_DATASET))

# Init the model and optimzier
MODEL = importlib.import_module(FLAGS.model) # import network module
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
num_input_channel = int(not FLAGS.no_height)*1

if FLAGS.model == 'boxnet':
    Detector = MODEL.BoxNet
else:
    Detector = MODEL.VoteNet

net = Detector(num_class=DATASET_CONFIG.num_class,
               num_heading_bin=DATASET_CONFIG.num_heading_bin,
               num_size_cluster=DATASET_CONFIG.num_size_cluster,
               mean_size_arr=DATASET_CONFIG.mean_size_arr,
               num_proposal=FLAGS.num_target,
               input_feature_dim=num_input_channel,
               vote_factor=FLAGS.vote_factor,
               sampling=FLAGS.cluster_sampling)

# if torch.cuda.device_count() > 1:
#   log_string("Let's use %d GPUs!" % (torch.cuda.device_count()))
#   # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
#   net = nn.DataParallel(net)

net.to(device)
criterion = MODEL.get_loss

# Load the Adam optimizer
optimizer = optim.Adam(net.parameters(), lr=BASE_LEARNING_RATE, weight_decay=FLAGS.weight_decay)

# Load checkpoint if there is any
it = -1 # for the initialize value of `LambdaLR` and `BNMomentumScheduler`
start_epoch = 0
if CHECKPOINT_PATH is not None and os.path.isfile(CHECKPOINT_PATH):
    checkpoint = torch.load(CHECKPOINT_PATH)
    net.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    log_string("-> loaded checkpoint %s (epoch: %d)"%(CHECKPOINT_PATH, start_epoch))

# Decay Batchnorm momentum from 0.5 to 0.999
# note: pytorch's BN momentum (default 0.1)= 1 - tensorflow's BN momentum
BN_MOMENTUM_INIT = 0.5
BN_MOMENTUM_MAX = 0.001
bn_lbmd = lambda it: max(BN_MOMENTUM_INIT * BN_DECAY_RATE**(int(it / BN_DECAY_STEP)), BN_MOMENTUM_MAX)
bnm_scheduler = BNMomentumScheduler(net, bn_lambda=bn_lbmd, last_epoch=start_epoch-1)

def get_current_lr(epoch):
    lr = BASE_LEARNING_RATE
    for i,lr_decay_epoch in enumerate(LR_DECAY_STEPS):
        if epoch >= lr_decay_epoch:
            lr *= LR_DECAY_RATES[i]
    return lr

def adjust_learning_rate(optimizer, epoch):
    lr = get_current_lr(epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# TFBoard Visualizers
TRAIN_VISUALIZER = TfVisualizer(FLAGS, 'train')
VAL_VISUALIZER = TfVisualizer(FLAGS, 'val')


# Used for AP calculation
CONFIG_DICT = {'remove_empty_box':False, 'use_3d_nms':True,
    'nms_iou':0.25, 'use_old_type_nms':False, 'cls_nms':True,
    'per_class_proposal': False, 'conf_thresh':0.05,
    'dataset_config':DATASET_CONFIG}

# ------------------------------------------------------------------------- GLOBAL CONFIG END

def train_one_epoch():
    stat_dict = {} # collect statistics
    adjust_learning_rate(optimizer, EPOCH_CNT)
    bnm_scheduler.step() # decay BN momentum
    net.train() # set model to training mode
    # print("\nGPU Memory usage before adding the inputs from data loader is {}".format(torch.cuda.memory_allocated(0)))
    for batch_idx, batch_data_label in enumerate(TRAIN_DATALOADER):
        print("Training Batch {}, Epoch {}".format(batch_idx, EPOCH_CNT))
        # if FLAGS.verbose: print("\nmoving input data dictionary to device")
        for key in batch_data_label:
            # if FLAGS.verbose: print("key: {}, dimensions: {}".format(key, batch_data_label[key].shape))
            batch_data_label[key] = batch_data_label[key].to(device)
        # print("\nGPU Memory usage after adding the inputs from data loader is {}".format(torch.cuda.memory_allocated(0)))
        

        # Forward pass
        optimizer.zero_grad()
        inputs = {'point_clouds': batch_data_label['point_clouds']}
        # if FLAGS.verbose: print("\n setting inputs dictionary with point clouds and feed it to the model, point cloud dimensions are {}".format(inputs['point_clouds'].shape))
        end_points = net(inputs)


        # if FLAGS.verbose:
            # print("contents of the end_points dictionary (return of the model) are ")
            # for key, label in end_points.items():
                # print("{}, dimensions: {}".format(key, end_points[key].shape))
            # print("end of end_points keys !")
        
        # if FLAGS.verbose: print("\n Now computing the loss, loop over keys in batch data and add it to end_points dictionary")
        # Compute loss and gradients, update parameters.
        for key in batch_data_label:
            assert(key not in end_points)
            # if FLAGS.verbose: print("{}, size is {}".format(key, batch_data_label[key].shape))
            end_points[key] = batch_data_label[key]
        loss, end_points = criterion(end_points, DATASET_CONFIG)
        loss.backward()
        # print("GPU Memory usage after calling backward function {}".format(torch.cuda.memory_allocated(0)))
        optimizer.step()
        # print("GPU Memory usage after calling optimizer step function {}".format(torch.cuda.memory_allocated(0)))

        # Accumulate statistics and print out
        for key in end_points:
            if 'loss' in key or 'acc' in key or 'ratio' in key:
                if key not in stat_dict: stat_dict[key] = 0
                stat_dict[key] += end_points[key].item()

        
        # ==== Temp implementation for visual Debugging === #
        # batch_pred_map_cls = parse_predictions(end_points, CONFIG_DICT) 
        # batch_gt_map_cls = parse_groundtruths(end_points, CONFIG_DICT) 
        # # Extract prediction for visialization
        # import pickle 
        # with open("./visualizations", 'wb') as fp:
        #     dictie = {}
        #     dictie['point_cloud'] = batch_data_label['point_clouds'].detach().cpu().numpy()
        #     dictie['sa1_xyz'] = end_points['sa1_xyz'].detach().cpu().numpy()
        #     dictie['sa2_xyz'] = end_points['sa2_xyz'].detach().cpu().numpy()
        #     dictie['sa3_xyz'] = end_points['sa3_xyz'].detach().cpu().numpy()
        #     dictie['sa4_xyz'] = end_points['sa4_xyz'].detach().cpu().numpy()
        #     dictie['parsed_gt'] = batch_gt_map_cls
        #     dictie['parsed_predictions'] = batch_pred_map_cls
        #     pickle.dump(dictie, fp)
        
        # ==== Temp implementation === #
    #     TRAIN_VISUALIZER.log_scalars({key:stat_dict[key]/float(batch_idx+1) for key in stat_dict},
    #     (EPOCH_CNT*len(TRAIN_DATALOADER)+batch_idx)*BATCH_SIZE)
            # for key in sorted(stat_dict.keys()):
            #     log_string('mean %s: %f'%(key, stat_dict[key]/float(batch_idx+1)))
            #     stat_dict[key] = 0    
        
        
   # Epoch Logging
    TRAIN_VISUALIZER.log_scalars({key:stat_dict[key]/float(batch_idx+1) for key in stat_dict},
        (EPOCH_CNT))
    for key in sorted(stat_dict.keys()):
        log_string('mean %s: %f'%(key, stat_dict[key]/float(batch_idx+1)))
        stat_dict[key] = 0

def evaluate_one_epoch():
    stat_dict = {} # collect statistics
    ap_calculator = APCalculator(ap_iou_thresh=FLAGS.ap_iou_thresh,
        class2type_map=DATASET_CONFIG.class2type)
    net.eval() # set model to eval mode (for bn and dp)
    for batch_idx, batch_data_label in enumerate(VAL_DATALOADER):
        if batch_idx % 10 == 0:
            print('Eval batch: %d'%(batch_idx))
        for key in batch_data_label:
            batch_data_label[key] = batch_data_label[key].to(device)
        
        # Forward pass
        inputs = {'point_clouds': batch_data_label['point_clouds']}
        with torch.no_grad():
            end_points = net(inputs)

        # Compute loss
        for key in batch_data_label:
            assert(key not in end_points)
            end_points[key] = batch_data_label[key]
        loss, end_points = criterion(end_points, DATASET_CONFIG)

        # Accumulate statistics and print out
        for key in end_points:
            if 'loss' in key or 'acc' in key or 'ratio' in key:
                if key not in stat_dict: stat_dict[key] = 0
                stat_dict[key] += end_points[key].item()

        batch_pred_map_cls = parse_predictions(end_points, CONFIG_DICT) 
        batch_gt_map_cls = parse_groundtruths(end_points, CONFIG_DICT)
        ap_calculator.step(batch_pred_map_cls, batch_gt_map_cls)

        # Dump evaluation results for visualization
        if FLAGS.dump_results and batch_idx == 0 and EPOCH_CNT %10 == 0:
            MODEL.dump_results(end_points, DUMP_DIR, DATASET_CONFIG) 

    # Log statistics
    VAL_VISUALIZER.log_scalars({key:stat_dict[key]/float(batch_idx+1) for key in stat_dict},
        (EPOCH_CNT+1)*len(TRAIN_DATALOADER)*BATCH_SIZE)
    for key in sorted(stat_dict.keys()):
        log_string('eval mean %s: %f'%(key, stat_dict[key]/(float(batch_idx+1))))

    # Evaluate average precision
    metrics_dict = ap_calculator.compute_metrics()
    for key in metrics_dict:
        log_string('eval %s: %f'%(key, metrics_dict[key]))

    mean_loss = stat_dict['loss']/float(batch_idx+1)
    return mean_loss

def evaluate_overfit_run():
    """
    temp function to test overfit over couple of frames. test network functionality
    """
    stat_dict = {} # collect statistics
    # ap_calculator = APCalculator(ap_iou_thresh=FLAGS.ap_iou_thresh,
        # class2type_map=DATASET_CONFIG.class2type)
    net.eval() # set model to eval mode (for bn and dp)
    for batch_idx, batch_data_label in enumerate(VAL_DATALOADER):
        
        print('Eval batch: %d'%(batch_idx))
        for key in batch_data_label:
            batch_data_label[key] = batch_data_label[key].to(device)
        
        # Forward pass
        inputs = {'point_clouds': batch_data_label['point_clouds']}
        with torch.no_grad():
            end_points = net(inputs)

        # Compute loss
        for key in batch_data_label:
            assert(key not in end_points)
            end_points[key] = batch_data_label[key]
        loss, end_points = criterion(end_points, DATASET_CONFIG)

        # Accumulate statistics and print out
        for key in end_points:
            if 'loss' in key or 'acc' in key or 'ratio' in key:
                if key not in stat_dict: stat_dict[key] = 0
                stat_dict[key] += end_points[key].item()

        
        batch_gt_map_cls = parse_groundtruths(end_points, CONFIG_DICT)
        batch_pred_map_cls = parse_predictions(end_points, CONFIG_DICT) 
        for i in range(len(batch_gt_map_cls)):
            print("number of ground Truth in frame {} is: {}".format(i, len(batch_gt_map_cls[i])))
            print("number of predictions in frame {} is: {}".format(i, len(batch_pred_map_cls[i])))

        # Extract prediction for visialization
        import pickle 
        print("exporting visualizations file")
        with open("./saved_viz/visualizations_{}".format(EPOCH_CNT), 'wb') as fp:
            dictie = {}
            # pick only the first frame in the batch
            dictie['point_cloud'] = batch_data_label['point_clouds'][0].detach().cpu().numpy()
            dictie['parsed_gt'] = batch_gt_map_cls[0]
            dictie['parsed_predictions'] = batch_pred_map_cls[0]
            dictie['sa1_xyz'] = end_points['sa1_xyz'][0].detach().cpu().numpy()
            dictie['sa2_xyz'] = end_points['sa2_xyz'][0].detach().cpu().numpy()
            dictie['sa3_xyz'] = end_points['sa3_xyz'][0].detach().cpu().numpy()
            dictie['sa4_xyz'] = end_points['sa4_xyz'][0].detach().cpu().numpy()
            dictie['fp2_xyz'] = end_points['fp2_xyz'][0].detach().cpu().numpy()
            pickle.dump(dictie, fp)

        
        
        # ap_calculator.step(batch_pred_map_cls, batch_gt_map_cls)

        # Dump evaluation results for visualization
        # if FLAGS.dump_results and batch_idx == 0 and EPOCH_CNT %10 == 0:
        #     MODEL.dump_results(end_points, DUMP_DIR, DATASET_CONFIG) 

    # Log statistics
    VAL_VISUALIZER.log_scalars({key:stat_dict[key]/float(batch_idx+1) for key in stat_dict},(EPOCH_CNT))
    for key in sorted(stat_dict.keys()):
        log_string('eval mean %s: %f'%(key, stat_dict[key]/(float(batch_idx+1))))

    # Evaluate average precision
    # metrics_dict = ap_calculator.compute_metrics()
    # for key in metrics_dict:
    #     log_string('eval %s: %f'%(key, metrics_dict[key]))

    mean_loss = stat_dict['loss']/float(batch_idx+1)
    return mean_loss


def train(start_epoch):
    global EPOCH_CNT 
    min_loss = 1e10
    loss = 0
    for epoch in range(start_epoch, MAX_EPOCH):
        EPOCH_CNT = epoch
        log_string('**** EPOCH %03d ****' % (epoch))
        log_string('Current learning rate: %f'%(get_current_lr(epoch)))
        log_string('Current BN decay momentum: %f'%(bnm_scheduler.lmbd(bnm_scheduler.last_epoch)))
        log_string(str(datetime.now()))
        # Reset numpy seed.
        # REF: https://github.com/pytorch/pytorch/issues/5059
        np.random.seed()
        train_one_epoch()
        print("Logging Visualizations for validation")
        loss = evaluate_overfit_run()
            # loss = evaluate_one_epoch()
        # Save checkpoint
        # save_dict = {'epoch': epoch+1, # after training one epoch, the start_epoch should be epoch+1
        #             'optimizer_state_dict': optimizer.state_dict(),
        #             'loss': loss,
        #             }
        # try: # with nn.DataParallel() the net is added as a submodule of DataParallel
            # save_dict['model_state_dict'] = net.module.state_dict()
        # except:
        # save_dict['model_state_dict'] = net.state_dict()
        # torch.save(save_dict, os.path.join(LOG_DIR, 'checkpoint.tar'))

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__=='__main__':
    import pickle
    
    if FLAGS.load_saved_model == True:
        MODEL_PATH = os.path.join(LOG_DIR, 'checkpoint.tar')
        loaded_dict = torch.load(MODEL_PATH)
        # print(loaded_dict['model_state_dict'].keys())
        net.load_state_dict(loaded_dict['model_state_dict'])
        net.eval()

        TRAIN_DATALOADER = DataLoader(TRAIN_DATASET, batch_size=1, shuffle=True, num_workers=0, worker_init_fn=my_worker_init_fn)
        
        for batch_idx, batch_data_label in enumerate(TRAIN_DATALOADER):
            
            for key in batch_data_label:
                batch_data_label[key] = batch_data_label[key].to(device)

                
            # Forward pass
            inputs = {'point_clouds': batch_data_label['point_clouds']}
            with torch.no_grad():
                end_points = net(inputs)

            for key in batch_data_label:
                assert(key not in end_points)
                end_points[key] = batch_data_label[key]

            # Extract prediction for visialization
            print(end_points.keys())
            batch_gt_map_cls = parse_groundtruths(end_points, CONFIG_DICT)
            batch_pred_map_cls = parse_predictions(end_points, CONFIG_DICT) 
        
            with open("./visualizations", 'wb') as fp:
                dictie = {}
                dictie['point_cloud'] = batch_data_label['point_clouds'].detach().cpu().numpy()
                dictie['parsed_gt'] = batch_gt_map_cls
                dictie['parsed_predictions'] = batch_pred_map_cls
                pickle.dump(dictie, fp)
            break # only parse one frame from the dataset
        exit()
    else:
        print("Total number of model parameters are {}".format(count_parameters(net)))
        torch.cuda.empty_cache()
    
        train(start_epoch)


