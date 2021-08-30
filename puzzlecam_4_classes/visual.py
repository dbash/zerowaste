# Copyright (C) 2020 * Ltd. All rights reserved.
# author : Sanghyeon Jo <josanghyeokn@gmail.com>

import os
import sys
import copy
import math
import shutil
import random
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

from torch.utils.data import DataLoader

from core.puzzle_utils import *
from core.networks import *
from core.datasets import *

from tools.general.io_utils import *
from tools.general.time_utils import *
from tools.general.json_utils import *

from tools.ai.log_utils import *
from tools.ai.demo_utils import *
from tools.ai.optim_utils import *
from tools.ai.torch_utils import *
from tools.ai.evaluate_utils import *

from tools.ai.augment_utils import *
from tools.ai.randaugment import *

parser = argparse.ArgumentParser()

###############################################################################
# Dataset
###############################################################################
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--num_workers', default=4, type=int)
parser.add_argument('--data_dir', default='../VOCtrainval_11-May-2012/', type=str)

###############################################################################
# Network
###############################################################################
parser.add_argument('--architecture', default='resnet50', type=str)
parser.add_argument('--mode', default='normal', type=str) # fix

###############################################################################
# Hyperparameter
###############################################################################
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--max_epoch', default=15, type=int)

parser.add_argument('--lr', default=0.1, type=float)
parser.add_argument('--wd', default=1e-4, type=float)
parser.add_argument('--nesterov', default=True, type=str2bool)

parser.add_argument('--image_size', default=512, type=int)
parser.add_argument('--min_image_size', default=320, type=int)
parser.add_argument('--max_image_size', default=640, type=int)

parser.add_argument('--print_ratio', default=0.1, type=float)

parser.add_argument('--tag', default='', type=str)
parser.add_argument('--augment', default='', type=str)

# For Puzzle-CAM
parser.add_argument('--num_pieces', default=4, type=int)

# 'cl_pcl'
# 'cl_re'
# 'cl_conf'
# 'cl_pcl_re'
# 'cl_pcl_re_conf'
parser.add_argument('--loss_option', default='cl_pcl_re', type=str)

parser.add_argument('--level', default='feature', type=str) 

parser.add_argument('--re_loss', default='L1_Loss', type=str) # 'L1_Loss', 'L2_Loss'
parser.add_argument('--re_loss_option', default='masking', type=str) # 'none', 'masking', 'selection'

# parser.add_argument('--branches', default='0,0,0,0,0,1', type=str)

parser.add_argument('--alpha', default=1.0, type=float)
parser.add_argument('--alpha_schedule', default=0.50, type=float)

if __name__ == '__main__':
    ###################################################################################
    # Arguments
    ###################################################################################
    args = parser.parse_args()
    
    log_dir = create_directory(f'./experiments/logs/')
    data_dir = create_directory(f'./experiments/data/')
    model_dir = create_directory('./experiments/models/')
    tensorboard_dir = create_directory(f'./experiments/tensorboards/{args.tag}/')
    
    log_path = log_dir + f'{args.tag}.txt'
    data_path = data_dir + f'{args.tag}.json'
    model_path = model_dir + f'{args.tag}.pth'

    set_seed(args.seed)
    log_func = lambda string='': log_print(string, log_path)
    
    log_func('[i] {}'.format(args.tag))
    log_func()

    ###################################################################################
    # Transform, Dataset, DataLoader
    ###################################################################################
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    normalize_fn = Normalize(imagenet_mean, imagenet_std)
    
    train_transforms = [
        RandomResize(args.min_image_size, args.max_image_size),
        RandomHorizontalFlip(),
    ]
    if 'colorjitter' in args.augment:
        train_transforms.append(transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1))
    
    if 'randaugment' in args.augment:
        train_transforms.append(RandAugmentMC(n=2, m=10))

    train_transform = transforms.Compose(train_transforms + \
        [
            Normalize(imagenet_mean, imagenet_std),
            RandomCrop(args.image_size),
            Transpose()
        ]
    )
    test_transform = transforms.Compose([
        Normalize_For_Segmentation(imagenet_mean, imagenet_std),
        Top_Left_Crop_For_Segmentation(args.image_size),
        Transpose_For_Segmentation()
    ])
    
    meta_dic = read_json('./data/VOC_2012.json')
    class_names = np.asarray(meta_dic['class_names'])
    
    train_dataset = VOC_Dataset_For_Classification(args.data_dir, 'train_aug', train_transform)

    valid_dataset_for_seg = VOC_Dataset_For_Testing_CAM(args.data_dir, 'val', test_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True, drop_last=True)
    valid_loader_for_seg = DataLoader(valid_dataset_for_seg, batch_size=args.batch_size, num_workers=1, drop_last=True)

    val_iteration = len(train_loader)
    log_iteration = int(val_iteration * args.print_ratio)
    max_iteration = args.max_epoch * val_iteration
    
    ###################################################################################
    # Network
    ###################################################################################
    model = Classifier(args.architecture, meta_dic['classes'], mode=args.mode)
    
    gap_fn = model.global_average_pooling_2d

    model = model.cuda()
    model.eval()
    try:
        use_gpu = os.environ['CUDA_VISIBLE_DEVICES']
    except KeyError:
        use_gpu = '0'

    the_number_of_gpu = len(use_gpu.split(','))
    if the_number_of_gpu > 1:
        log_func('[i] the number of gpu : {}'.format(the_number_of_gpu))
        model = nn.DataParallel(model)


    load_model(model, model_path, parallel=the_number_of_gpu > 1)

    

    train_timer = Timer()
    eval_timer = Timer()

    vis_dir = '/research/axns2/mabdelfa/core/puzzle_zerowaste_4_org/cams/'
    def evaluate(loader):
        model.eval()
        eval_timer.tik()

        cc = 0
        with torch.no_grad():
            length = len(loader)
            for step, (images, labels, gt_masks) in enumerate(loader):
                images = images.cuda()
                labels = labels.cuda()

                _, features = model(images, with_cam=True)
                
                # features = resize_for_tensors(features, images.size()[-2:])
                # gt_masks = resize_for_tensors(gt_masks, features.size()[-2:], mode='nearest')

                mask = labels.unsqueeze(2).unsqueeze(3)
                cams = (make_cam(features) * mask)

                # for visualization
                if step == 0:
                    obj_cams = cams.max(dim=1)[0]
                    
                    for b in range(8):
                        image = get_numpy_from_tensor(images[b])
                        cam = get_numpy_from_tensor(obj_cams[b])

                        image = denormalize(image, imagenet_mean, imagenet_std)[..., ::-1]
                        h, w, c = image.shape

                        cam = (cam * 255).astype(np.uint8)
                        cam = cv2.resize(cam, (w, h), interpolation=cv2.INTER_LINEAR)
                        cam = colormap(cam)

                        image = cv2.addWeighted(image, 0.3, cam, 0.1, 0)[..., ::-1]

                        typee = '_cam'
                        cv2.imwrite(vis_dir + str(cc) + typee +'.png', image)
                        cc+=1
                        image = image.astype(np.float32) / 255.
                        print('Creating cam...', cc)


    evaluate(valid_loader_for_seg)
            