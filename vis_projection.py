#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os

# os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3,4,5'
import argparse
import random
import numpy as np
import json
import yaml
import torch
import shutil
import time

from tqdm import tqdm
from nuscenes import NuScenes
from utils.AppLogger import AppLogger
from network.ptBEV import ptBEVnet
from dataloader.dataset import collate_fn_BEV, Nuscenes_pt, spherical_dataset, collate_dataset_info, SemKITTI_pt
from dataloader.eval_sampler import SequentialDistributedSampler
from network.util.instance_post_processing import get_panoptic_segmentation
from utils.visualize_utils import visualize_img 

import datetime
import warnings
from utils.metric_util import per_class_iu, fast_hist_crop
from utils.wechat_msg import send_msg_wechat

warnings.filterwarnings("ignore")
assync_compensation = True


# 将0-16的语义label转化为0-15和255,为了让语义模型输出16类，不输出noise
def SemKITTI2train(label):
    if isinstance(label, list):
        return [SemKITTI2train_single(a) for a in label]
    else:
        return SemKITTI2train_single(label)


def SemKITTI2train_single(label):
    return label - 1  # uint8 trick


def load_pretrained_model(model, pretrained_model):
    model_dict = model.state_dict()
    pretrained_model = {k: v for k, v in pretrained_model.items() if k in model_dict}
    model_dict.update(pretrained_model)
    model.load_state_dict(model_dict)
    return model


def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-c', '--configs', default='configs/pa_po_nuscenes_val.yaml')
    parser.add_argument('-l', '--logdir', default='count.log')
    parser.add_argument("--local_rank", default=-1, type=int)
    parser.add_argument('-r', "--resume", action="store_true", default=False)
    args = parser.parse_args()

    # 分布式初始化
    if args.local_rank != -1:
        seed = 1
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        torch.cuda.set_device(args.local_rank)
        os.environ['MASTER_ADDR'] = 'localhost'
        # os.environ['MASTER_PORT'] = '29503'
        torch.distributed.init_process_group(backend="nccl", timeout=datetime.timedelta(seconds=5400))
        torch.cuda.set_device(args.local_rank)

    # log文件初始化
    logger = AppLogger("logA", args.logdir)
    
    # 加载cfg文件
    with open(args.configs, 'r') as s:
        cfgs = yaml.safe_load(s)
    logger.info(cfgs)

    datasetname = cfgs['dataset']['name']
    version = cfgs['dataset']['version']
    data_path = cfgs['dataset']['path']
    num_worker = cfgs['dataset']['num_worker']
    val_batch_size = cfgs['model']['val_batch_size']
    model_load_path = cfgs['model']['model_load_path']
    pix_fusion = cfgs['model']['pix_fusion']
    CAM_CHANNELS = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
                             'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']

    # 初始化类别名称和数量.抛弃noise类，剩下类别从0开始对齐。
    unique_label, unique_label_str = collate_dataset_info(cfgs)

    # 加noise类
    nclasses = len(unique_label) + 1

    my_model = ptBEVnet(cfgs, nclasses)

    # 加载模型
    if args.resume:
        print(f'load resumed checkpoints from {model_load_path}')
        pretrained_model = torch.load(model_load_path)
        # 消除分布式训练时在保存参数的时候多出来的module.
        weights_dict = {}
        for k, v in pretrained_model.items():
            new_k = k.replace('module.', '') if 'module' in k else k
            weights_dict[new_k] = v
        # # debug的时候查看参数量
        # model_dict = my_model.state_dict()
        my_model.load_state_dict(weights_dict)


    # DDP的sync_bn，让多卡训练的bn范围正常
    if args.local_rank != -1:
        my_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(my_model)
    my_model.cuda()
    if args.local_rank != -1:
        my_model = torch.nn.parallel.DistributedDataParallel(my_model, device_ids=[args.local_rank],
                                                             output_device=args.local_rank,
                                                             find_unused_parameters=False)

    if datasetname == 'SemanticKitti':
        val_pt_dataset = SemKITTI_pt(os.path.join(data_path, 'dataset', 'sequences'), cfgs, split='val', return_ref=True)
    elif datasetname == 'nuscenes':
        nusc = NuScenes(version=version, dataroot=data_path, verbose=True)
        assert version == "v1.0-trainval" or version == "v1.0-mini"
        val_pt_dataset = Nuscenes_pt(data_path, split="val", cfgs=cfgs, nusc=nusc, version=version, assync_compensation=assync_compensation)
    else:
        raise NotImplementedError

    val_dataset = spherical_dataset(val_pt_dataset, cfgs, ignore_label=0, use_aug=False)

    if args.local_rank != -1:
        val_sampler = SequentialDistributedSampler(val_dataset, val_batch_size)
        val_dataset_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                                         batch_size=val_batch_size,
                                                         collate_fn=collate_fn_BEV,
                                                         sampler=val_sampler,
                                                         pin_memory=True,
                                                         num_workers=num_worker)
    else:
        val_dataset_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                                         batch_size=val_batch_size,
                                                         collate_fn=collate_fn_BEV,
                                                         shuffle=False,
                                                         pin_memory=True,
                                                         num_workers=num_worker)

    # val set
    my_model.eval()
    if args.local_rank != -1:
        torch.distributed.barrier()

    pbar_val = tqdm(total=len(val_dataset_loader))
    if args.local_rank > 0:
        save_dict = {
            'item1': [],
            'item2': [],
            'item3': [],
            'item4': [],
            'item5': [],
        }


    with torch.no_grad():
        if assync_compensation:
            vis_folder_name = "./correct"
        else:
            vis_folder_name = "./wrong"
        if os.path.exists(vis_folder_name):
            print("remove old folders f{vis_folder_name}}")
            shutil.rmtree(vis_folder_name)
        os.mkdir(vis_folder_name)
        for i_iter_val, val_dict in enumerate(val_dataset_loader):
            if args.local_rank < 1 and pix_fusion:
                camera_channel = val_dict['ori_camera_channel']
                pixel_coordinates = val_dict['pixel_coordinates']
                pt_sem_label = val_dict['pt_sem_label']
                
                # if i_iter_val % 10 == 0:
                if i_iter_val % 1 == 0:
                    for bs in range(val_batch_size):
                        pixel_coord_sg = pixel_coordinates[bs]
                        pt_sem_label_sg = pt_sem_label[bs]
                        camera_channel_sg = camera_channel[bs].squeeze()
                        
                        for im_idx in range(camera_channel_sg.shape[0]):
                            pixel_coord_per_img = pixel_coord_sg[im_idx].squeeze()
                            pt_coord_label = np.concatenate((pixel_coord_per_img, pt_sem_label_sg), axis=1)
                            save_path = os.path.join(vis_folder_name, 'iter_' + str(i_iter_val) + '_batch_' + str(bs) + '_' + CAM_CHANNELS[im_idx] + '.png')
                            
                            # filtered by valid mask
                            visualize_img(image=camera_channel_sg[im_idx].squeeze().astype(np.uint8),
                                point=pt_coord_label[val_dict["valid_mask"][bs]==im_idx], class_list=[1,2,3,4,5,6,7,8,9],
                                fig_save_path=save_path)
                            # not filtered by valid mask
                            # visualize_img(image=camera_channel_sg[im_idx].squeeze().astype(np.uint8),
                            #     point=pt_coord_label, class_list=[1,2,3,4,5,6,7,8,9],
                            #     fig_save_path=save_path)
            del val_dict
            pbar_val.update(1)          
        pbar_val.close()

    if args.local_rank != -1:
        torch.distributed.destroy_process_group()


if __name__ == '__main__':
    main()
