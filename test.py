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

import datetime
import warnings
from utils.metric_util import per_class_iu, fast_hist_crop

warnings.filterwarnings("ignore")


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
    parser.add_argument('-c', '--configs', default='configs/pa_po_nuscenes_test.yaml')
    parser.add_argument('-l', '--logdir', default='test.log')
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
    # TODO test the influence of the followings in the test benchmark:
    # torch.backends.cudnn.benchmark = True  # 是否自动加速，自动选择合适算法，false选择固定算法
    # torch.backends.cudnn.deterministic = True  # 为了消除该算法本身的不确定性
    
    # 加载cfg文件
    with open(args.configs, 'r') as s:
        cfgs = yaml.safe_load(s)
    logger.info(cfgs)

    datasetname = cfgs['dataset']['name']
    version = cfgs['dataset']['version']
    data_path = cfgs['dataset']['path']
    num_worker = cfgs['dataset']['num_worker']
    test_batch_size = cfgs['model']['test_batch_size']
    model_load_path = cfgs['model']['model_load_path']
    
    grid_size = cfgs['dataset']['grid_size']
    pix_fusion = cfgs['model']['pix_fusion']

    # 初始化类别名称和数量.抛弃noise类，剩下类别从0开始对齐。
    unique_label, unique_label_str = collate_dataset_info(cfgs)

    # 加noise类
    nclasses = len(unique_label) + 1

    my_model = ptBEVnet(cfgs, nclasses)
    # my_model = ptBEVnet(grid_size, nclasses, pix_fusion=cfgs['model']['pix_fusion_path'])

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
        test_pt_dataset = SemKITTI_pt(os.path.join(data_path, 'dataset', 'sequences'), cfgs, split='test', return_ref=True)
    elif datasetname == 'nuscenes':
        nusc = NuScenes(version=version, dataroot=data_path, verbose=True)
        assert version == "v1.0-test"
        test_pt_dataset = Nuscenes_pt(data_path, split="", cfgs=cfgs, nusc=nusc, version=version)
    else:
        raise NotImplementedError

    test_dataset = spherical_dataset(test_pt_dataset, cfgs, ignore_label=0, use_aug=False)

    if args.local_rank != -1:
        test_sampler = SequentialDistributedSampler(test_dataset, test_batch_size)
        test_dataset_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                         batch_size=test_batch_size,
                                                         collate_fn=collate_fn_BEV,
                                                         sampler=test_sampler,
                                                         pin_memory=True,
                                                         num_workers=num_worker)
    else:
        test_dataset_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                         batch_size=test_batch_size,
                                                         collate_fn=collate_fn_BEV,
                                                         shuffle=False,
                                                         pin_memory=True,
                                                         num_workers=num_worker)

    # test set
    my_model.eval()
    if args.local_rank == 0 or args.local_rank == -1:
        if datasetname == "nuscenes":
            if os.path.exists("./semantic_test_preds/"):
                print("=> removing old semantic_test_preds")
                os.system("rm -rf ./semantic_test_preds")    
            if os.path.exists("./panoptic_test_preds/"):
                print("=> removing old panoptic_test_preds")
                os.system("rm -rf ./panoptic_test_preds")
            os.mkdir("semantic_test_preds")
            os.mkdir("semantic_test_preds/lidarseg")
            os.mkdir("semantic_test_preds/lidarseg/test")
            os.mkdir("semantic_test_preds/test/")
            with open("./semantic_test_preds/test/submission.json", "w") as f:
                lidarseg_meta={
                    "meta": 
                        {"use_camera": True,
                        "use_lidar": True,
                        "use_radar": False,
                        "use_map": False,
                        "use_external": False}
                    }
                json.dump(lidarseg_meta, f)

            os.mkdir("panoptic_test_preds")
            os.mkdir("panoptic_test_preds/panoptic")
            os.mkdir("panoptic_test_preds/panoptic/test")
            os.mkdir("panoptic_test_preds/test/")
            with open("./panoptic_test_preds/test/submission.json", "w") as f:
                panoseg_meta={
                    "meta": 
                        {"task": "segmentation",
                        "use_camera": True,
                        "use_lidar": True,
                        "use_radar": False,
                        "use_map": False,
                        "use_external": False}
                    }
                json.dump(panoseg_meta, f)
        elif datasetname == 'SemanticKitti':
            if os.path.exists("./sequences/"):
                print("=> removing old kitti sequences preds")
                os.system("rm -rf ./sequences")
            os.mkdir("sequences")
            for i in range(11,22, 1):
                os.mkdir("sequences/%02d" % i)
                os.mkdir("sequences/%02d/predictions" % i)

    if args.local_rank != -1:
        torch.distributed.barrier()

    pbar_test = tqdm(total=len(test_dataset_loader))
    if args.local_rank > 0:
        save_dict = {
            'item1': [],
            'item2': [],
            'item3': [],
            'item4': [],
            'item5': [],
        }

    with torch.no_grad():
        for i_iter_val, test_dict in enumerate(test_dataset_loader):

            test_dict['pol_voxel_ind'] = [torch.from_numpy(i).cuda() for i in test_dict['pol_voxel_ind']]
            test_dict['return_fea'] = [torch.from_numpy(i).type(torch.FloatTensor).cuda() for i in
                                        test_dict['return_fea']]

            if pix_fusion:
                test_dict['camera_channel'] = torch.from_numpy(test_dict['camera_channel']).cuda()
            if pix_fusion:
                test_dict['pixel_coordinates'] = [torch.from_numpy(i).cuda() for i in
                                                    test_dict['pixel_coordinates']]
                test_dict['masks'] = [torch.from_numpy(i).cuda() for i in test_dict['masks']]
                test_dict['valid_mask'] = [torch.from_numpy(i).cuda() for i in test_dict['valid_mask']]

            predict_labels, center, offset, instmap, _, cam = my_model(test_dict)
            predict_labels_sem = torch.argmax(predict_labels, dim=1)
            predict_labels_sem = predict_labels_sem.cpu().detach().numpy()
            predict_labels_sem = predict_labels_sem + 1

            test_grid = [i.cpu().numpy() for i in test_dict['pol_voxel_ind']]

            for count, i_val_grid in enumerate(test_grid):
                # get foreground_mask
                for_mask = torch.zeros(1, grid_size[0], grid_size[1], grid_size[2], dtype=torch.bool).cuda()
                for_mask[0, test_grid[count][:, 0], test_grid[count][:, 1], test_grid[count][:, 2]] = True
                # post processing
                panoptic_labels, center_points = get_panoptic_segmentation(instmap[count],
                    torch.unsqueeze(predict_labels[count], 0), torch.unsqueeze(center[count], 0),
                    torch.unsqueeze(offset[count], 0),
                    test_pt_dataset.thing_list, threshold=cfgs['model']['post_proc']['threshold'],
                    nms_kernel=cfgs['model']['post_proc']['nms_kernel'],
                    top_k=cfgs['model']['post_proc']['top_k'], polar=True, foreground_mask=for_mask)
                panoptic_labels = panoptic_labels.cpu().detach().numpy().astype(np.int32)
                panoptic = panoptic_labels[0, test_grid[count][:, 0], test_grid[count][:, 1], test_grid[count][:, 2]]

                # 语义分割预测出是前景类别，但是全景分割预测它是背景(全景ID预测为0)，当作noise处理
                if datasetname == 'SemanticKitti':
                    panoptic_mask1 = (panoptic <= 8) & (panoptic > 0)
                    panoptic[panoptic_mask1] = 0
                elif datasetname == 'nuscenes':
                    panoptic_mask1 = (panoptic <= 10) & (panoptic > 0)
                    panoptic[panoptic_mask1] = 0
                else:
                    raise NotImplementedError
            
                sem_test_labels = panoptic & 0xFFFF

                if datasetname == 'SemanticKitti':
                    lidar_seq = test_dict["lidar_token"][count].split('_')[0]
                    lidar_name = test_dict["lidar_token"][count].split('_')[1]
                    panoptic_bin_file_path = os.path.join("sequences", lidar_seq, "predictions", str(lidar_name) + '.label')

                    sem_test_labels %= 20
                    panoptic_test_labels = sem_test_labels + (panoptic >> 16) << 16
                    panoptic_test_labels[sem_test_labels == 0] = 0
                    panoptic_test_labels[sem_test_labels > 8] = sem_test_labels[sem_test_labels > 8]

                    panoptic.astype(np.uint32).tofile(panoptic_bin_file_path)
                    
                elif datasetname == 'nuscenes':
                    lidar_token = test_dict["lidar_token"][count]
                    lidarseg_bin_file_path = lidar_token + '_lidarseg.bin'
                    lidarseg_bin_file_path = os.path.join("semantic_test_preds/lidarseg/test", lidarseg_bin_file_path )
                    panoptic_bin_file_path = lidar_token + '_panoptic.npz'
                    panoptic_bin_file_path = os.path.join("panoptic_test_preds/panoptic/test", panoptic_bin_file_path)
                    # save sematic test results
                    # since nuscenes test set requires output 1-16
                    sem_test_labels %= 17
                    lidarseg = np.copy(sem_test_labels)
                    lidarseg[lidarseg == 0] = np.random.choice(16) + 1 ## Que ? 对于0号noise, test里并没要求提交，所以随机赋值是否合理？

                    np.array(lidarseg).astype(np.uint8).tofile(lidarseg_bin_file_path)

                    # save panoptic test results
                    inst_test_labels = (panoptic >> 16) % 1000
                    panoptic_test_labels = sem_test_labels * 1000 + inst_test_labels
                    panoptic_test_labels[sem_test_labels == 0] = 0
                    panoptic_test_labels[sem_test_labels >10] = sem_test_labels[sem_test_labels >10] * 1000
                    np.savez_compressed(panoptic_bin_file_path, data=panoptic_test_labels.astype(np.uint16))
                else:
                    raise NotImplementedError   
                
            pbar_test.update(1)
            # if i_iter_val==10:
            #     break
            del test_dict
        pbar_test.close()

    if args.local_rank != -1:
        torch.distributed.destroy_process_group()


if __name__ == '__main__':
    main()
