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
import pickle
import shutil

from tqdm import tqdm
from nuscenes import NuScenes
from utils.AppLogger import AppLogger
from network.ptBEV import ptBEVnet
from dataloader.dataset import collate_fn_BEV, Nuscenes_pt, spherical_dataset, collate_dataset_info, SemKITTI_pt
from dataloader.eval_sampler import SequentialDistributedSampler
from network.util.instance_post_processing import get_panoptic_segmentation
from utils.eval_pq import PanopticEval

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
    parser.add_argument('-c', '--configs', default='configs/pa_po_nuscenes_val.yaml')
    parser.add_argument('-l', '--logdir', default='val.log')
    parser.add_argument("--local-rank", default=-1, type=int)
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
    torch.backends.cudnn.benchmark = False  # 是否自动加速，自动选择合适算法，false选择固定算法
    torch.backends.cudnn.deterministic = True  # 为了消除该算法本身的不确定性

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
    
    grid_size = cfgs['dataset']['grid_size']
    pix_fusion = cfgs['model']['pix_fusion']
    min_points = cfgs['dataset']['min_points']

    if_export_npz = False
    
    # 初始化类别名称和数量.抛弃noise类，剩下类别从0开始对齐。
    unique_label, unique_label_str = collate_dataset_info(cfgs)

    # 加noise类
    nclasses = len(unique_label) + 1

    my_model = ptBEVnet(cfgs, nclasses)

    # 加载模型
    if args.resume:
        print(f'load resumed checkpoints from {model_load_path}')
        pretrained_model = torch.load(model_load_path, map_location=torch.device('cpu'))
        # 消除分布式训练时在保存参数的时候多出来的module.
        weights_dict = {}
        for k, v in pretrained_model.items():
            new_k = k.replace('module.', '') if 'module' in k else k
            weights_dict[new_k] = v
        # # debug的时候查看参数量
        # model_dict = my_model.state_dict()
        my_model.load_state_dict(weights_dict, strict=False)


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
        val_pt_dataset = Nuscenes_pt(data_path, split="val", cfgs=cfgs, nusc=nusc, version=version)
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
    if datasetname == 'nuscenes':
        with open("nuscenes.yaml", 'r') as stream:
            nuscenesyaml = yaml.safe_load(stream)
        learning_map = nuscenesyaml['learning_map']
    evaluator = PanopticEval(len(unique_label) + 1 + 1, None, [0, len(unique_label) + 1], min_points=min_points)

    # val set
    if args.local_rank == 0 or args.local_rank == -1:
        if datasetname == "nuscenes":
            if os.path.exists("./semantic_val_preds/"):
                print("=> removing old semantic_val_preds")
                os.system("rm -rf ./semantic_val_preds")    
            if os.path.exists("./panoptic_val_preds/"):
                print("=> removing old panoptic_val_preds")
                os.system("rm -rf ./panoptic_val_preds")
            os.mkdir("semantic_val_preds")
            os.mkdir("semantic_val_preds/lidarseg")
            os.mkdir("semantic_val_preds/lidarseg/val")
            os.mkdir("semantic_val_preds/val/")
            with open("./semantic_val_preds/val/submission.json", "w") as f:
                lidarseg_meta={
                    "meta": 
                        {"use_camera": True,
                        "use_lidar": True,
                        "use_radar": False,
                        "use_map": False,
                        "use_external": False}
                    }
                json.dump(lidarseg_meta, f)

            os.mkdir("panoptic_val_preds")
            os.mkdir("panoptic_val_preds/panoptic")
            os.mkdir("panoptic_val_preds/panoptic/val")
            os.mkdir("panoptic_val_preds/val/")
            with open("./panoptic_val_preds/val/submission.json", "w") as f:
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
            for i in range(8, 9, 1):
                os.mkdir("sequences/%02d" % i)
                os.mkdir("sequences/%02d/predictions" % i)

    if args.local_rank != -1:
        torch.distributed.barrier()

    my_model.eval()
    evaluator.reset()
    sem_hist_list = []
    pbar_val = tqdm(total=len(val_dataset_loader))
    if args.local_rank > 0:
        save_dict = {
            'item1': [],
            'item2': [],
            'item3': [],
            'item4': [],
            'item5': [],
        }
    
    if args.local_rank <= 0:
        print(f"=> Start Evaluation...")    
    with torch.no_grad():
        for i_iter_val, val_dict in enumerate(val_dataset_loader):
            val_dict['voxel_label'] = SemKITTI2train(torch.from_numpy(val_dict['voxel_label']))
            val_dict['voxel_label'] = val_dict['voxel_label'].type(torch.LongTensor).cuda()
            val_dict['gt_center'] = torch.from_numpy(val_dict['gt_center']).cuda()
            val_dict['gt_offset'] = torch.from_numpy(val_dict['gt_offset']).cuda()
            val_dict['inst_map_sparse'] = torch.from_numpy(val_dict['inst_map_sparse']).cuda()
            val_dict['bev_mask'] = torch.from_numpy(val_dict['bev_mask']).cuda()
            val_dict['pol_voxel_ind'] = [torch.from_numpy(i).cuda() for i in val_dict['pol_voxel_ind']]
            val_dict['return_fea'] = [torch.from_numpy(i).type(torch.FloatTensor).cuda() for i in
                                        val_dict['return_fea']]

            if pix_fusion:
                val_dict['camera_channel'] = torch.from_numpy(val_dict['camera_channel']).cuda()
                val_dict['pixel_coordinates'] = [torch.from_numpy(i).cuda() for i in
                                                    val_dict['pixel_coordinates']]
                val_dict['masks'] = [torch.from_numpy(i).cuda() for i in val_dict['masks']]
                val_dict['valid_mask'] = [torch.from_numpy(i).cuda() for i in val_dict['valid_mask']]

            predict_labels, center, offset, instmap, _, cam = my_model(val_dict)
            predict_labels_sem = torch.argmax(predict_labels, dim=1)
            predict_labels_sem = predict_labels_sem.cpu().detach().numpy()
            predict_labels_sem = predict_labels_sem + 1

            val_grid = [i.cpu().numpy() for i in val_dict['pol_voxel_ind']]
            val_pt_labels = val_dict['pt_sem_label']
            val_pt_inst = val_dict['pt_ins_label']

            for count, i_val_grid in enumerate(val_grid):
                # get foreground_mask
                for_mask = torch.zeros(1, grid_size[0], grid_size[1], grid_size[2], dtype=torch.bool).cuda()
                for_mask[0, val_grid[count][:, 0], val_grid[count][:, 1], val_grid[count][:, 2]] = True
                # post processing
                panoptic_labels, center_points = get_panoptic_segmentation(instmap[count],
                    torch.unsqueeze(predict_labels[count], 0), torch.unsqueeze(center[count], 0),
                    torch.unsqueeze(offset[count], 0),
                    val_pt_dataset.thing_list, threshold=cfgs['model']['post_proc']['threshold'],
                    nms_kernel=cfgs['model']['post_proc']['nms_kernel'],
                    top_k=cfgs['model']['post_proc']['top_k'], polar=True, foreground_mask=for_mask)
                panoptic_labels = panoptic_labels.cpu().detach().numpy().astype(np.int32)
                panoptic = panoptic_labels[0, val_grid[count][:, 0], val_grid[count][:, 1], val_grid[count][:, 2]]

                # 语义分割预测出是前景类别，但是全景分割预测它是背景(全景ID预测为0)，当作noise处理
                if datasetname == 'SemanticKitti':
                    panoptic_mask1 = (panoptic <= 8) & (panoptic > 0)
                    panoptic[panoptic_mask1] = 0
                elif datasetname == 'nuscenes':
                    panoptic_mask1 = (panoptic <= 10) & (panoptic > 0)
                    panoptic[panoptic_mask1] = 0
                else:
                    raise NotImplementedError

                if args.local_rank < 1:
                    # 用实例标签的语义
                    if datasetname == 'SemanticKitti':
                        sem_gt = np.squeeze(val_pt_labels[count])
                        inst_gt = np.squeeze(val_pt_inst[count])      
                    elif datasetname == 'nuscenes':
                        sem_gt = np.squeeze(val_pt_inst[count]) // 1000
                        sem_gt = np.vectorize(learning_map.__getitem__)(sem_gt)
                        inst_gt = np.squeeze(val_pt_inst[count])
                    else:
                        raise NotImplementedError

                    evaluator.addBatch(panoptic & 0xFFFF, panoptic, 
                                        sem_gt, inst_gt)
                    # PQ, SQ, RQ, class_all_PQ, class_all_SQ, class_all_RQ = evaluator.getPQ() # for debug
                    sem_hist_list.append(fast_hist_crop(
                        predict_labels_sem[count, val_grid[count][:, 0], val_grid[count][:, 1], val_grid[count][:, 2]],
                        val_pt_labels[count],
                        unique_label))
                else:
                    save_dict['item1'].append(panoptic & 0xFFFF)
                    save_dict['item2'].append(panoptic)
                    save_dict['item3'].append(val_pt_labels[count])
                    save_dict['item4'].append(val_pt_inst[count])
                    save_dict['item5'].append(fast_hist_crop(
                        predict_labels_sem[count, val_grid[count][:, 0], val_grid[count][:, 1], val_grid[count][:, 2]],
                        val_pt_labels[count],
                        unique_label))
                ######################################################################################################
                # export panoptic label predictions
                ######################################################################################################
                sem_val_labels = panoptic & 0xFFFF

                if if_export_npz:
                    if datasetname == 'SemanticKitti':
                        lidar_seq = val_dict["lidar_token"][count].split('_')[0]
                        lidar_name = val_dict["lidar_token"][count].split('_')[1]
                        panoptic_bin_file_path = os.path.join("sequences", lidar_seq, "predictions", str(lidar_name) + '.label')
                        panoptic.astype(np.uint32).tofile(panoptic_bin_file_path)
                        
                    elif datasetname == 'nuscenes':
                        lidar_token = val_dict["lidar_token"][count]
                        lidarseg_bin_file_path = lidar_token + '_lidarseg.bin'
                        lidarseg_bin_file_path = os.path.join("semantic_val_preds/lidarseg/val", lidarseg_bin_file_path )
                        panoptic_bin_file_path = lidar_token + '_panoptic.npz'
                        panoptic_bin_file_path = os.path.join("panoptic_val_preds/panoptic/val", panoptic_bin_file_path)
                        # save sematic val results
                        # since nuscenes val set requires output 1-16
                        sem_val_labels %= 17
                        lidarseg = np.copy(sem_val_labels)
                        lidarseg[lidarseg == 0] = np.random.choice(16) + 1 ## Que ? 对于0号noise, val里并没要求提交，所以随机赋值是否合理？

                        np.array(lidarseg).astype(np.uint8).tofile(lidarseg_bin_file_path)

                        # save panoptic val results
                        inst_val_labels = (panoptic >> 16) % 1000
                        panoptic_val_labels = sem_val_labels * 1000 + inst_val_labels
                        panoptic_val_labels[sem_val_labels == 0] = 0
                        panoptic_val_labels[sem_val_labels >10] = sem_val_labels[sem_val_labels >10] * 1000
                        np.savez_compressed(panoptic_bin_file_path, data=panoptic_val_labels.astype(np.uint16))
                    else:
                        raise NotImplementedError   
                
            pbar_val.update(1)
            # if i_iter_val==10:
            #     break
            del val_dict
        #end for
        pbar_val.close()
        
        if args.local_rank != -1:
            torch.distributed.barrier()
            if args.local_rank > 0:
                os.makedirs('./tmpdir', exist_ok=True)
                pickle.dump(save_dict,
                            open(os.path.join('./tmpdir', 'result_part_{}.pkl'.format(args.local_rank)), 'wb'))
            torch.distributed.barrier()

        if args.local_rank < 1:
            if args.local_rank == 0:
                world_size = torch.distributed.get_world_size()
                for i in range(world_size - 1):
                    part_file = os.path.join('./tmpdir', 'result_part_{}.pkl'.format(i + 1))
                    cur_dict = pickle.load(open(part_file, 'rb'))
                    for j in range(len(cur_dict['item1'])):
                        
                        # 用实例标签的语义
                        if datasetname == 'SemanticKitti':
                            sem_gt = np.squeeze(cur_dict['item3'][j])
                            inst_gt = np.squeeze(cur_dict['item4'][j])
                        elif datasetname == 'nuscenes':
                            sem_gt = np.squeeze(cur_dict['item4'][j] // 1000)
                            sem_gt = np.vectorize(learning_map.__getitem__)(sem_gt)
                            inst_gt = np.squeeze(cur_dict['item4'][j])
                        else:
                            raise NotImplementedError

                        evaluator.addBatch(cur_dict['item1'][j], cur_dict['item2'][j], sem_gt,
                                        inst_gt)
                        sem_hist_list.append(cur_dict['item5'][j])
                if os.path.isdir('./tmpdir'):
                    shutil.rmtree('./tmpdir')
            # end args.local_rank == 0
            ######################################################################################################
            # get PQ results, only for rank 0(Distributed GPU) or rank -1(single GPU)
            ######################################################################################################
            PQ, SQ, RQ, class_all_PQ, class_all_SQ, class_all_RQ = evaluator.getPQ()
            miou, ious = evaluator.getSemIoU()
            logger.info('Validation per class PQ, SQ, RQ and IoU: ')
            for class_name, class_pq, class_sq, class_rq, class_iou in zip(unique_label_str, class_all_PQ[1:-1],
                                                                        class_all_SQ[1:-1], class_all_RQ[1:-1],
                                                                        ious[1:-1]):
                logger.info('%20s : %6.2f%%  %6.2f%%  %6.2f%%  %6.2f%%' % (
                    class_name, class_pq * 100, class_sq * 100, class_rq * 100, class_iou * 100))
            
            thing_upper_idx_dict = {"nuscenes": 10, "SemanticKitti":8} # thing label index: nusc 1-9, kitti 1-8
            upper_idx = thing_upper_idx_dict[datasetname]
            from utils.metric_util import cal_PQ_dagger
            PQ_dagger = cal_PQ_dagger(class_all_PQ, class_all_SQ, upper_idx + 1)
            PQ_th = np.nanmean(class_all_PQ[1: upper_idx + 1]) # exclude 0
            SQ_th = np.nanmean(class_all_SQ[1: upper_idx + 1])
            RQ_th = np.nanmean(class_all_RQ[1: upper_idx + 1])
            PQ_st = np.nanmean(class_all_PQ[upper_idx+1: -1]) # exlucde 17 or 20
            SQ_st = np.nanmean(class_all_SQ[upper_idx+1: -1])
            RQ_st = np.nanmean(class_all_RQ[upper_idx+1: -1])
        
            logger.info('PQ %.3f  PQ_dagger  %.3f  SQ %.3f  RQ %.3f  |  PQ_th %.3f  SQ_th %.3f  RQ_th %.3f  |  PQ_st %.3f  SQ_st %.3f  RQ_st %.3f  |  mIoU %.3f' %
                        (PQ * 100, PQ_dagger * 100, SQ * 100, RQ * 100,
                        PQ_th * 100, SQ_th * 100, RQ_th * 100,
                        PQ_st * 100, SQ_st * 100, RQ_st * 100,
                        miou * 100))
            ######################################################################################################

            logger.info('Current val PQ is %.3f' %
                        (PQ * 100))
            logger.info('Current val miou is %.3f' % (miou * 100))

            ######################################################################################################
            # get mIoU result
            ######################################################################################################

            iou = per_class_iu(sum(sem_hist_list))
            logger.info('Validation per class iou: ')
            for class_name, class_iou in zip(unique_label_str, iou):
                logger.info('%s : %.2f%%' % (class_name, class_iou * 100))
            val_miou = np.nanmean(iou) * 100
            logger.info('Current val miou is %.3f' %
                        val_miou)
            print('*' * 40)
        # end if args.local_rank < 1
        
        if args.local_rank != -1:
            torch.distributed.barrier()

    # end with torch.no_grad():

    if args.local_rank != -1:
        torch.distributed.destroy_process_group()


if __name__ == '__main__':
    main()
