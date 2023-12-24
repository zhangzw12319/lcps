#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import numba as nb

from network.cylinder_fea_generator import cylinder_fea
from network.segmentator_3d_asymm_spconv import Asymm_3d_spconv
from network.BEV_Unet import BEV_Unet


class ptBEVnet(nn.Module):

    def __init__(self, cfgs, nclasses):
        super(ptBEVnet, self).__init__()
        self.nclasses = nclasses
        self.cylinder_3d_generator = cylinder_fea(
            cfgs,
            grid_size=cfgs['dataset']['grid_size'],
            fea_dim=9,
            out_pt_fea_dim=256,
            fea_compre=16,
            nclasses=nclasses,
            use_sara=cfgs['model']['use_sara'],
            use_att=cfgs['model']['use_att'] if 'use_att' in cfgs['model'] else False)
        self.cylinder_3d_spconv_seg = Asymm_3d_spconv(
            cfgs,
            output_shape=cfgs['dataset']['grid_size'],
            use_norm=True,
            num_input_features=16,
            init_size=32,
            nclasses=nclasses)
        self.UNet = BEV_Unet(128)

    def forward(self, train_dict):
        train_pt_fea_ten, train_vox_ten = train_dict['return_fea'], train_dict['pol_voxel_ind']
        # 对每个网格的点进行池化
        coords, features_3d, softmax_pix_logits, cam = self.cylinder_3d_generator(train_pt_fea_ten, train_vox_ten, train_dict)
        # 主干网络
        sem_prediction, center, offset, instmap = self.cylinder_3d_spconv_seg(features_3d, coords, len(train_pt_fea_ten), train_dict)

        center = self.UNet(center)

        return sem_prediction, center, offset, instmap, softmax_pix_logits, cam


def grp_range_torch(a, dev):
    idx = torch.cumsum(a, 0)
    id_arr = torch.ones(idx[-1], dtype=torch.int64, device=dev)
    id_arr[0] = 0
    id_arr[idx[:-1]] = -a[:-1] + 1
    return torch.cumsum(id_arr, 0)


def parallel_FPS(np_cat_fea, K):
    return nb_greedy_FPS(np_cat_fea, K)


@nb.jit('b1[:](f4[:,:],i4)', nopython=True, cache=True)
def nb_greedy_FPS(xyz, K):
    start_element = 0
    sample_num = xyz.shape[0]
    sum_vec = np.zeros((sample_num, 1), dtype=np.float32)
    xyz_sq = xyz ** 2
    for j in range(sample_num):
        sum_vec[j, 0] = np.sum(xyz_sq[j, :])
    pairwise_distance = sum_vec + np.transpose(sum_vec) - 2 * np.dot(xyz, np.transpose(xyz))

    candidates_ind = np.zeros((sample_num,), dtype=np.bool_)
    candidates_ind[start_element] = True
    remain_ind = np.ones((sample_num,), dtype=np.bool_)
    remain_ind[start_element] = False
    all_ind = np.arange(sample_num)

    for i in range(1, K):
        if i == 1:
            min_remain_pt_dis = pairwise_distance[:, start_element]
            min_remain_pt_dis = min_remain_pt_dis[remain_ind]
        else:
            cur_dis = pairwise_distance[remain_ind, :]
            cur_dis = cur_dis[:, candidates_ind]
            min_remain_pt_dis = np.zeros((cur_dis.shape[0],), dtype=np.float32)
            for j in range(cur_dis.shape[0]):
                min_remain_pt_dis[j] = np.min(cur_dis[j, :])
        next_ind_in_remain = np.argmax(min_remain_pt_dis)
        next_ind = all_ind[remain_ind][next_ind_in_remain]
        candidates_ind[next_ind] = True
        remain_ind[next_ind] = False

    return candidates_ind
