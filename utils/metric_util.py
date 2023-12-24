# -*- coding:utf-8 -*-
# author: Xinge
# @file: metric_util.py 

import numpy as np


def fast_hist(pred, label, n):
    k = (label >= 0) & (label < n)
    bin_count = np.bincount(
        n * label[k].astype(int) + pred[k], minlength=n ** 2)
    return bin_count[:n ** 2].reshape(n, n)


def per_class_iu(hist):
    return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))


def fast_hist_crop(output, target, unique_label):
    hist = fast_hist(output.flatten(), target.flatten(), np.max(unique_label) + 3)
    hist = hist[unique_label + 1, :]
    hist = hist[:, unique_label + 1]
    return hist


def cal_PQ_dagger(pq_arr, sq_arr, upper_idx):
    pq_arr_tmp = []
    for idx, _ in enumerate(pq_arr):
        if idx >= upper_idx:
            pq_arr_tmp.append(sq_arr[idx].item())
        else:
            pq_arr_tmp.append(pq_arr[idx].item())
    return np.nanmean(np.array(pq_arr_tmp[1:-1])) # exclude 0 and the No.17/No.20