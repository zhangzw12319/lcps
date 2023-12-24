#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import torch
from .lovasz_losses import lovasz_softmax

def _neg_loss(pred, gt):
    ''' Modified focal loss. Exactly the same as CornerNet.
        Runs faster and costs a little bit more memory
        (https://github.com/tianweiy/CenterPoint)
    Arguments:
        pred (batch x c x h x w)
        gt (batch x c x h x w)
    '''
    pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(1).float()

    neg_weights = torch.pow(1 - gt, 4)

    # loss = 0

    pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
    neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds
    return - (pos_loss + neg_loss)

class FocalLoss(torch.nn.Module):
    '''nn.Module warpper for focal loss'''
    def __init__(self):
        super(FocalLoss, self).__init__()
        self.neg_loss = _neg_loss

    def forward(self, out, target):
        return self.neg_loss(out, target)

class PanopticLoss(torch.nn.Module):
    def __init__(self, ignore_label = 255, center_loss_weight = 100, offset_loss_weight = 1, instmap_loss_weight = 1, center_loss = 'MSE', offset_loss = 'L1'):
        super(PanopticLoss, self).__init__()
        self.CE_loss = torch.nn.CrossEntropyLoss(ignore_index=ignore_label)
        assert center_loss in ['MSE','FocalLoss']
        assert offset_loss in ['L1','SmoothL1']
        if center_loss == 'MSE':
            self.center_loss_fn = torch.nn.MSELoss()
        elif center_loss == 'FocalLoss':
            self.center_loss_fn = FocalLoss()
        else: raise NotImplementedError
        if offset_loss == 'L1':
            self.offset_loss_fn = torch.nn.L1Loss()
        elif offset_loss == 'SmoothL1':
            self.offset_loss_fn = torch.nn.SmoothL1Loss()
        else: raise NotImplementedError
        self.instmap_loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
        self.center_loss_weight = center_loss_weight
        self.offset_loss_weight = offset_loss_weight
        self.instmap_loss_weight = instmap_loss_weight

        print('Using '+ center_loss +' for heatmap regression, weight: '+str(center_loss_weight))
        print('Using '+ offset_loss +' for offset regression, weight: '+str(offset_loss_weight))

        self.loss_dict={'semantic_loss':[],
                        'heatmap_loss':[],
                        'offset_loss':[],
                        'instmap_loss':[]}

    def reset_loss_dict(self):
        self.loss_dict={'semantic_loss':[],
                        'heatmap_loss':[],
                        'offset_loss':[],
                        'instmap_loss':[]}

    def forward(self,prediction,center,offset,instmap,gt_label,gt_center,gt_offset,gt_instmap,bev_mask,save_loss = True):
        # semantic loss
        loss = lovasz_softmax(torch.nn.functional.softmax(prediction), gt_label,ignore=255) + self.CE_loss(prediction,gt_label)
        sem_loss = loss
        if save_loss:
            self.loss_dict['semantic_loss'].append(loss.item())
        # center heatmap loss
        center_mask = (gt_center>0.01) | (torch.min(torch.unsqueeze(gt_label, 1),dim=4)[0]<255) ####################################################
        center_loss = self.center_loss_fn(center,gt_center) * center_mask
        # safe division
        if center_mask.sum() > 0:
            center_loss = center_loss.sum() / center_mask.sum() * self.center_loss_weight
        else:
            center_loss = center_loss.sum() * 0
        if save_loss:
            self.loss_dict['heatmap_loss'].append(center_loss.item())
        loss += center_loss
        # offset loss
        offset_mask = gt_offset != 0
        offset_loss = self.offset_loss_fn(offset,gt_offset) * offset_mask
        # safe division
        if offset_mask.sum() > 0:
            offset_loss = offset_loss.sum() / offset_mask.sum() * self.offset_loss_weight
        else:
            offset_loss = offset_loss.sum() * 0
        if save_loss:
            self.loss_dict['offset_loss'].append(offset_loss.item())
        loss += offset_loss

        # instmap loss
        # gt_instmap_view = gt_instmap.cpu().numpy()
        instmap_loss = self.instmap_loss_fn(instmap,gt_instmap)
        instmap_loss = instmap_loss * bev_mask
        if bev_mask.sum() > 0:
            instmap_loss = instmap_loss.sum() / bev_mask.sum() * 10
        else:
            instmap_loss = instmap_loss.sum() * 0
        if save_loss:
            self.loss_dict['instmap_loss'].append(instmap_loss.item())
        loss += self.instmap_loss_weight * instmap_loss

        return loss


class PixelLoss(torch.nn.Module):
    def __init__(self, ignore_label = 255, save_loss = True):
        super(PixelLoss, self).__init__()
        self.CE_loss = torch.nn.CrossEntropyLoss(ignore_index=ignore_label)
        self.ignore_label = ignore_label
        self.loss_dict={'pix_loss':[]}
        self.save_loss = save_loss
        
    def forward(self, prediction, label):
        prediction = prediction.to(torch.float32)
        pix_sem_loss = self.CE_loss(prediction, label.squeeze())
        if self.save_loss:
            self.loss_dict['pix_loss'].append(torch.tensor(pix_sem_loss, device="cpu").item())
        return pix_sem_loss