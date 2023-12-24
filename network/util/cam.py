import torch
import numpy as np
import cv2

from PIL import Image
from torch.nn import functional as F
from torchvision.transforms import transforms


# modified from CAMs https://github.com/zhoubolei/CAM
def returnCAM(feature_conv, weight_softmax):
    """
    feature_conv: tensor, shape (Batch, C, H, W)
    weight_softmax: tensor, shape (nclasses, C)
    """
    assert len(feature_conv.shape) == 4
    assert feature_conv.shape[1] == weight_softmax.shape[1]
    bs, nc, h, w = feature_conv.shape
    feature_conv = feature_conv.permute(1, 0, 2, 3)
    
    cam = torch.matmul(weight_softmax, feature_conv.reshape((nc, bs*h*w)))
    cam = (cam - cam.min()) / (cam.max() - cam.min())
    cam = cam.reshape(-1, bs, h, w)
    return cam