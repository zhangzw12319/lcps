import random

import numpy as np
from copy import deepcopy
from PIL import ImageFilter


class PCDTransformTool:

    def __init__(self, pcd):
        self.pcd = deepcopy(pcd.T)

    def rotate(self, rm):
        self.pcd[:3, :] = np.dot(rm, self.pcd[:3, :])

    def translate(self, tm):
        for i in range(3):
            self.pcd[i, :] = self.pcd[i, :] + tm[i]

    def pcd2image(self, intrinsic, normalize=True):
        assert intrinsic.shape[0] == 3
        assert intrinsic.shape[1] == 3
        self.pcd = np.dot(intrinsic, self.pcd)
        if normalize:
            self.pcd = self.pcd / self.pcd[2:3, :].repeat(3, 0).reshape(3, self.pcd.shape[1])


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


def fetch_color(images: np.ndarray, pixel_coords: np.ndarray, masks: np.ndarray):
    _, h, w, _ = images.shape
    color = np.zeros(shape=(masks.shape[-1], 3), dtype=np.float32)
    for image, coord, mask in zip(images, pixel_coords, masks):
        coord = coord[mask, :]
        coord[:, 0] = (coord[:, 0] + 1.0) / 2 * (w - 1.0)
        coord[:, 1] = (coord[:, 1] + 1.0) / 2 * (h - 1.0)
        coord = np.floor(np.flip(coord, axis=-1)).astype(np.int64)
        color[mask] = image[coord[:, 0], coord[:, 1]]
    return color



