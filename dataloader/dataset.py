#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SemKITTI dataloader
"""
import os
import math
import numpy as np
import numba as nb
import yaml
import pickle
from torch.utils import data
from torchvision.transforms import transforms
from torchvision.transforms.functional import hflip, rotate, _get_inverse_affine_matrix, to_tensor, to_pil_image
from PIL import Image
from dataloader.utils import PCDTransformTool, GaussianBlur, fetch_color
from pyquaternion import Quaternion

from .process_panoptic_ori import PanopticLabelGenerator
from .instance_augmentation import Instance_Augmentation, Cont_Mix_InstAugmentation


class SemKITTI_pt(data.Dataset):
    def __init__(self, data_path, cfgs, split='train', return_ref=False):
        self.return_ref = return_ref
        with open("semantic-kitti.yaml", 'r') as stream:
            semkittiyaml = yaml.safe_load(stream)
        self.learning_map = semkittiyaml['learning_map']
        thing_class = semkittiyaml['thing_class'] #0-20类，第20类为需要预测出来的noise类
        self.thing_list = [cl for cl, ignored in thing_class.items() if ignored]
        self.split = split
        # 多模态
        self.root = data_path
        self.labels_mapping = semkittiyaml['learning_map']
        self.pix_fusion = cfgs['model']['pix_fusion']

        self.things_label2name = semkittiyaml['things_label2name']

        # get class distribution weight 
        epsilon_w = 0.001
        origin_class = semkittiyaml['content'].keys()
        # ensure actual mapped classes + 1
        weights = np.zeros((len(semkittiyaml['learning_map_inv']) + 1,) , dtype=np.float32) #这里需要预测第20类（noise），所以weights数量需要+1
        for class_num in origin_class:
            if semkittiyaml['learning_map'][class_num] != 0:
                weights[semkittiyaml['learning_map'][class_num]] += semkittiyaml['content'][class_num]
        self.CLS_LOSS_WEIGHT = 1 / (weights + epsilon_w)
        self.CLS_LOSS_WEIGHT[0] = 0.0

        # sequences
        if self.split == 'train':
            self.seqs = ['00', '01', '02', '03', '04', '05', '06', '07', '09', '10']
        elif self.split == 'val':
            self.seqs = ['08']
        elif self.split == 'test':
            self.seqs = ['11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21']
        else:
            raise Exception('Split must be train/val/test')

        #####################################################################################
        # 多模态
        self.P_dict = {}
        self.Tr_dict = {}
        for seq in self.seqs:
            with open(os.path.join(self.root, seq, 'calib.txt'), 'r') as calib:
                P = []
                for idx in range(4):
                    line = calib.readline().rstrip('\n')[4:]
                    data = line.split(" ")
                    P.append(np.array(data, dtype=np.float32).reshape(3, -1))
                self.P_dict[seq + "_left"] = P[2]
                self.P_dict[seq + "_right"] = P[3]
                line = calib.readline().rstrip('\n')[4:]
                data = line.split(" ")
                self.Tr_dict[seq] = np.array(data, dtype=np.float32).reshape((3, -1))

        self.pcd_files = []
        self.img_files = [[], []]
        self.tokens = [] # token = seq + pcd_name
        # TODO 这里img_files用了kitti_image的双目，需要考虑3D和2D的对应问题，是否需要改变内参
        self.map_idx2seq = []
        for seq in self.seqs:
            for pcd_name in sorted(os.listdir(os.path.join(self.root, seq, 'velodyne'))):
                self.tokens.append(str(seq) + '_' + str(pcd_name[:-4]))
                self.pcd_files.append(os.path.join(self.root, seq, 'velodyne', str(pcd_name)))
                self.img_files[0].append(os.path.join(self.root, seq, 'image_2', str(pcd_name[:-4]) + '.png'))
                self.img_files[1].append(os.path.join(self.root, seq, 'image_3', str(pcd_name[:-4]) + '.png'))
                self.map_idx2seq.append(seq)
        # self.IMAGE_SIZE = [368, 1224]  # 368, 1216
        self.IMAGE_SIZE = [360, 640] #ppn_th68中，归一化size从[368, 1224]变为[360, 640]
        # self.transform = transforms.CenterCrop(size=self.CROP_SIZE)
        self.resize = transforms.Compose([
            transforms.Resize(size=self.IMAGE_SIZE)
        ])
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225] ),
        ])
        self.augment = transforms.Compose([
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4)  # not strengthened
            ], p=0.5),
            # transforms.RandomGrayscale(p=0.1),
        ])
        self.img_aug = True
        self.flip_aug = False
        self.flip_aug_rate = 0.5
        self.rotate_aug = False
        self.rotate_max_angle = [-15, 15]
        self.mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.pcd_files)

    def _load_pcd(self, index):
        filepath = self.pcd_files[index]
        pts = np.fromfile(filepath, dtype=np.float32).reshape((-1, 4))
        img1 = Image.open(self.img_files[0][index]).convert('RGB')
        img2 = Image.open(self.img_files[1][index]).convert('RGB')
        if self.split == 'test':
            return pts, img1, img2
        else:
            lidar_label_path = filepath.replace('velodyne', 'labels')[:-3] + 'label'
            annotates = np.fromfile(lidar_label_path, dtype=np.uint32).reshape([-1, 1])
            sem_labels = annotates & 0xFFFF # delete high 16 digits binary
            # do label mapping in point_dataset.py, not in spherical_dataset.py
            sem_labels = np.vectorize(self.labels_mapping.__getitem__)(sem_labels).flatten()
            return pts, sem_labels, annotates, img1, img2

    def _mappcd2img(self, index, pts, im_size, color_lorr='left'):
        seq = self.map_idx2seq[index]
        P, Tr = self.P_dict[seq + "_" +color_lorr], self.Tr_dict[seq]
        pts_homo = np.column_stack((pts, np.array([1] * pts.shape[0], dtype=pts.dtype)))
        Tr_homo = np.row_stack((Tr, np.array([0, 0, 0, 1], dtype=Tr.dtype)))
        pixel_coord = np.matmul(Tr_homo, pts_homo.T)
        pixel_coord = np.matmul(P, pixel_coord).T
        pixel_coord = pixel_coord / (pixel_coord[:, 2].reshape(-1, 1))
        pixel_coord = pixel_coord[:, :2]

        x_on_image = (pixel_coord[:, 0] >= 0) & (pixel_coord[:, 0] <= (im_size[0] - 1))
        y_on_image = (pixel_coord[:, 1] >= 0) & (pixel_coord[:, 1] <= (im_size[1] - 1))
        mask = x_on_image & y_on_image & (pts[:, 0] > 0) # only front points
        return pixel_coord, mask

    def __getitem__(self, index):
        if self.split in ["train", "val"]:
            pts, sem_data, inst_data, img1, img2 = self._load_pcd(index)
            noise_mask = sem_data == 0
            sem_data[noise_mask] = 20
        elif self.split == "test":
            pts, img1, img2 = self._load_pcd(index)
        else:
            raise NotImplementedError("only train, val or test")
            # pts_ahead_idx = pts[:, 0] > 0 #过滤掉x<=0的点 ppn_th68版本中不进行过滤
            # pts = pts[pts_ahead_idx]
            # sem_data = sem_data[pts_ahead_idx]
            # inst_data = inst_data[pts_ahead_idx]
        ######################################################################
        
        pixel_coordinates1, mask1 = self._mappcd2img(index, pts[:, :3], img1.size, "left")
        pixel_coordinates2, mask2 = self._mappcd2img(index, pts[:, :3], img2.size, "right")
        # pts = pts[mask, :] #ppn69 版本中，load kitti时，对不在相机视野内的点进行筛除
        # sem_data = sem_data[mask]
        # inst_data = inst_data[mask]

        pixel_coordinates1[:, 0] = pixel_coordinates1[:, 0] / (img1.size[0] - 1) * 2 - 1.0
        pixel_coordinates1[:, 1] = pixel_coordinates1[:, 1] / (img1.size[1] - 1) * 2 - 1.0
        pixel_coordinates2[:, 0] = pixel_coordinates2[:, 0] / (img2.size[0] - 1) * 2 - 1.0
        pixel_coordinates2[:, 1] = pixel_coordinates2[:, 1] / (img2.size[1] - 1) * 2 - 1.0
        pixel_coordinates = np.array([pixel_coordinates1, pixel_coordinates2])
        masks = np.array([mask1, mask2])
        # masks = np.logical_or(mask1, mask2)

        img1 = self.resize(img1)
        img2 = self.resize(img2)
        ori_camera = np.stack((np.array(img1).astype('float32'), np.array(img2).astype('float32')), axis=0)
        if self.img_aug:
            img1 = self.augment(img1)
            img2 = self.augment(img2)
        img1 = self.transform(img1).permute((1,2,0))
        img2 = self.transform(img2).permute((1,2,0))
        
        camera = np.stack((img1, img2), axis=0)
        valid_mask = np.array([-1] * pts.shape[0])
        monocular = False
        if monocular:
            valid_mask[mask1] = 1
        else:
            valid_mask[mask1] = 1
            valid_mask[mask2] = 2

        
        data_tuple = (pts[:, :3], pts[:, 3], self.tokens[index])
        if self.split in ["train", "val"]:
            # data_dict = {
            #     "xyz": pts[:, :3],
            #     "feat": pts[:, 3],
            #     "labels": sem_data,
            #     "insts": inst_data,
            # }
            data_tuple += (sem_data, inst_data)
        else:
            data_tuple += (-1, -1)

        if self.pix_fusion:
            if monocular: #如果只是单目
                fusion_tuple = (camera[0], pixel_coordinates[0], masks[0], valid_mask, ori_camera[0])
            else:#双目
                fusion_tuple = (camera, pixel_coordinates, masks, valid_mask, ori_camera)
            data_tuple += (fusion_tuple,)

        return data_tuple


class Nuscenes_pt(data.Dataset):
    def __init__(self, data_path, split, cfgs, nusc, version, assync_compensation=True):
        with open("nuscenes.yaml", 'r') as stream:
            nuscenesyaml = yaml.safe_load(stream)
        sample_pkl_path = cfgs['dataset']['sample_pkl_path']

        if version == 'v1.0-mini':
            if split == 'train':
                imageset = os.path.join(sample_pkl_path, "nuscenes_infos_train_mini.pkl")
            elif split == 'val':
                imageset = os.path.join(sample_pkl_path, "nuscenes_infos_val_mini.pkl")
        elif version == 'v1.0-trainval':
            if split == 'train':
                imageset = os.path.join(sample_pkl_path, "nuscenes_infos_train.pkl")
            elif split == 'val':
                imageset = os.path.join(sample_pkl_path, "nuscenes_infos_val.pkl")
        elif version == 'v1.0-test':
            imageset = os.path.join(sample_pkl_path, "nuscenes_infos_test.pkl")
        else:
            raise NotImplementedError

        with open(imageset, 'rb') as f:
            data = pickle.load(f)

        self.learning_map = nuscenesyaml['learning_map']
        self.split = split
        self.thing_list = [cl for cl, is_thing in nuscenesyaml['thing_class'].items() if is_thing]
        self.nusc_infos = data['infos']
        self.data_path = data_path
        self.cfgs = cfgs
        self.nusc = nusc
        self.version = version

        # 多模态
        self.pix_fusion = self.cfgs['model']['pix_fusion']
        self.IMAGE_SIZE = (900, 1600)
        self.transform = transforms.Compose([transforms.Resize(size=[int(x * 0.4) for x in self.IMAGE_SIZE])])
        self.CAM_CHANNELS = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
                             'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']
        self.open_asynchronous_compensation = assync_compensation
        # corresponding to self.CAM_CHANNELS，fov from https://www.nuscenes.org/nuscenes
        # 6 * 3 (cosine lowerbound, cosine upperbound, if_front)
        self.cam_fov = [[-np.cos(11 * math.pi/36), np.cos(11 * math.pi/36), 1], # CAM_FRONT
                        [np.cos(7 * math.pi /18), 1, 1], # CAM_FRONT_RIGHT
                        [-1, -np.cos(7 * math.pi / 18), 1], # CAM_FRONT_LEFT
                        [-0.5, 0.5, -1], # CAM_BACK 120 degrees fov
                        [-1, -np.cos(7 * math.pi /18), -1], # CAM_BACK_LEFT
                        [np.cos(7 * math.pi / 18), 1, -1]] #CAM_BACK_RIGHT 

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.nusc_infos)

    def __getitem__(self, index):
        info = self.nusc_infos[index]
        lidar_sd_token = self.nusc.get('sample', info['token'])['data']['LIDAR_TOP']

        if self.version == "v1.0-trainval":
            lidar_path = info['lidar_path'][16:]
        elif self.version == "v1.0-mini":
            lidar_path = info['lidar_path'][44:]
        elif self.version == "v1.0-test":
            lidar_path = info['lidar_path'][16:]
            
        points = np.fromfile(os.path.join(self.data_path, lidar_path), dtype=np.float32, count=-1).reshape([-1, 5])

        # 多模态部分
        if self.pix_fusion:
            camera_channel = []  # 6, H, W, 3
            pixel_coordinates = []  # 6, N, 2
            masks = []
            valid_mask = np.array([-1] * points.shape[0])
            lidar_token = lidar_sd_token
            lidar_channel = self.nusc.get("sample_data", lidar_token)
            rho = np.sqrt(points[:, 0] ** 2 + points[:, 1] ** 2)
            cosine_value = points[:, 0] / rho
            
            for idx, channel in enumerate(self.CAM_CHANNELS):
                cam_token = info['cams'][channel]['sample_data_token']
                cam_channel = self.nusc.get('sample_data', cam_token)
                im = Image.open(os.path.join(self.nusc.dataroot, cam_channel['filename'])).convert('RGB')
                camera_channel.append(np.array(self.transform(im)).astype('float32'))
                pcd_trans_tool = PCDTransformTool(points[:, :3])
                # Points live in the point sensor frame. So they need to be transformed via global to the image plane.
                # First step: transform the pointcloud to the ego vehicle frame for the timestamp of the sweep.
                cs_record = self.nusc.get('calibrated_sensor', lidar_channel['calibrated_sensor_token'])
                pcd_trans_tool.rotate(Quaternion(cs_record['rotation']).rotation_matrix)
                pcd_trans_tool.translate(np.array(cs_record['translation']))

                if self.open_asynchronous_compensation:
                    # Second step: transform from ego to the global frame at timestamp of the first frame in the sequence pack.
                    poserecord = self.nusc.get('ego_pose', lidar_channel['ego_pose_token'])
                    pcd_trans_tool.rotate(Quaternion(poserecord['rotation']).rotation_matrix)
                    pcd_trans_tool.translate(np.array(poserecord['translation']))
                
                    # Third step: transform from global into the ego vehicle frame for the timestamp of the image.
                    poserecord = self.nusc.get('ego_pose', cam_channel['ego_pose_token'])
                    pcd_trans_tool.translate(-np.array(poserecord['translation']))
                    pcd_trans_tool.rotate(Quaternion(poserecord['rotation']).rotation_matrix.T)

                # Fourth step: transform from ego into the camera.
                cs_record = self.nusc.get('calibrated_sensor', cam_channel['calibrated_sensor_token'])
                pcd_trans_tool.translate(-np.array(cs_record['translation']))
                pcd_trans_tool.rotate(Quaternion(cs_record['rotation']).rotation_matrix.T)
                mask = np.ones(points.shape[0], dtype=bool)
                mask = np.logical_and(mask, pcd_trans_tool.pcd[2, :] > 1)
                # Fifth step: project from 3d coordinate to 2d coordinate
                pcd_trans_tool.pcd2image(np.array(cs_record['camera_intrinsic']))
                pixel_coord = pcd_trans_tool.pcd[:2, :]
                pixel_coord[0, :] = pixel_coord[0, :] / (im.size[0] - 1.0) * 2.0 - 1.0  # width
                pixel_coord[1, :] = pixel_coord[1, :] / (im.size[1] - 1.0) * 2.0 - 1.0  # height
                # pixel_coordinates.append(pixel_coord.T)

                # Remove points that are either outside or behind the camera. Leave a margin of 1 pixel for aesthetic reasons.
                # Also make sure points are at least 1m in front of the camera to avoid seeing the lidar points on the camera
                # casing for non-keyframes which are slightly out of sync.
                mask = np.logical_and(mask, pixel_coord[0, :] > -1)
                mask = np.logical_and(mask, pixel_coord[0, :] < 1)
                mask = np.logical_and(mask, pixel_coord[1, :] > -1)
                mask = np.logical_and(mask, pixel_coord[1, :] < 1)
                # detailed filter
                # mask = np.logical_and(mask, points[:, 1] * self.cam_fov[idx][2] > 0)
                # filtered by fov angle has been disbaled, 
                # since bugs:the NuScenes Official doesn't release the camera fov
                # coordinated at the vehicle center, but at the camera center
                # mask = np.logical_and(mask, cosine_value > self.cam_fov[idx][0])
                # mask = np.logical_and(mask, cosine_value < self.cam_fov[idx][1])
                valid_mask[mask] = idx
                masks.append(mask)
                pixel_coordinates.append(pixel_coord.T)

            ori_camera_channel = np.stack(camera_channel, axis=0)
            for i in range(6):
                # 归一化
                camera_channel[i] /= 255.0
                camera_channel[i][:, :, 0] = (camera_channel[i][:, :, 0] - 0.485) / 0.229
                camera_channel[i][:, :, 1] = (camera_channel[i][:, :, 1] - 0.456) / 0.224
                camera_channel[i][:, :, 2] = (camera_channel[i][:, :, 2] - 0.406) / 0.225
            camera_channel = np.stack(camera_channel, axis=0)
            pixel_coordinates = np.stack(pixel_coordinates, axis=0)
            masks = np.stack(masks, axis=0)
            fusion_tuple = (camera_channel, pixel_coordinates, masks, valid_mask, ori_camera_channel)

        
        data_tuple = (points[:, :3], points[:, 3], lidar_sd_token)
        
        # load label
        if self.version != "v1.0-test":
            lidarseg_labels_filename = os.path.join(self.nusc.dataroot,
                                                    self.nusc.get('lidarseg', lidar_sd_token)['filename'])
            panoptic_labels_filename = os.path.join(self.nusc.dataroot,
                                                    self.nusc.get('panoptic', lidar_sd_token)['filename'])
            panoptic_label = np.load(panoptic_labels_filename)['data']
            points_label = np.fromfile(lidarseg_labels_filename, dtype=np.uint8).reshape([-1, 1])
            points_label = np.vectorize(self.learning_map.__getitem__)(points_label)
            noise_mask = points_label == 0
            points_label[noise_mask] = 17
            data_tuple += (points_label.astype(np.uint8), panoptic_label)
        else:
            data_tuple += (-1, -1)

        # data_tuple = (points[:, :3], points[:, 3], points_label.astype(np.uint8), panoptic_label)
        if self.pix_fusion:
            data_tuple += (fusion_tuple,)

        return data_tuple


def absoluteFilePaths(directory):
    for dirpath, _, filenames in os.walk(directory):
        for f in filenames:
            yield os.path.abspath(os.path.join(dirpath, f))


# transformation between Cartesian coordinates and polar coordinates
def cart2polar(input_xyz):
    rho = np.sqrt(input_xyz[:, 0] ** 2 + input_xyz[:, 1] ** 2)
    phi = np.arctan2(input_xyz[:, 1], input_xyz[:, 0])
    return np.stack((rho, phi, input_xyz[:, 2]), axis=1)


def polar2cat(input_xyz_polar):
    x = input_xyz_polar[0] * np.cos(input_xyz_polar[1])
    y = input_xyz_polar[0] * np.sin(input_xyz_polar[1])
    return np.stack((x, y, input_xyz_polar[2]), axis=0)


class spherical_dataset(data.Dataset):
    def __init__(self, in_dataset, cfgs, ignore_label=0, fixed_volume_space=True, use_aug=True):
        'Initialization'
        self.point_cloud_dataset = in_dataset
        self.grid_size = np.asarray(cfgs['dataset']['grid_size'])
        self.rotate_aug = cfgs['dataset']['rotate_aug'] if use_aug else False
        self.flip_aug = cfgs['dataset']['flip_aug'] if use_aug else False
        self.ignore_label = ignore_label
        self.fixed_volume_space = fixed_volume_space
        self.max_volume_space = cfgs['dataset']['max_volume_space']
        self.min_volume_space = cfgs['dataset']['min_volume_space']
        self.inst_aug=("inst_aug" in cfgs['dataset']) and ("if_use" in cfgs['dataset']['inst_aug']) and \
            cfgs['dataset']['inst_aug']['if_use'] if use_aug else False
        

        self.panoptic_proc = PanopticLabelGenerator(self.grid_size, sigma=cfgs['dataset']['gt_generator']['sigma'],
                                                    polar=True)
        
        ### add instance augmentation ###
        if self.inst_aug:
            assert ("aug_type" in cfgs['dataset']['inst_aug'])
            assert ("inst_pkl_path" in cfgs['dataset']['inst_aug'])
            assert ("inst_trans" in cfgs['dataset']['inst_aug'])
            assert ("inst_rotate" in cfgs['dataset']['inst_aug'])
            assert ("inst_flip" in cfgs['dataset']['inst_aug'])
            assert ("inst_add" in cfgs['dataset']['inst_aug'])
            thing_list = self.point_cloud_dataset.thing_list
            if cfgs['dataset']['inst_aug']['aug_type'] == "contmix":
                self.copy_paste = Cont_Mix_InstAugmentation(dataset_name=cfgs["dataset"]["name"],
                                instance_pkl_path=cfgs['dataset']['inst_aug']['inst_pkl_path'],
                                thing_list= thing_list,
                                class_weight=None,
                                random_trans=cfgs['dataset']['inst_aug']['inst_trans'],
                                random_flip=cfgs['dataset']['inst_aug']['inst_flip'],
                                random_rotate=cfgs['dataset']['inst_aug']['inst_rotate']
                                )
            else:
                self.copy_paste = Instance_Augmentation(instance_pkl_path=cfgs['dataset']['inst_aug']['inst_pkl_path'],
                                                    thing_list=thing_list,
                                                    class_weight=None,
                                                    random_flip=cfgs['dataset']['inst_aug']['inst_flip'],
                                                    random_add=cfgs['dataset']['inst_aug']['inst_add'],
                                                    random_rotate=cfgs['dataset']['inst_aug']['inst_rotate'])
        else:
            self.inst_aug = None

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.point_cloud_dataset)

    def __getitem__(self, index):
        'Generates one sample of data'
        # index = 8709
        data = self.point_cloud_dataset[index]
        if len(data) == 5:
            xyz, feat, token, labels, insts = data
            fusion_tuple = None
            if len(feat.shape) == 1: feat = feat[..., np.newaxis]
        elif len(data) == 6:
            xyz, feat, token, labels, insts, fusion_tuple = data
            if len(feat.shape) == 1: feat = feat[..., np.newaxis]
        else:
            raise Exception('Return invalid data tuple')
        

        if type(labels)==np.ndarray:
            if len(labels.shape) == 1: labels = labels[..., np.newaxis]
            if len(insts.shape) == 1: insts = insts[..., np.newaxis]

        # copy-paste数据增强
        if self.inst_aug:
            old_xyz_number = xyz.shape[0]
            # Note: 此处的点云的labels, 已经把noise移动到最后一类，但是所有类别都没有-1.
            xyz, labels, insts, feat = self.copy_paste.instance_aug(
                                            point_xyz=xyz,
                                            point_label=labels,
                                            point_inst=insts,
                                            point_feat=feat)
            # Currently for pasted points, we do not align them to camera features
            # TODO: Support the image alignment for pasted points, refer to PointAugmenting
            # url: https://github.com/VISION-SJTU/PointAugmenting
            if fusion_tuple is not None:
                _, pixel_coordinates, masks, valid_mask, _ = fusion_tuple
                add_number = xyz.shape[0] - old_xyz_number
                pixel_coordinates = np.pad(pixel_coordinates, ((0, 0), (0, add_number), (0,0)), 'constant', constant_values=0)
                masks = np.pad(masks, ((0, 0), (0, add_number)), 'constant', constant_values=False)
                valid_mask = np.pad(valid_mask, ((0, add_number)), 'constant', constant_values=-1)
                fusion_tuple = (fusion_tuple[0], pixel_coordinates, masks, valid_mask, fusion_tuple[4])

        # 逆时针旋转，保存角度
        rotate_deg = 0
        if self.rotate_aug:
            #offset_aug2
            x_offset = np.random.random() * 2 * 2 - 2
            y_offset = np.random.random() * 2 * 2 - 2
            xyz[:,0] = xyz[:,0] + x_offset
            xyz[:, 1] = xyz[:, 1] + y_offset


            rotate_deg = np.random.random() * 360
            rotate_rad = np.deg2rad(rotate_deg)
            c, s = np.cos(rotate_rad), np.sin(rotate_rad)
            j = np.matrix([[c, s], [-s, c]])
            xyz[:, :2] = np.dot(xyz[:, :2], j)


        # 随机翻转
        if self.flip_aug:
            flip_type = np.random.choice(3, 1)
            if flip_type == 1:
                xyz[:, 0] = -xyz[:, 0]
                if len(data) == 6:
                    img = fusion_tuple[0]
                    img = img[:, :, ::-1, :]
                    flip_coor = fusion_tuple[1]
                    flip_coor[:, :, 0] = -flip_coor[:, :, 0]
                    fusion_tuple = (img, flip_coor, fusion_tuple[2], fusion_tuple[3], fusion_tuple[4])
            elif flip_type == 2:
                xyz[:, 1] = -xyz[:, 1]
                if len(data) == 6:
                    img = fusion_tuple[0]
                    img = img[:, :, ::-1, :]
                    flip_coor = fusion_tuple[1]
                    flip_coor[:, :, 0] = -flip_coor[:, :, 0]
                    fusion_tuple = (img, flip_coor, fusion_tuple[2], fusion_tuple[3], fusion_tuple[4])

        # 转化成极坐标系
        xyz_pol = cart2polar(xyz)

        # 统一使用预先定义好的坐标范围
        if self.fixed_volume_space:
            max_bound = np.asarray(self.max_volume_space)
            min_bound = np.asarray(self.min_volume_space)
        else:
            max_bound_r = np.percentile(xyz_pol[:, 0], 100, axis=0)
            min_bound_r = np.percentile(xyz_pol[:, 0], 0, axis=0)
            max_bound = np.max(xyz_pol[:, 1:], axis=0)
            min_bound = np.min(xyz_pol[:, 1:], axis=0)
            max_bound = np.concatenate(([max_bound_r], max_bound))
            min_bound = np.concatenate(([min_bound_r], min_bound))

        # 把点转换成其对应的网格坐标，其中加一个1e-8是为了clip后的点，边界上不会越界
        crop_range = max_bound - min_bound
        intervals = crop_range / (self.grid_size)
        min_bound = min_bound + [1e-8, 1e-8, 1e-8]
        if (intervals == 0).any(): print("Zero interval!")
        grid_ind = (np.floor((np.clip(xyz_pol, min_bound, max_bound) - min_bound) / intervals)).astype(int)

        # 每个网格的起始角落在真实坐标系下的绝对位置
        dim_array = np.ones(len(self.grid_size) + 1, int)
        dim_array[0] = -1
        voxel_position = np.indices(self.grid_size) * intervals.reshape(dim_array) + min_bound.reshape(dim_array)

        if type(labels)==np.ndarray and type(insts)==np.ndarray:
            # 生成每个voxel的语义label，单个网格内采用最大投票
            voxel_label = np.ones(self.grid_size, dtype=np.uint8) * self.ignore_label
            current_grid = grid_ind[:np.size(labels)]
            label_voxel_pair = np.concatenate([current_grid, labels], axis=1)
            label_voxel_pair = label_voxel_pair[np.lexsort((current_grid[:, 0], current_grid[:, 1], current_grid[:, 2])), :]
            voxel_label = nb_process_label(np.copy(voxel_label), label_voxel_pair)

            # 生成前景点的mask，insts为0的点要单独特判一下，可能是有个别前景物体的点被标记为了insts为0
            mask = np.zeros_like(labels, dtype=bool)
            for label in self.point_cloud_dataset.thing_list:
                # mask[labels == label] = True
                mask[np.logical_and(labels == label, insts != 0)] = True

            # 生成每个voxel的实例insts id，单个网格内采用最大投票
            voxel_inst = insts[mask].squeeze()
            unique_inst = np.unique(voxel_inst)
            unique_inst_dict = {label: idx + 1 for idx, label in enumerate(unique_inst)}
            if voxel_inst.size > 1:
                voxel_inst = np.vectorize(unique_inst_dict.__getitem__)(voxel_inst)
                # process panoptic
                processed_inst = np.ones(self.grid_size[:2], dtype=np.uint8) * self.ignore_label
                inst_voxel_pair = np.concatenate([current_grid[mask[:, 0], :2], voxel_inst[..., np.newaxis]], axis=1)
                inst_voxel_pair = inst_voxel_pair[np.lexsort((current_grid[mask[:, 0], 0], current_grid[mask[:, 0], 1])), :]
                processed_inst = nb_process_inst(np.copy(processed_inst), inst_voxel_pair)
            else:
                # processed_inst = np.zeros([480, 360])
                processed_inst = np.zeros([self.grid_size[0], self.grid_size[1]])

            center, center_points, offset = self.panoptic_proc(insts[mask], xyz[:np.size(labels)][mask[:, 0]],
                                                            processed_inst, voxel_position[:2, :, :, 0],
                                                            unique_inst_dict, min_bound, intervals)

        # center data on each voxel for PTnet
        voxel_centers = (grid_ind.astype(np.float32) + 0.5) * intervals + min_bound
        return_xyz = xyz_pol - voxel_centers
        return_xyz = np.concatenate((return_xyz, xyz_pol, xyz[:, :2]), axis=1)
        return_fea = np.concatenate((return_xyz, feat), axis=1)

        # bev_mask = np.zeros((1, 480, 360), dtype=bool)
        bev_mask = np.zeros((1, self.grid_size[0], self.grid_size[1]), dtype=bool)
        uni_out = np.unique(grid_ind[:,0:2],axis=0)
        bev_mask[0, uni_out[:,0], uni_out[:,1]] = True

        return_dict = {}
        return_dict['lidar_token'] = token
        return_dict['xyz_cart'] = xyz
        return_dict['return_fea'] = return_fea
        return_dict['pol_voxel_ind'] = grid_ind
        return_dict['rotate_deg'] = rotate_deg
        
        if type(labels) == np.ndarray and type(insts) == np.ndarray:
            return_dict['voxel_label'] = voxel_label
            return_dict['gt_center'] = center
            return_dict['gt_offset'] = offset
            return_dict['inst_map_sparse'] = processed_inst != 0
            return_dict['bev_mask'] = bev_mask
            return_dict['pt_sem_label'] = labels
            return_dict['pt_ins_label'] = insts

        if len(data) == 6:
            return_dict['camera_channel'] = fusion_tuple[0]
            return_dict['pixel_coordinates'] = fusion_tuple[1]
            return_dict['masks'] = fusion_tuple[2]
            return_dict['valid_mask'] = fusion_tuple[3]
            return_dict['ori_camera_channel'] = fusion_tuple[4]
            if type(labels) == np.ndarray and type(insts) == np.ndarray:
                point_with_pix_mask = return_dict['valid_mask'] > -1
                # image labels projected by points
                return_dict['im_label'] = return_dict['pt_sem_label'][point_with_pix_mask]

        return return_dict



@nb.jit('u1[:,:,:](u1[:,:,:],i8[:,:])', cache=True, parallel=False)
def nb_process_label(processed_label, sorted_label_voxel_pair):
    label_size = 256
    counter = np.zeros((label_size,), dtype=np.uint16)
    counter[sorted_label_voxel_pair[0, 3]] = 1
    cur_sear_ind = sorted_label_voxel_pair[0, :3]
    for i in range(1, sorted_label_voxel_pair.shape[0]):
        cur_ind = sorted_label_voxel_pair[i, :3]
        if not np.all(np.equal(cur_ind, cur_sear_ind)):
            processed_label[cur_sear_ind[0], cur_sear_ind[1], cur_sear_ind[2]] = np.argmax(counter)
            counter = np.zeros((label_size,), dtype=np.uint16)
            cur_sear_ind = cur_ind
        counter[sorted_label_voxel_pair[i, 3]] += 1
    processed_label[cur_sear_ind[0], cur_sear_ind[1], cur_sear_ind[2]] = np.argmax(counter)
    return processed_label


@nb.jit('u1[:,:](u1[:,:],i8[:,:])', cache=True, parallel=False)
def nb_process_inst(processed_inst, sorted_inst_voxel_pair):
    label_size = 256
    counter = np.zeros((label_size,), dtype=np.uint16)
    counter[sorted_inst_voxel_pair[0, 2]] = 1
    cur_sear_ind = sorted_inst_voxel_pair[0, :2]
    for i in range(1, sorted_inst_voxel_pair.shape[0]):
        cur_ind = sorted_inst_voxel_pair[i, :2]
        if not np.all(np.equal(cur_ind, cur_sear_ind)):
            processed_inst[cur_sear_ind[0], cur_sear_ind[1]] = np.argmax(counter)
            counter = np.zeros((label_size,), dtype=np.uint16)
            cur_sear_ind = cur_ind
        counter[sorted_inst_voxel_pair[i, 2]] += 1
    processed_inst[cur_sear_ind[0], cur_sear_ind[1]] = np.argmax(counter)
    return processed_inst


def collate_fn_BEV(data):
    return_dict = {}
    for i, k in enumerate(data[0]):
        return_dict[k] = [d[k] for d in data]
    if 'voxel_label' in return_dict:
        return_dict['voxel_label'] = np.stack(return_dict['voxel_label'])
    if 'gt_center' in return_dict:
        return_dict['gt_center'] = np.stack(return_dict['gt_center'])
    if 'gt_offset' in return_dict:
        return_dict['gt_offset'] = np.stack(return_dict['gt_offset'])
    if 'bev_mask' in return_dict:
        return_dict['bev_mask'] = np.stack(return_dict['bev_mask'])
    if 'inst_map_sparse' in return_dict:
        return_dict['inst_map_sparse'] = np.stack(return_dict['inst_map_sparse'])
    if 'camera_channel' in return_dict:
        return_dict['camera_channel'] = np.stack(return_dict['camera_channel'])
    if 'ori_camera_channel' in return_dict:
        return_dict['ori_camera_channel'] = np.stack(return_dict['ori_camera_channel'])

    return return_dict


def collate_dataset_info(cfgs):
    dataset_type = cfgs['dataset']['name']
    if dataset_type == 'SemanticKitti':
        with open("semantic-kitti.yaml", 'r') as stream:
            semkittiyaml = yaml.safe_load(stream)
        SemKITTI_label_name = dict()
        for i in sorted(list(semkittiyaml['learning_map'].keys()))[::-1]:
            SemKITTI_label_name[semkittiyaml['learning_map'][i]] = semkittiyaml['labels'][i]
        unique_label = np.asarray(sorted(list(SemKITTI_label_name.keys())))[1:] - 1
        unique_label_str = [SemKITTI_label_name[x] for x in unique_label + 1]
        return unique_label, unique_label_str
    elif dataset_type == 'nuscenes':
        with open("nuscenes.yaml", 'r') as stream:
            nuscenesyaml = yaml.safe_load(stream)
        nuscenes_label_name = nuscenesyaml['labels_16']
        unique_label = np.asarray(sorted(list(nuscenes_label_name.keys())))[1:] - 1
        unique_label_str = [nuscenes_label_name[x] for x in unique_label + 1]
        return unique_label, unique_label_str
    else:
        raise NotImplementedError
