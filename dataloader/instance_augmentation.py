#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pickle
import os
import random

# import numba as nb


class Cont_Mix_InstAugmentation:

    def __init__(self, dataset_name, instance_pkl_path, thing_list,
                 class_min_num=None, class_name=None, class_weight=None, random_flip=False, random_rotate=False,
                 random_trans=False, thing_coincide_lb=0.001, stuff_coincide_lb=0.05):
        """
        instance_pkl_path: 用于数据增强的单个instance的pkl文件
        thing_list：前景list
        ground_list: 被认为与“路面”相近的类别
        pair_list: 每个thing可能被“放”在哪些ground类上的对应类别，如[[11], [11,12],[11,12,13]]
        add_num: mix 的实例数量
        num_classes:
        class_min_num: 每个类的inst point最少点数，默认不得少于10
        thing_coincide_lb: 用于实例碰撞检测，thing类别点的重合度的下界
        stuff_coincide_lb: stuff类别点的重合度的下界
        """
        self.dataset_name = dataset_name  # 'nuscenes' or 'semantickitti'
        assert self.dataset_name in ['nuscenes', 'SemanticKitti']
        if self.dataset_name == 'nuscenes':
            num_classes = 17 + 1  # 类别总数
            # nus_thing_list  1: barrier 2: bicycle 3: bus 4: car 5: construction_vehicle 6: motorcycle 7: pedestrian 8: traffic_cone 9: trailer 10: truck
            ground_list = [11, 12, 13]  # 被认为接近路面的类别id,   11: 'driveable_surface'  12: 'other_flat'   13: 'sidewalk'
            pair_list = [[11], [11], [11], [11], [11], [11], [11, 12, 13], [11, 12, 13], [11], [11]]
        elif self.dataset_name == 'SemanticKitti':
            num_classes = 20 + 1  # 类别总数
            # kitti_thing_list #  1:car  2:bicycle  3:motorcycle  4:truck  5:other-vehicle 6:person 7:bicyclist 8:motorcyclist
            ground_list = [9, 10, 11]  # 9: "road"   10: "parking"   11: "sidewalk"  #12 "other-ground" 数量过少难以检测暂时不加
            pair_list = [[9, 10], [9, 11], [9], [9], [9], [9, 10, 11], [9, 11], [9]]

        self.add_num = random.randint(2,10) #5 #默认每个场景随机增强的实例数量是5个
        self.thing_coincide_lb = thing_coincide_lb
        self.stuff_coincide_lb = stuff_coincide_lb
        self.thing_list = thing_list
        self.ground_list = ground_list
        if class_weight is not None: #增强时每个类抽取的概率
            self.instance_weight = [class_weight[i] for i in self.thing_list]
            self.instance_weight = np.array(self.instance_weight) / np.sum(self.instance_weight)
            self.instance_weight = 1 / np.sqrt(self.instance_weight + 1e-4)
            self.instance_weight = self.instance_weight / np.sum(self.instance_weight)
        else:
            assert len(self.thing_list) > 0
            self.instance_weight = np.array([1.0 / len(self.thing_list) for _ in thing_list])
        if class_min_num is not None:
            assert len(class_min_num) == len(thing_list)
            self.class_min_num = class_min_num
        else:
            self.class_min_num = [10] * len(thing_list)

        self.class_name = class_name #指定实例增强的类别的名字
        self.random_flip = random_flip
        self.random_rotate = random_rotate
        self.random_trans = random_trans
        self.inst_root = os.path.dirname(instance_pkl_path)
        self.instance_pkl_path = instance_pkl_path
        with open(instance_pkl_path, 'rb') as f:
            self.instance_path = pickle.load(f)
        if class_name is not None:
            self.instance_path = [self.instance_path[c] for c in self.class_name]

        self.grid_size = np.array([5., 5.], dtype=np.float32)
        self.ground_classes = ground_list
        self.pair_list = pair_list
        self.num_classes = num_classes
        self.thing_class = np.zeros(shape=(num_classes,), dtype=bool)
        for c_i in thing_list:
            self.thing_class[c_i] = True

    def _cat_grid(self, xyz: np.ndarray):
        # 将x,y坐标grid化
        if isinstance(self.grid_size, list):
            self.grid_size = np.array(self.grid_size, dtype=np.float32)
        assert isinstance(self.grid_size, np.ndarray)
        grid = np.round(xyz[:, :2] / self.grid_size).astype(np.int32)
        grid -= grid.min(0, keepdims=True)
        return grid

    def ground_analyze(self, point_xyz, point_label):
        ground_info = {}
        for g_i in self.ground_list:
            g_m = (point_label == g_i) #属于ground类的mask
            if np.sum(g_m) == 0:
                continue
            g_xyz = point_xyz[g_m[:,0]] #需要注意g_m[:,0]
            grid = self._cat_grid(g_xyz) #对于xy坐标grid化，每个点属于xoy平面的grid下标
            uq, inv, count = np.unique(grid, axis=0, return_inverse=True, return_counts=True)
            patch_center = np.zeros(shape=(uq.shape[0], g_xyz.shape[1]))
            for idx, p_id in enumerate(inv): #求每个grid的重心
                patch_center[p_id] += g_xyz[idx]
            patch_center /= count.reshape(-1, 1)
            patch_center = patch_center[count >= 20] #筛除点数小于20个点grid
            ground_info[g_i] = patch_center
        return ground_info
    # compared to Panoptic-Polarnet, we incorporate sweep aggregation and copy paste together(agg=)
    # @nb.jit('int32[:,:,:](int32[:,:,:],int32[:,:], int32)', nopython=True, cache=True, parallel=False)
    def instance_aug(self, point_xyz, point_label, point_inst, point_feat=None):
        """
        Args:
            point_xyz: [N, 3], point location
            point_label: [N, 1], class label
            point_inst: [N, 1], instance label
            point_feat: [N, 1], l
        """
        ground_info = self.ground_analyze(point_xyz, point_label) #每个grid的重心
        instance_choice = np.random.choice(len(self.thing_list), self.add_num, replace=True, p=self.instance_weight) #根据add_num数量，随机抽取实例类别
        uni_inst, uni_inst_count = np.unique(instance_choice, return_counts=True)
        un_xyz_inst_l = np.unique(point_inst) #所有点云的实例标签unique
        total_point_num = 0
        for inst_i, count in zip(uni_inst, uni_inst_count): #对于抽取的每一类实例
            # random_choice = np.random.choice(self.instance_path[inst_i], count)
            key_list = [k for k in self.instance_path.keys()]
            random_choice = np.random.choice(self.instance_path[key_list[inst_i]], count) #bin 名称
            pair_ground = self.pair_list[inst_i] #这个类可能被放在哪些ground类上
            for add_idx, inst_info in enumerate(random_choice):
                if self.dataset_name=='nuscenes':
                    if len(inst_info.split("nuscenes")) == 2:
                        path = os.path.join("../data/nuscenes", inst_info.split("nuscenes")[1][1:]) # it is ugly
                    else:
                        path = inst_info
                    points = np.fromfile(path, dtype=np.float32).reshape(-1, 5)[:, :4]
                elif self.dataset_name=='SemanticKitti':
                    # print(inst_info)
                    # print(inst_info.split("preprocess"))
                    if len(inst_info.split("preprocess")) == 2:
                        path = os.path.join("../data/SemanticKitti/dataset", inst_info.split("preprocess")[1][1:]) # it is ugly
                    else:
                        path = inst_info
                    points = np.fromfile(path, dtype=np.float32).reshape(-1, 4)[:, :4]
                add_xyz = points[:, :3]
                if add_xyz.shape[0] < self.class_min_num[inst_i]: #少于10个点，不添加
                    continue
                center = np.mean(add_xyz, axis=0) #要添加的实例的重心
                min_xyz = np.min(add_xyz, axis=0) #左下角坐标
                # max_xyz = np.max(add_xyz, axis=0)
                center[2] = min_xyz[2]
                ground_list = []
                for i in pair_ground: #对于每个可能放置在其上面的类
                    ch = ground_info.get(i, None) #每个属于第i类的grid的坐标
                    if ch is None or ch.shape[0] == 0:
                        continue
                    ch_i = np.random.choice(ground_info[i].shape[0], 5)  #从这些grid中随机选5个
                    ground_list.append(ground_info[i][ch_i])
                ground_list = np.concatenate(ground_list, axis=0)
                ground_list = np.random.permutation(ground_list)
                break_flag = False
                for g_center in ground_list:  #对于刚刚选出的每个grid地板的重心
                    for _ in range(5): #“放五次”，放进则break，放不进跳过这个实例增强
                        if self.random_trans: #是否随机平移
                            rand_xy = (2 * np.random.random(2) - 1) * self.grid_size / 5
                            rand_z = np.random.random(1) * 0.05
                            g_center[:2] += rand_xy
                            g_center[2] += rand_z
                        if self.random_flip: #随机翻转
                            long_axis = [center[0], center[1]] / (center[0] ** 2 + center[1] ** 2) ** 0.5
                            short_axis = [-long_axis[1], long_axis[0]]
                            # random flip
                            add_xyz[:, :2] = self.instance_flip(add_xyz[:, :2], [long_axis, short_axis], [center[0], center[1]])
                        if self.random_rotate:#随机旋转
                            rot_noise = np.random.uniform(-np.pi / 20, np.pi / 20)
                            add_xyz = self.rotate_origin(add_xyz - center, rot_noise)
                            add_xyz = add_xyz + center
                        arrow = g_center - center #网格和原实例重心的偏差
                        min_xyz_a = np.min(add_xyz, axis=0) + arrow #移至该地板块
                        max_xyz_a = np.max(add_xyz, axis=0) + arrow
                        mask_occ = point_xyz[:, 0] > min_xyz_a[0] #边界检测，
                        mask_occ = np.logical_and(mask_occ, point_xyz[:, 1] > min_xyz_a[1])
                        mask_occ = np.logical_and(mask_occ, point_xyz[:, 2] > min_xyz_a[2])
                        mask_occ = np.logical_and(mask_occ, point_xyz[:, 0] < max_xyz_a[0])
                        mask_occ = np.logical_and(mask_occ, point_xyz[:, 1] < max_xyz_a[1])
                        mask_occ = np.logical_and(mask_occ, point_xyz[:, 2] < max_xyz_a[2])
                        if np.sum(mask_occ) > 0: #去掉增强后的实例范围大于原先点云xyz坐标范围的点
                            occ_cls = point_label[mask_occ]
                            num_thing = np.sum(self.thing_class[occ_cls]) #属于thing类别的点的个数
                            if num_thing / add_xyz.shape[0] > self.thing_coincide_lb: #thing的重合度下界要求严格一点，nuscenes默认0.001
                                continue
                            elif (occ_cls.shape[0] - num_thing) / add_xyz.shape[0] > self.stuff_coincide_lb: #stuff类的重合度下界，默认0.05
                                continue
                        add_sem_l = self.thing_list[inst_i]
                        add_label = np.ones(shape=(points.shape[0],1), dtype=np.uint8) * add_sem_l #增加的点云的语义标签
                        
                        if self.dataset_name == "nuscenes":
                            num_inst_exist = (un_xyz_inst_l // 1000 == self.thing_list[inst_i]).sum() #与add的实例的类别相同的实例个数
                            add_inst_l = add_sem_l * 1000 + num_inst_exist + add_idx + 1
                        elif self.dataset_name == "SemanticKitti":
                            num_inst_exist = (un_xyz_inst_l & 0xFFFF == self.thing_list[inst_i]).sum()
                            add_inst_l = add_sem_l + ((num_inst_exist + add_idx + 1) << 16)
                        else:
                            raise NotImplementedError

                        add_inst = np.ones(shape=(points.shape[0],1), dtype=np.uint8) * add_inst_l
                        point_xyz = np.concatenate((point_xyz, add_xyz + arrow), axis=0)
                        point_label = np.concatenate((point_label, add_label), axis=0)
                        point_inst = np.concatenate((point_inst, add_inst), axis=0)
                        if point_feat is not None:
                            add_fea = points[:, 3:]
                            if len(point_feat.shape) == 1: point_feat = point_feat[..., np.newaxis]
                            if len(add_fea.shape) == 1: add_fea = add_fea[..., np.newaxis]
                            point_feat = np.concatenate((point_feat, add_fea), axis=0)

                        total_point_num += points.shape[0]
                        break_flag = True
                        break #？
                    if break_flag:
                        break
                if total_point_num > 5000: #最大的增强点的实例个数，不能超过3000 #可设置的参数
                    break

        if point_feat is not None:
            return point_xyz, point_label, point_inst, point_feat
        else:
            return point_xyz, point_label, point_inst

    def instance_flip(self, points, axis, center):
        flip_type = np.random.choice(4, 1)
        points = points[:] - center
        if flip_type == 0:
            # rotate 180 degree
            points = -points + center
        elif flip_type == 1:
            # flip over long axis
            a = axis[0][0]
            b = axis[0][1]
            flip_matrix = np.array([[b ** 2 - a ** 2, -2 * a * b], [2 * a * b, b ** 2 - a ** 2]])
            points = np.matmul(flip_matrix, np.transpose(points, (1, 0)))
            points = np.transpose(points, (1, 0)) + center
        elif flip_type == 2:
            # flip over short axis
            a = axis[1][0]
            b = axis[1][1]
            flip_matrix = np.array([[b ** 2 - a ** 2, -2 * a * b], [2 * a * b, b ** 2 - a ** 2]])
            points = np.matmul(flip_matrix, np.transpose(points, (1, 0)))
            points = np.transpose(points, (1, 0)) + center

        return points

    def rotate_origin(self, xyz, radians):
        'rotate a point around the origin'
        x = xyz[:, 0]
        y = xyz[:, 1]
        new_xyz = xyz.copy()
        new_xyz[:, 0] = x * np.cos(radians) + y * np.sin(radians)
        new_xyz[:, 1] = -x * np.sin(radians) + y * np.cos(radians)
        return new_xyz

class Instance_Augmentation():
    def __init__(self, instance_pkl_path, thing_list, datapath_pre='/data2/share/semantickitti', class_weight=None, random_flip = False,random_add = False,random_rotate = False,local_transformation = False):
        self.thing_list = thing_list
        # self.datapath_pre = datapath_pre
        if class_weight is not None:
            self.instance_weight = [class_weight[thing_class_num-1] for thing_class_num in thing_list]
            self.instance_weight = np.asarray(self.instance_weight) / np.sum(self.instance_weight)
        else:
            assert len(thing_list) > 0
            self.instance_weight = [1.0 / len(thing_list) for _ in thing_list]
        self.random_flip = random_flip
        self.random_add = random_add
        self.random_rotate = random_rotate
        self.local_transformation = local_transformation

        self.add_num = 5

        with open(instance_pkl_path, 'rb') as f:
            self.instance_path = pickle.load(f)

    # compared to Panoptic-Polarnet, we incorporate sweep aggregation and copy paste together(agg=)
    # @nb.jit('int32[:,:,:](int32[:,:,:],int32[:,:], int32)', nopython=True, cache=True, parallel=False)
    # TODO: want to use numba to acclerate
    def instance_aug(self, point_xyz, point_label, point_inst, point_feat = None, agg=None):
        """random rotate and flip each instance independently.

        Args:
            point_xyz: [N, 3], point location
            point_label: [N, 1], class label
            point_inst: [N, 1], instance label
        """        
        # random add instance to this scan
        if self.random_add:
            # choose which instance to add
            instance_choice = np.random.choice(len(self.thing_list),self.add_num,replace=True, p=self.instance_weight)
            uni_inst, uni_inst_count = np.unique(instance_choice,return_counts=True)
            add_idx = 1
            total_point_num = 0
            early_break = False
            for n, count in zip(uni_inst, uni_inst_count):
                # find random instance
                random_choice = np.random.choice(len(self.instance_path[self.thing_list[n]]),count)
                # add to current scan
                for idx in random_choice:
                    inst_info = self.instance_path[self.thing_list[n]][idx]
                    if len(inst_info.split("preprocess")) == 2:
                        path = os.path.join("../data/SemanticKitti/dataset", inst_info.split("preprocess")[1][1:]) # it is ugly
                    else:
                        path = inst_info
                    points = np.fromfile(path, dtype=np.float32).reshape((-1, 4))
                    
                    add_xyz = points[:,:3]
                    center = np.mean(add_xyz,axis=0)

                    # need to check occlusion
                    fail_flag = True
                    if self.random_rotate:
                        # random rotate
                        random_choice = np.random.random(20)*np.pi*2
                        for r in random_choice:
                            center_r = self.rotate_origin(center[np.newaxis,...],r)
                            # check if occluded
                            if self.check_occlusion(point_xyz,center_r[0]):
                                fail_flag = False
                                break
                            if (agg is not None) and (self.check_occlusion(agg, center_r[0])):
                                fail_flag = False
                                break
                        # rotate to empty space
                        if fail_flag: continue
                        add_xyz = self.rotate_origin(add_xyz,r)
                    else:
                        fail_flag = not self.check_occlusion(point_xyz,center)
                        if (agg is not None):    
                            fail_flag = fail_flag and (not self.check_occlusion(agg, center))
                                
                    if fail_flag: continue

                    add_label = np.ones((points.shape[0],1),dtype=np.uint8)*(self.thing_list[n])
                    add_inst = np.ones((points.shape[0],1),dtype=np.uint32)*(add_idx<<16)
                    point_xyz = np.concatenate((point_xyz,add_xyz),axis=0)
                    point_label = np.concatenate((point_label,add_label),axis=0)
                    point_inst = np.concatenate((point_inst,add_inst),axis=0)
                    if point_feat is not None:
                        add_fea = points[:,3:]
                        if len(add_fea.shape) == 1: add_fea = add_fea[..., np.newaxis]
                        point_feat = np.concatenate((point_feat,add_fea),axis=0)
                    add_idx +=1
                    total_point_num += points.shape[0]
                    if total_point_num>5000:
                        early_break=True
                        break
                # prevent adding too many points which cause GPU memory error
                if early_break: break

        # instance mask
        mask = np.zeros_like(point_label,dtype=bool)
        for label in self.thing_list:
            mask[point_label == label] = True

        # create unqiue instance list
        inst_label = point_inst[mask].squeeze()
        unique_label = np.unique(inst_label)
        num_inst = len(unique_label)

        
        for inst in unique_label:
            # get instance index
            index = np.where(point_inst == inst)[0]
            # skip small instance
            if index.size<10: continue
            # get center
            center = np.mean(point_xyz[index,:],axis=0)

            if self.local_transformation:
                # random translation and rotation
                point_xyz[index,:] = self.local_tranform(point_xyz[index,:],center)
            
            # random flip instance based on it center 
            if self.random_flip:
                # get axis
                long_axis = [center[0], center[1]]/(center[0]**2+center[1]**2)**0.5
                short_axis = [-long_axis[1],long_axis[0]]
                # random flip
                flip_type = np.random.choice(5,1)
                if flip_type==3:
                    point_xyz[index,:2] = self.instance_flip(point_xyz[index,:2],[long_axis,short_axis],[center[0], center[1]],flip_type)
            
            # 20% random rotate
            random_num = np.random.random_sample()
            if self.random_rotate:
                if random_num>0.8 and inst & 0xFFFF > 0:
                    random_choice = np.random.random(20)*np.pi*2
                    fail_flag = True
                    for r in random_choice:
                        center_r = self.rotate_origin(center[np.newaxis,...],r)
                        # check if occluded
                        if self.check_occlusion(np.delete(point_xyz, index, axis=0),center_r[0]):
                            fail_flag = False
                            break
                    if not fail_flag:
                        # rotate to empty space
                        point_xyz[index,:] = self.rotate_origin(point_xyz[index,:],r)

        if len(point_label.shape) == 1: point_label = point_label[..., np.newaxis]
        if len(point_inst.shape) == 1: point_inst = point_inst[..., np.newaxis]
        if point_feat is not None:
            return point_xyz,point_label,point_inst,point_feat
        else:
            return point_xyz,point_label,point_inst

    def instance_flip(self, points,axis,center,flip_type = 1):
        points = points[:]-center
        if flip_type == 1:
            # rotate 180 degree
            points = -points+center
        elif flip_type == 2:
            # flip over long axis
            a = axis[0][0]
            b = axis[0][1]
            flip_matrix = np.array([[b ** 2 - a ** 2, -2 * a * b], [2 * a * b, b ** 2 - a ** 2]])
            points = np.matmul(flip_matrix,np.transpose(points, (1, 0)))
            points = np.transpose(points, (1, 0))+center
        elif flip_type == 3:
            # flip over short axis
            a = axis[1][0]
            b = axis[1][1]
            flip_matrix = np.array([[b ** 2 - a ** 2, -2 * a * b], [2 * a * b, b ** 2 - a ** 2]])
            points = np.matmul(flip_matrix,np.transpose(points, (1, 0)))
            points = np.transpose(points, (1, 0))+center

        return points

    def check_occlusion(self,points,center,min_dist=2):
        'check if close to a point'
        if points.ndim == 1:
            dist = np.linalg.norm(points[np.newaxis,:]-center,axis=1)
        else:
            dist = np.linalg.norm(points-center,axis=1)
        return np.all(dist>min_dist)

    def rotate_origin(self,xyz,radians):
        'rotate a point around the origin'
        x = xyz[:,0]
        y = xyz[:,1]
        new_xyz = xyz.copy()
        new_xyz[:,0] = x * np.cos(radians) + y * np.sin(radians)
        new_xyz[:,1] = -x * np.sin(radians) + y * np.cos(radians)
        return new_xyz

    def local_tranform(self,xyz,center):
        'translate and rotate point cloud according to its center'
        # random xyz
        loc_noise = np.random.normal(scale = 0.25, size=(1,3))
        # random angle
        rot_noise = np.random.uniform(-np.pi/20, np.pi/20)

        xyz = xyz-center
        xyz = self.rotate_origin(xyz,rot_noise)
        xyz = xyz+loc_noise
        
        return xyz+center
