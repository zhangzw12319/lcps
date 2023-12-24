import os
import numpy as np
# import mlcrate as mlc
from functools import partial
import pickle
import yaml

from nuscenes import NuScenes as NuScenes_devkit
from utils.visualize_utils import visualize_pcd

from tqdm import tqdm

DATASET_ROOT = '/data2/share/nuscenes'
SPLIT = 'train'
DATABASE_SAVE_DIR = os.path.join('/data/cyj/Code/ppn_th68/data/nuscenes', 'inst_database_' + SPLIT)
INST_DBINFO_PKL_SAVE_PATH = os.path.join('/data/cyj/Code/ppn_th68/data/nuscenes', 'inst_database_train_info.pkl')

# ppn_th68中，原先的noise 0类移到第17类，而0类用于表示没有point的voxel
with open("nuscenes.yaml", 'r') as stream:
    nuscenesyaml = yaml.safe_load(stream)
labels_mapping = nuscenesyaml['learning_map']
labels17_id2name = nuscenesyaml['labels_17']
# labels_mapping = {
#     1: 17,
#     5: 17,
#     7: 17,
#     8: 17,
#     10: 17,
#     11: 17,
#     13: 17,
#     19: 17,
#     20: 17,
#     0: 17,
#     29: 17,
#     31: 17,
#     9: 1,
#     14: 2,
#     15: 3,
#     16: 3,
#     17: 4,
#     18: 5,
#     21: 6,
#     2: 7,
#     3: 7,
#     4: 7,
#     6: 7,
#     12: 8,
#     22: 9,
#     23: 10,
#     24: 11,
#     25: 12,
#     26: 13,
#     27: 14,
#     28: 15,
#     30: 16
# }

THING_LIST = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
MIN_INST_POINT = 6

INST_DBINFO_PKL = dict()
for c_i in THING_LIST:
    INST_DBINFO_PKL[labels17_id2name[c_i]] = []

def prepare_file_list():
    nusc = NuScenes_devkit(dataroot=DATASET_ROOT, version='v1.0-trainval', verbose=True)
    if SPLIT == "train":
        select_idx = np.load('./data/nuscenes/nuscenes_train_official.npy')
        sample_list = [nusc.sample[i] for i in select_idx]
    elif SPLIT == "val":
        select_idx = np.load("./data/nuscenes/nuscenes_val_official.npy")
        sample_list = [nusc.sample[i] for i in select_idx]
    else:
        print('%s not support' % SPLIT)
        exit(-1)
    pcd_info_list = []
    for idx, sample in enumerate(tqdm(sample_list)):
        info = {}
        sample = nusc.sample[idx]
        lidar_token = sample["data"]["LIDAR_TOP"]
        lidar_channel = nusc.get("sample_data", lidar_token)
        lidar_path = os.path.join(nusc.dataroot, lidar_channel["filename"])
        lidar_label_path = os.path.join(nusc.dataroot, nusc.get("lidarseg", lidar_token)["filename"])
        panoptic_labels_path = os.path.join(nusc.dataroot, nusc.get('panoptic', lidar_token)['filename'])
        info['lidar_token'] = lidar_token
        info['path'] = lidar_path
        info['sem_label_path'] = lidar_label_path
        info['pano_label_path'] = panoptic_labels_path
        pcd_info_list.append(info)

    return pcd_info_list


def process_one_sequences(info: dict, save_file=True):

    global INST_DBINFO_PKL

    lidar_token = info['lidar_token']
    lidar_path = info['path']
    sem_label_path = info['sem_label_path']
    pano_label_path = info['pano_label_path']
    point_xyzie = np.fromfile(lidar_path, dtype=np.float32).reshape([-1, 5])
    sem_label = np.fromfile(sem_label_path, dtype=np.uint8).reshape([-1, 1])
    sem_label = np.vectorize(labels_mapping.__getitem__)(sem_label).flatten()
    panoptic_label = np.load(pano_label_path)['data']
    if not save_file:
        visualize_pcd(point_xyzie[:, :3], predict=sem_label, target=panoptic_label)

    for thing_id in THING_LIST:
        thing_mask = np.zeros_like(sem_label, dtype=bool)  # 每个类别一个mask
        thing_mask[sem_label == thing_id] = True  # 所有属于thing_id的类
        panoptic_label_thing = panoptic_label[thing_mask]  # 全景标签存在小于 2^16 的异常点
        unique_inst_label = np.unique(panoptic_label_thing)

        thing_name = labels17_id2name[thing_id]
        for uq_inst_label in unique_inst_label:  # 对于每一个实例
            index = np.where(panoptic_label == uq_inst_label)[0]
            if index.shape[0] < MIN_INST_POINT:  # 如果
                continue
            if np.sum(panoptic_label[index]) == 0:
                continue
            if save_file:
                dir_path = os.path.join(DATABASE_SAVE_DIR, labels17_id2name[thing_id])
                if not os.path.exists(dir_path):
                    try:
                        os.makedirs(dir_path)
                    except OSError:
                        print(f'Error occurred when creating ground truth mask dir "{dir_path}".')
                    else:
                        print(f'dir created "{dir_path}.')
                file_path = os.path.join(dir_path, '%s_%s_%s.bin' % (str(lidar_token), str(thing_name), str(uq_inst_label)))
                inst_points = point_xyzie[index, :]
                if not os.path.exists(file_path):
                    inst_points.tofile(file_path)
                    INST_DBINFO_PKL[thing_name].append(file_path)
            else:
                inst_points = point_xyzie[index, :]
                inst_sem_label = sem_label[index]
                inst_pano_label = panoptic_label[index]
                visualize_pcd(inst_points[:, :3], predict=inst_sem_label, target=inst_pano_label)

# def main_eval_saved_file():
#
#     def _load_image(filepath: str):
#         return np.array(Image.open(filepath).convert('RGB'))
#
#     def _load_superpixel(name_list: list[str]):
#         sp_list = []
#         for name in name_list:
#             token_list = name.split('/')
#             seq, im_name = token_list[-3], token_list[-1]
#             sp_name = os.path.join(SAVE_DIR, SPLIT, str(seq), 'seeds', im_name[:-5] + '.npy')
#             sp_list.append(np.load(sp_name))
#         return sp_list
#
#     keyframe_path = os.path.join(DATASET_ROOT, SPLIT, 'keyframes.txt')
#     with open(keyframe_path, 'r') as f:
#         keyframe_list = f.read().splitlines()
#     for path in keyframe_list:
#         im_list, name_list = _load_image(seq_path=path)
#         sp_list = _load_superpixel(name_list)
#         for im, sp in zip(im_list, sp_list):
#             visualize_img(im, superpixel=sp)
#         # process_one_sequences(path, save_file=True)


def main_save_npy_multiprocess():
    name_token_pair = prepare_file_list()
    pool = mlc.SuperPool(24)
    pool.map(partial(process_one_sequences, save_file=True), name_token_pair, description='process nusc instdb %s' % str(SPLIT))
    # partial(process_one_sequences, save_file=True), name_token_pair, description = 'process nusc instdb %s' % str(SPLIT)
    for k, v in INST_DBINFO_PKL.items():
        print(f'load {len(v)} {k} database infos')
    with open(INST_DBINFO_PKL_SAVE_PATH, 'wb') as f:
        pickle.dump(INST_DBINFO_PKL, f, protocol=pickle.HIGHEST_PROTOCOL)
    print('pkl saved at', INST_DBINFO_PKL_SAVE_PATH)


if __name__ == '__main__':
    name_token_pair = prepare_file_list()
    for pair in tqdm(name_token_pair):
        process_one_sequences(pair, save_file=True)
    for k, v in INST_DBINFO_PKL.items():
        print(f'load {len(v)} {k} database infos')
    with open(INST_DBINFO_PKL_SAVE_PATH, 'wb') as f:
        pickle.dump(INST_DBINFO_PKL, f)
    print('pkl saved at', INST_DBINFO_PKL_SAVE_PATH)

# if __name__ == '__main__':
#     main_eval_saved_file()

# if __name__ == '__main__':
#     main_save_npy_multiprocess()

