import os
import numpy as np
import torch
import cv2

from typing import List

from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

CAM_CHANNELS = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']

VIEW_COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]

labels_mapping = {
    1: 0,
    5: 0,
    7: 0,
    8: 0,
    10: 0,
    11: 0,
    13: 0,
    19: 0,
    20: 0,
    0: 0,
    29: 0,
    31: 0,
    9: 1,
    14: 2,
    15: 3,
    16: 3,
    17: 4,
    18: 5,
    21: 6,
    2: 7,
    3: 7,
    4: 7,
    6: 7,
    12: 8,
    22: 9,
    23: 10,
    24: 11,
    25: 12,
    26: 13,
    27: 14,
    28: 15,
    30: 16
}

IDX2COLOR_16 = [(125, 125, 125),
                (112, 128, 144),  # barrier 蓝灰色 1
                (220, 20, 60),  # bicycle 玫红色 2
                (255, 127, 80),  # bus 浅橙色，和5，9两个类别有点撞色
                (255, 158, 0),  # car 橙黄色 4
                (233, 150, 70),  # construction_vehicle 工程车 浅一点的橙色 5
                (255, 61, 99),  # motorcycle 桃红色 6
                (0, 0, 230),  # pedestrian 蓝色 7
                (47, 79, 79),  # traffic_cone 锥形交通路标 灰绿色 8
                (255, 140, 0),  # trailer 拖车 橙色 9
                (255, 99, 71),  # truck 卡车 10
                (0, 207, 191),  # driveable_surface 蓝绿色
                (175, 0, 75),  # other_flat 紫红色
                (75, 0, 75),  # sidewalk 紫色
                (112, 180, 60),  # terrain 草绿色
                (222, 184, 135),  # manmade 土黄色
                (0, 175, 0)]  # vegetation 深绿色

SemKITTI_label_name_16 = {
    0: 'noise',
    1: 'barrier',
    2: 'bicycle',
    3: 'bus',
    4: 'car',
    5: 'construction_vehicle',
    6: 'motorcycle',
    7: 'pedestrian',
    8: 'traffic_cone',
    9: 'trailer',
    10: 'truck',
    11: 'driveable_surface',
    12: 'other_flat',
    13: 'sidewalk',
    14: 'terrain',
    15: 'manmade',
    16: 'vegetation',
}

MapSemKITTI2NUSC = {
    0: 0,
    1: 4,
    2: 2,
    3: 6,
    4: 10,
    5: 5,
    6: 7,
    7: 2,
    8: 6,
    9: 11,
    10: 12,
    11: 13,
    12: 12,
    13: 15,
    14: 15,
    15: 16,
    16: 10,
    17: 14,
    18: 15,
    19: 15
}


def draw_bar_chart(bar_val_list: List, bar_name_list: List, col_name_list: List, width_per_col=0.25, fig_save_path=None):
    """
    :param bar_val_list: <List[ndarray], [N,]; <ndarray, [C,]>> len表示每个bar有多少列数据, C表示bar的数量
    :param bar_name_list: <List[str], [C,]> 每个bar的标签
    :param col_name_list: <List[str], [N,]> 每个col的标签
    :param width_per_col: float 每个col的宽
    :param fig_save_path:
    :return:
    """
    if fig_save_path is not None:
        import matplotlib
        matplotlib.use('agg')
        import matplotlib.pyplot as plt
    else:
        import matplotlib.pyplot as plt

    col_per_bar = len(bar_val_list)
    color_per_col = ['yellowgreen', 'tomato', 'silver', 'c', 'b', 'm']
    bar_val_list_numpy = []
    for bar_val in bar_val_list:
        if isinstance(bar_val, np.ndarray):
            bar_val_list_numpy.append(bar_val)
        elif isinstance(bar_val, torch.Tensor):
            bar_val_list_numpy.append(bar_val.cpu().numpy())
        elif isinstance(bar_val, list):
            bar_val_list_numpy.append(np.array(bar_val))
        else:
            print("only accept bar_val of type ndarray, tensor or list")
            exit(-1)
    num_bar = bar_val_list_numpy[0].shape[0]
    base_x = np.arange(num_bar)
    for i, (val, col_name) in enumerate(zip(bar_val_list, col_name_list)):
        val = np.round(val, 2)
        plt.bar(base_x + i * width_per_col, val, width=width_per_col, label=col_name, fc=color_per_col[i])
    plt.legend()

    plt.xticks(base_x + width_per_col / 2, bar_name_list, rotation=45)

    if fig_save_path is not None:
        plt.savefig(fig_save_path)
        # print("figure save to", fig_save_path)
    else:
        plt.show()


def load_bin_file(bin_path: str) -> np.ndarray:
    """
    Loads a .bin file containing the labels.
    :param bin_path: Path to the .bin file.
    :return: An array containing the labels.
    """
    assert os.path.exists(bin_path), 'Error: Unable to find {}.'.format(bin_path)
    bin_content = np.fromfile(bin_path, dtype=np.uint8)
    assert len(bin_content) > 0, 'Error: {} is empty.'.format(bin_path)

    return bin_content


def visualize_pcd(xyz, **kwargs):
    """
    使用open3d渲染点云
    Args:
        xyz: <ndarray> [N, 3] 点云三维坐标xyz
        **kwargs: 可选参数
        1. predict <ndarray> [N,] 网络预测的点云标签, 第二维取值范围[0, num_class];
        2. target <ndarray> [N,] 点云标签真值
        3. view <ndarray> [N,] 每个点所在相机视野标签, 第二维取值范围[0,6)
        4. rgb <ndarray> [N, 3] 每个点的颜色, 取值范围[0, 255]
        5. select_inds <ndarray> bool标签[N, ]或者序号标签[npoint, ]
    Returns:

    """
    import open3d as o3d

    for k, v in kwargs.items():
        if isinstance(v, torch.Tensor):
            v = v.cpu().numpy()
        if k == "predict":
            predict_color = o3d.utility.Vector3dVector(np.array([IDX2COLOR_16[int(c % 17)] for c in v]) / 255.0)
            print("load predict, render with W")
        elif k == "target":
            gt_color = o3d.utility.Vector3dVector(np.array([IDX2COLOR_16[int(c % 17)] for c in v]) / 255.0)
            print("load target, render with Q")
        elif k == "view":
            view_color = o3d.utility.Vector3dVector(
                np.array([VIEW_COLORS[c] if c != -1 else (127, 127, 127) for c in v]) / 255.0)
        elif k == 'rgb':
            rgb_color = o3d.utility.Vector3dVector(v / 255.0)
        elif k == 'select_inds':
            s_color = np.ones((xyz.shape[0], 3), dtype=np.float32) / 2
            s_color[v, :] = np.array([1., 0., 0.])
            s_color = o3d.utility.Vector3dVector(s_color)

    if isinstance(xyz, torch.Tensor):
        xyz = xyz.cpu().numpy()

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz[:, :3])

    def render_gt_color_callback(viewer):
        if "target" in kwargs.keys():
            pcd.colors = gt_color
            viewer.update_geometry(pcd)
            print("render target")
        else:
            print("No ground truth color provided")

    def render_predict_color_callback(viewer):
        if "predict" in kwargs.keys():
            pcd.colors = predict_color
            viewer.update_geometry(pcd)
            print("render predict")
        else:
            print("No predict color provided")

    def render_view_color_callback(viewer):
        if "view" in kwargs.keys():
            pcd.colors = view_color
            viewer.update_geometry(pcd)
        else:
            print("No view color provided")

    def render_rgb_color_callback(viewer):
        if 'rgb' in kwargs.keys():
            pcd.colors = rgb_color
            viewer.update_geometry(pcd)
        else:
            print("No RGB color provided")

    def render_select_points_callback(viewer):
        if 'select_inds' in kwargs.keys():
            pcd.colors = s_color
            viewer.update_geometry(pcd)
        else:
            print("No select inds provided")

    viewer = o3d.visualization.VisualizerWithKeyCallback()
    viewer.create_window()
    opt = viewer.get_render_option()
    opt.background_color = np.asarray([1., 1., 1.])
    viewer.register_key_callback(ord("Q"), render_gt_color_callback)
    viewer.register_key_callback(ord("W"), render_predict_color_callback)
    viewer.register_key_callback(ord("V"), render_view_color_callback)
    viewer.register_key_callback(ord("R"), render_rgb_color_callback)
    viewer.register_key_callback(ord("S"), render_select_points_callback)
    viewer.add_geometry(pcd)
    viewer.run()
    viewer.destroy_window()


def visualize_img(image: np.ndarray, **kwargs):
    """
    使用Image可视化图像
    :param image: <np.ndarray[dtype=uint8], [H, W, 3]>
    :param kwargs:
    1. predict <np.ndarray, [H, W]> 标签
    2. points <np.ndarray, [N, 3]> N个点, 0,1 -> w,h; 2->label
    3. class_list <list, [T]> T个类别的类别列表，选择只展示这些类的点云投影。里面数值会%17.
    :return:
    """

    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy().astype(np.uint8)

    oh, ow, c = image.shape
    assert image.ndim == 3
    image = Image.fromarray(image).convert(mode='RGB')

    if len(kwargs) == 0:
        plt.imshow(image)
        plt.show()
    else:
        if 'class_list' in kwargs.keys():
            class_list = kwargs['class_list']
            if isinstance(class_list, int):
                class_list = [class_list]
            else:
                assert (isinstance(class_list, np.ndarray)) or (isinstance(class_list, List))
        else:
            class_list = None

        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.detach().cpu().numpy()
            if k == 'predict':
                h, w = v.shape
                image_resize = image
                if h != image.height or w != image.width:
                    from torchvision.transforms import transforms
                    trans = transforms.Resize(size=v.shape)
                    image_resize = trans(image)
                color = np.array([IDX2COLOR_16[c] for c in v.flatten()]).reshape((h, w, 3)).astype(np.uint8)
                color = Image.fromarray(color).convert(mode='RGB')
                mix = Image.blend(image_resize, color, alpha=0.25)
                plt.imshow(mix)
            elif k == 'point':
                co= v[:, :2]
                co[:, 0] = (co[:, 0] + 1.0) / 2 * (ow - 1.0)
                co[:, 1] = (co[:, 1] + 1.0) / 2 * (oh - 1.0)
                co = np.floor(co).astype(np.int32)
                if v.shape[-1] == 3:
                    l = v[:, 2]
                    l = l.astype(np.int32)
                    color = [IDX2COLOR_16[c % 17] for c in l.flatten()]
                    if class_list is not None:
                        mask = np.zeros(v.shape[0], dtype=np.bool)
                        for cls in class_list:
                            mask = np.logical_or(mask, l == cls)
                        co = co[mask, :]
                        masked_color = []
                        for idx, flag in enumerate(mask):
                            if flag:
                                masked_color.append(color[idx])
                        color = masked_color
                elif v.shape[-1] == 5:
                    color = v[:, 2:].astype(np.uint8)
                imagedraw = ImageDraw.Draw(image)
                rad = 1.1
                for (x, y), c in zip(co, color):
                    imagedraw.ellipse(xy=[x - rad, y - rad, x + rad, y + rad], fill=tuple(c))
                plt.imshow(image)
            elif k == 'superpixel':
                h, w = v.shape
                v = v.astype(np.int32)
                # color = np.array([IDX2COLOR_16[1:][c % 16] for c in v.flatten()]).reshape((h, w, 3)).astype(np.uint8)
                color = np.array([IDX2COLOR_16[c % 17] for c in v.flatten()]).reshape((h, w, 3)).astype(np.uint8)
                color = Image.fromarray(color).convert(mode='RGB')
                mix = Image.blend(image, color, alpha=0.25)
                plt.imshow(mix)
            elif k == 'heatmap':
                heatmap = Image.fromarray(np.array(plt.cm.jet(v)*255).astype(np.uint8)).convert('RGB')
                mix = Image.blend(image, heatmap, 0.25)
                plt.imshow(mix)

            fig_save_path = kwargs.get('fig_save_path', None)
            if fig_save_path is not None:
                plt.savefig(fig_save_path)
            else:
                plt.show()


def visualize_bev(xyz, mass_centers, axis_aligned_centers):
    # 设置鸟瞰图范围
    side_range = (-40, 40)  # 左右距离
    fwd_range = (-40, 40)  # 后前距离

    x_points = np.concatenate([xyz[:, 0], mass_centers[:, 0]])
    y_points = np.concatenate([xyz[:, 1], mass_centers[:, 1]])
    z_points = np.concatenate([xyz[:, 2], np.zeros((len(mass_centers)))])

    # x_points = xyz[:, 0]
    # y_points = xyz[:, 1]
    # z_points = xyz[:, 2]

    # 获得区域内的点
    f_filt = np.logical_and(x_points > fwd_range[0], x_points < fwd_range[1])
    s_filt = np.logical_and(y_points > side_range[0], y_points < side_range[1])
    filter = np.logical_and(f_filt, s_filt)
    indices = np.argwhere(filter).flatten()
    x_points = x_points[indices]
    y_points = y_points[indices]
    z_points = z_points[indices]

    res = 0.05  # 分辨率0.05m
    x_img = (-y_points / res).astype(np.int32)
    y_img = (-x_points / res).astype(np.int32)
    # 调整坐标原点
    x_img -= int(np.floor(side_range[0]) / res)
    y_img += int(np.floor(fwd_range[1]) / res)

    # 填充像素值
    height_range = (-2, 0.5)
    pixel_value = np.clip(a=z_points, a_max=height_range[1], a_min=height_range[0])  # 根据z坐标决定颜色程度

    def scale_to_255(a, min, max, dtype=np.uint8):
        return ((a - min) / float(max - min) * 255).astype(dtype).repeat(3).reshape(-1, 3)

    pixel_value = scale_to_255(pixel_value, height_range[0], height_range[1])

    inst_num = len(mass_centers)
    # pixel_value[-2*inst_num:-inst_num] = np.array([0, 0, 255])
    pixel_value[-inst_num:] = np.array([0, 0, 255])

    # 创建图像数组
    x_max = 1 + int((side_range[1] - side_range[0]) / res)
    y_max = 1 + int((fwd_range[1] - fwd_range[0]) / res)
    im = np.zeros([y_max, x_max, 3], dtype=np.uint8)
    im[y_img, x_img] = pixel_value

    # Image
    im2 = Image.fromarray(im)
    im2.save('/data1/zwj/' + str(len(xyz) // inst_num) + '_mass.png')
    print('mass')
    # print(np.stack([y_img[-2*inst_num:-inst_num], x_img[-2*inst_num:-inst_num]], axis=1))
    # print('axis')
    print(np.stack([y_img[-inst_num:], x_img[-inst_num:]], axis=1))
    print()
    # im2.show()

    # plt
    # plt.imshow(im)
    # plt.show()
    # plt.imsave('/data1/zwj/bev.png', im)


def flow_to_img(flow, normalize=True):
    """Convert flow to viewable image, using color hue to encode flow vector orientation, and color saturation to
    encode vector length. This is similar to the OpenCV tutorial on dense optical flow, except that they map vector
    length to the value plane of the HSV color model, instead of the saturation plane, as we do here.
    Args:
        flow: optical flow
        normalize: Normalize flow to 0..255
    Returns:
        img: viewable representation of the dense optical flow in RGB format
    Ref:
        https://github.com/philferriere/tfoptflow/blob/33e8a701e34c8ce061f17297d40619afbd459ade/tfoptflow/optflow.py
    """
    hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
    flow_magnitude, flow_angle = cv2.cartToPolar(flow[..., 0].astype(np.float32), flow[..., 1].astype(np.float32))

    # A couple times, we've gotten NaNs out of the above...
    nans = np.isnan(flow_magnitude)
    if np.any(nans):
        nans = np.where(nans)
        flow_magnitude[nans] = 0.

    # Normalize
    hsv[..., 0] = flow_angle * 180 / np.pi / 2
    if normalize is True:
        hsv[..., 1] = cv2.normalize(flow_magnitude, None, 0, 255, cv2.NORM_MINMAX)
    else:
        hsv[..., 1] = flow_magnitude
    hsv[..., 2] = 255
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    return img


if __name__ == '__main__':
    bar_val_list = []
    for i in range(3):
        bar_val_list.append(np.random.random(17, )[1:])
    draw_bar_chart(bar_val_list=bar_val_list, bar_name_list=list(SemKITTI_label_name_16.values())[1:],
                   col_name_list=['A', 'B', 'C'], fig_save_path='./debug.png')
