import torch
import torch.nn as nn
import torch.nn.functional as F


class IA_Layer(nn.Module):
    def __init__(self, channels, return_att=False):
        """
        ic: [64, 128, 256, 512]
        pc: [96, 256, 512, 1024]
        """
        super(IA_Layer, self).__init__()

        self.return_att = return_att
        self.ic, self.pc = channels
        rc = self.pc // 4
        self.conv1 = nn.Sequential(nn.Conv1d(self.ic, self.pc, 1),
                                   nn.BatchNorm1d(self.pc),
                                   nn.ReLU())
        # self.fc1 = nn.Linear(self.ic, rc)
        self.fc1 = nn.Sequential(
            nn.BatchNorm1d(self.ic),
            nn.ReLU(),
            nn.Linear(self.ic, rc)
        )
        self.fc2 = nn.Linear(self.pc, rc)
        self.fc3 = nn.Linear(rc, 1)

    def forward(self, img_feats, point_feats):
        """

        Args:
            img_feas: <Tensor, N, C> image_feature
            point_feas: <Tensor, N, C'> point_feature

        Returns:

        """
        img_feats_l = img_feats.contiguous()
        point_feats_l = point_feats.contiguous()
        # batch = img_feas.size(0)
        # img_feas_f = img_feas.transpose(1, 2).contiguous().view(-1, self.ic)  # BCN->BNC->(BN)C
        # point_feas_f = point_feas.transpose(1, 2).contiguous().view(-1, self.pc)  # BCN->BNC->(BN)C'
        # 将图像特征和点云特征映射成相同维度
        ri = self.fc1(img_feats_l)
        rp = self.fc2(point_feats_l)
        # 直接逐元素相加作为融合手段，基于假设：如果相同位置图像特征和点云特征比较相似，那么图像特征将有利于提高网络的performance
        att = torch.sigmoid(self.fc3(torch.tanh(ri + rp)))  # BNx1
        att = att.unsqueeze(1)
        att = att.view(1, 1, -1)  # B1N
        # print(img_feas.size(), att.size())
        img_feats_c = img_feats.unsqueeze(0).transpose(1, 2)
        img_feas_new = self.conv1(img_feats_c)
        # 依据图像特征和点云特征的相关程度筛选图像特征
        out = img_feas_new * att

        if self.return_att:
            return out, att
        else:
            return out, None


class Atten_Fusion_Conv(nn.Module):
    def __init__(self, inplanes_I, inplanes_P, outplanes, return_att=False):
        """
        inplanes_I: [64, 128, 256, 512]
        inplanes_P: [96, 256, 512, 1024]
        outplanes: [96, 256, 512, 1024]
        """
        super(Atten_Fusion_Conv, self).__init__()

        self.return_att = return_att

        self.ai_layer = IA_Layer(channels=[inplanes_I, inplanes_P], return_att=return_att)
        # self.fusion_layer = LI_Fusion_Layer(channels=[inplanes_I, inplanes_P])
        # self.conv1 = torch.nn.Conv1d(inplanes_P, outplanes, 1)
        self.conv1 = torch.nn.Conv1d(inplanes_P + inplanes_P, outplanes, 1)
        self.bn1 = torch.nn.BatchNorm1d(outplanes)

    def forward(self, point_features, img_features):
        """
        point_feature: 点云特征 [B, C, N]
        img_feature: 图像特征 [B, N, C]
        """

        img_features, att = self.ai_layer(img_features, point_features)  # [B, C, N]
        # print("img_features:", img_features.shape)
        point_feats = point_features.unsqueeze(0).transpose(1, 2)
        # 将筛选的图像特征与点云特征直接拼接
        fusion_features = torch.cat([point_feats, img_features], dim=1)
        fusion_features = F.relu(self.bn1(self.conv1(fusion_features)))
        fusion_features = fusion_features.squeeze(0).transpose(0, 1)

        if att is not None:
            return fusion_features, att
        else:
            return fusion_features


# @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
def Feature_Gather(feature_map, xy):
    """
    :param xy:(B,N,2), normalize to [-1,1], (width, height)
    :param feature_map:(B,C,H,W)
    :return:
    """

    # use grid_sample for this.
    # xy(B,N,2)->(B,1,N,2)
    xy = xy.unsqueeze(1)

    interpolate_feature = nn.functional.grid_sample(feature_map, xy, padding_mode='zeros', align_corners=True)  # (B,C,1,N)

    return interpolate_feature.squeeze(2)  # (B,C,N)


def Feature_Fetch(masks, pix_coord, imfeats):
    """

    Args:
        masks:
        pix_coord:
        imfeats: <Tensor, B, 6, C, H, W>

    Returns:

    """
    imfs = []
    for mask, coord, img in zip(masks, pix_coord, imfeats):
        mask = mask.cuda()
        imf = torch.zeros(size=(mask.size(1), img.size(1))).cuda()
        imf_list = Feature_Gather(img, coord.cuda()).permute(0, 2, 1)  # [6, N, C]
        # assert mask.size(0) == coord.size(0)
        for idx in range(mask.size(0)):
            imf[mask[idx]] = imf_list[idx, mask[idx], :]
        imfs.append(imf)
    return torch.cat(imfs, dim=0)