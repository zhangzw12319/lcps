
import torch
import torch.nn as nn
import torch.nn.functional as F
# import torch.utils.model_zoo as model_zoo



def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

upsample = lambda x, size: F.interpolate(x, size, mode='bilinear', align_corners=True)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, efficient=True, use_bn=True):
        super(BasicBlock, self).__init__()
        self.use_bn = use_bn
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes) if self.use_bn else None
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes) if self.use_bn else None
        self.downsample = downsample
        self.stride = stride
        self.efficient = efficient

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        relu = self.relu(out)

        return relu, out


class SwiftNetResNet(nn.Module):
    def __init__(self, block, layers, num_features=(128, 128, 128), k_up=3, efficient=True, use_bn=True,
                 spp_grids=(8, 4, 2, 1), spp_square_grid=False, **kwargs):
        super(SwiftNetResNet, self).__init__()
        self.inplanes = 64
        self.efficient = efficient
        self.use_bn = use_bn
        self.img_cs = [64, 64, 128, 256, 128]
        self.im_cr = 1.0
        self.img_cs = [int(c * self.im_cr) for c in self.img_cs]
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64) if self.use_bn else lambda x: x
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.inplanes_list = []
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.inplanes_list.append(self.inplanes)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.inplanes_list.append(self.inplanes)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.inplanes_list.append(self.inplanes)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        num_levels = 3
        self.spp_size = kwargs.get('spp_size', num_features)[0]
        bt_size = self.spp_size

        level_size = self.spp_size // num_levels

        self.spp = SpatialPyramidPooling(self.inplanes, num_levels, bt_size=bt_size, level_size=level_size,
                                         out_size=num_features[0], grids=spp_grids, square_grid=spp_square_grid,
                                         bn_momentum=0.024 / 2, use_bn=self.use_bn)

        self.build_decoder = kwargs.get('build_decoder', True)
        if self.build_decoder:
            upsamples = []
            upsamples += [_Upsample(num_features[1], self.inplanes_list[0], num_features[2], use_bn=self.use_bn, k=k_up)]
            upsamples += [_Upsample(num_features[0], self.inplanes_list[1], num_features[1], use_bn=self.use_bn, k=k_up)]
            upsamples += [_Upsample(num_features[0], self.inplanes_list[2], num_features[0], use_bn=self.use_bn, k=k_up)]
            self.upsample = nn.ModuleList(list(reversed(upsamples)))
        else:
            self.upsample = None

        self.random_init = [self.spp, self.upsample]

        self.num_features = num_features[-1]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):

        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            layers = [nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False)]
            if self.use_bn:
                layers += [nn.BatchNorm2d(planes * block.expansion)]
            downsample = nn.Sequential(*layers)
        layers = [block(self.inplanes, planes, stride, downsample, efficient=self.efficient, use_bn=self.use_bn)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers += [block(self.inplanes, planes, efficient=self.efficient, use_bn=self.use_bn)]

        return nn.Sequential(*layers)

    def forward_resblock(self, x, layers):
        skip = None
        for l in layers:
            x = l(x)
            if isinstance(x, tuple):
                x, skip = x
        return x, skip

    def forward_down(self, image):
        x = self.conv1(image)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        features = []
        x, skip = self.forward_resblock(x, self.layer1)
        features += [skip]
        x, skip = self.forward_resblock(x, self.layer2)
        features += [skip]
        x, skip = self.forward_resblock(x, self.layer3)
        features += [skip]

        x, skip = self.forward_resblock(x, self.layer4)

        features += [self.spp.forward(skip)]

        return features

    def forward_up(self, features):
        assert self.build_decoder
        features = features[::-1]
        x = features[0]
        for skip, up in zip(features[1:], self.upsample):
            x = up(x, skip)
        return x

    def forward(self, image):
        return self.forward_up(self.forward_down(image))


class SpatialPyramidPooling(nn.Module):
    """
        SPP module is little different from ppm by inserting middle level feature to save the computation and  memory.
    """

    def __init__(self, num_maps_in, num_levels, bt_size=512, level_size=128, out_size=128,
                 grids=(6, 3, 2, 1), square_grid=False, bn_momentum=0.1, use_bn=True):

        super(SpatialPyramidPooling, self).__init__()
        self.grids = grids
        self.square_grid = square_grid
        self.spp = nn.Sequential()
        self.spp.add_module('spp_bn',
                            _BNReluConv(num_maps_in, bt_size, k=1, bn_momentum=bn_momentum, batch_norm=use_bn))
        num_features = bt_size
        final_size = num_features
        for i in range(num_levels):
            final_size += level_size
            self.spp.add_module('spp' + str(i),
                                _BNReluConv(num_features, level_size, k=1, bn_momentum=bn_momentum, batch_norm=use_bn))
        self.spp.add_module('spp_fuse',
                            _BNReluConv(final_size, out_size, k=1, bn_momentum=bn_momentum, batch_norm=use_bn))

    def forward(self, x):
        levels = []
        target_size = x.size()[2:4]  # (H, W)

        ar = target_size[1] / target_size[0]

        x = self.spp[0].forward(x)
        levels.append(x)
        num = len(self.spp) - 1

        for i in range(1, num):
            if not self.square_grid:
                grid_size = (self.grids[i - 1], max(1, round(ar * self.grids[i - 1])))
                x_pooled = F.adaptive_avg_pool2d(x, grid_size)
            else:
                x_pooled = F.adaptive_avg_pool2d(x, self.grids[i - 1])
            level = self.spp[i].forward(x_pooled)

            level = upsample(level, target_size)
            levels.append(level)
        x = torch.cat(levels, 1)
        x = self.spp[-1].forward(x)
        return x


class _BNReluConv(nn.Sequential):
    def __init__(self, num_maps_in, num_maps_out, k=3, batch_norm=True, bn_momentum=0.1, bias=False, dilation=1):
        super(_BNReluConv, self).__init__()
        if batch_norm:
            self.add_module('norm', nn.BatchNorm2d(num_maps_in, momentum=bn_momentum))
        self.add_module('relu', nn.ReLU(inplace=batch_norm is True))
        padding = k // 2
        self.add_module('conv', nn.Conv2d(num_maps_in, num_maps_out,
                                          kernel_size=k, padding=padding, bias=bias, dilation=dilation))


class _Upsample(nn.Module):
    def __init__(self, num_maps_in, skip_maps_in, num_maps_out, use_bn=True, k=3):
        super(_Upsample, self).__init__()
        # print(f'Upsample layer: in = {num_maps_in}, skip = {skip_maps_in}, out = {num_maps_out}')
        self.bottleneck = _BNReluConv(skip_maps_in, num_maps_in, k=1, batch_norm=use_bn)
        self.blend_conv = _BNReluConv(num_maps_in, num_maps_out, k=k, batch_norm=use_bn)

    def forward(self, x, skip):
        skip = self.bottleneck.forward(skip)
        skip_size = skip.size()[2:4]
        x = upsample(x, skip_size)
        x = x + skip
        x = self.blend_conv.forward(x)
        return x


def SwiftNetRes18(num_feature=(128, 128, 128),pretrained_path=None):
    """Constructs a ResNet-18 model.
    Args:
        num_feature
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = SwiftNetResNet(BasicBlock, [2, 2, 2, 2], num_feature, build_decoder=True)
    # pretrained_path = '/home/group3d/yq/CVPR2022_ID2833_Code/pretrain/resnet18-5c106cde.pth'
    if pretrained_path is not None:
        print("load pretrained weigth from", pretrained_path)
        model.load_state_dict(torch.load(pretrained_path, map_location=torch.device('cpu')), strict=False)
    else:
        print("train swiftnet from sketch")
    return model