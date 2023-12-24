import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torchvision import models


class resnet(nn.Module):
    def __init__(self,global_fusion):
        super(resnet, self).__init__()
        self.model = models.resnet50() # pretrained=True
        self.model.load_state_dict(torch.load(global_fusion))
        for p in self.model.parameters():
            p.requires_grad = False
        for p in self.model.layer4.parameters():
            p.requires_grad = True

        self.fc1 = nn.Linear(2048, 1024)
        self.bn = nn.BatchNorm1d(1024)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(1024, 512)

    def forward(self, x):
        x = x.permute(0,1,4,2,3).contiguous()
        x_shape = x.shape
        x = x.reshape(x_shape[0]*x_shape[1],x_shape[2],x_shape[3],x_shape[4]).contiguous() # 448*800
        with torch.no_grad():
            x = self.model.conv1(x)
            x = self.model.bn1(x)
            x = self.model.relu(x)
            x = self.model.maxpool(x)
            x = self.model.layer1(x)
            x = self.model.layer2(x)
            x = self.model.layer3(x)  # 28*50
        x = self.model.layer4(x)  # 14*25
        x = self.model.avgpool(x)

        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.fc2(x)
        return x
