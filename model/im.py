import torch
from torch import nn
from torch.nn import functional as F

from torchvision.models import resnet50


class ResNetFeatExtracotr(nn.Module):
    def __init__(self, n_feature):
        super().__init__()
        res_net = resnet50(pretrained=True)
        self.conv1 = res_net.conv1
        self.bn1 = res_net.bn1
        self.maxpool = res_net.maxpool

        self.layer1 = res_net.layer1
        self.layer2 = res_net.layer2
        self.layer3 = res_net.layer3
        self.layer4 = res_net.layer4

        self.conv_linear = nn.Conv2d(2048, n_feature, kernel_size=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        if self.training:
            x = x.detach()
            x.volatile=False
        #x = self.avgpool(x)
        x = self.conv_linear(x)
        return x


