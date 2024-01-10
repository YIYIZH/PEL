
from torch import nn
import torch
from torchvision import models, transforms, datasets
import torchvision.models.detection.backbone_utils as backbone_utils
from torchvision.models import densenet121

class MainModel(nn.Module):
    def __init__(self, num_cls, arch, pretained=True):
        super(MainModel, self).__init__()
        self.num_classes = num_cls
        self.backbone_arch = arch
        print(self.backbone_arch)
        self.model = densenet121(pretrained=pretained)
        self.model = nn.Sequential(*list(self.model.children())[:-1])
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=1)
        self.fc = nn.Linear(1024, self.num_classes, bias=False)

    def forward(self, x):
        x = self.model(x) #[8,1024,14,14]
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        out = self.fc(x)
        return x, out

def DenseNet121(num_cls, pretrained=True):
    return MainModel(num_cls, 'densenet121', pretrained)