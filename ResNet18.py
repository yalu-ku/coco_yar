import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim as optim
import os

class BasicBlock(nn.Moudle):
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()

        # 넓이와 높이를 줄일때 strid=2 로 조절
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential() # mapping
        if stride != 1: # 입력값과 dimension 다르다는 뜻, 동일한 차원으로 input이 connection될 수 있도록
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, input):
        output = F.relu(self.bn1(self.conv1(input)))
        output = self.bn2(self.conv2(output))
        output += self.shortcut(input) # == skip connection *
        return output

