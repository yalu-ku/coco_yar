import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torchsummary import summary
import torchsummary


class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()

        # 넓이와 높이를 줄일때 strid=2 로 조절(디멘젼 변경 가능)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
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
        output = F.relu(output)
        return output


class ResNet(nn.Module): # BasicBlock을 여러변 연결하는 형태
    def __init__(self, block, num_blocks, num_classes=91):
        super(ResNet, self).__init__()
        self.in_planes = 64
        # 초반에 하나의 conv layer로 dimension 바꾸어줌
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        # Basic block 당 conv layer 2개 * 2번 연결, 한 레이어 당 4개의 conv layer
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1) 
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512, num_classes) # FCL

    def _make_layer(self, block, planes, num_blocks, stride):
        # 첫번째 레이어에서만 넓이/높이 조절 가능, 나머지는 전부 stride=1(같은 넓이/높이)
        strides = [stride] + [1] * (num_blocks - 1) 
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes 
        return nn.Sequential(*layers)

    def forward(self, input):
        output = F.relu(self.bn1(self.conv1(input)))
        output = self.layer1(output)
        output = self.layer2(output)
        output = self.layer3(output)
        output = self.layer4(output)
        output = F.avg_pool2d(output, 4)
        output = output.view(output.size(0), -1)
        output = self.linear(output)
        return output # 91개의 클래스


def ResNet18():
    return ResNet(BasicBlock, [2,2,2,2])

model = ResNet18()
summary(model, (3, 256,256), device='cuda')

# net = ResNet18()
# net = net.to('cuda')
# cudnn.benchmark = True

# learning_rate = 0.1
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.0002)

# def train(epoch):
#     print('----- Train epoch : %d' % epoch)
#     net.train()
#     train_loss = 0
#     correct = 0
#     total = 0
#     for batch_idx, (inputs, targets) in enumerate(train_loader):
#         inputs, targets = inputs.to(device)
