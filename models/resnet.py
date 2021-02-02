import torchvision.models as m
import torch as t
import torch.nn as nn
from torch.nn import functional as F

class resBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1, shortcut=None):
        super(resBlock, self).__init__()
        self.left = nn.Sequential(nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride,padding=1, bias=False),
                                  nn.BatchNorm2d(outchannel),
                                  nn.ReLU(inplace=True),
                                  nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1,bias=False),
                                  nn.BatchNorm2d(outchannel))
        self.right = shortcut

    def forward(self, x):
        out = self.left(x)
        residual = x if self.right is None else self.right(x)
        out += residual
        return F.relu(out)

class ResNet34(nn.Module):
    def __init__(self, num_classes=2):
        super(ResNet34, self).__init__()
        self.model_name = 'resnet34'
        self.pre = nn.Sequential(nn.Conv2d(3, 64, 7, 2, 3, bias=False),
                                 nn.BatchNorm2d(64),
                                 nn.ReLU(inplace=True),
                                 nn.MaxPool2d(3, 2, 1))
        self.layer1 = self._make_layer(64, 128, 3)
        self.layer2 = self._make_layer(128, 256, 4, stride=2)
        self.layer3 = self._make_layer(256, 512, 6, stride=2)
        self.layer4 = self._make_layer(512, 512, 3, stride=2)

        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, inchannel, outchannel, block_num, stride=1):
        short_cut = nn.Sequential(nn.Conv2d(inchannel, outchannel, 1, stride, bias=False),
                                  nn.BatchNorm2d(outchannel))
        layers = []
        layers.append(resBlock(inchannel, outchannel, stride, short_cut))

        for i in range(1, block_num):
            layers.append(resBlock(outchannel, outchannel))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.pre(x)
        # print(out.shape)
        out = self.layer1(out)
        # print(out.shape)
        out = self.layer2(out)
        # print(out.shape)
        out = self.layer3(out)
        # print(out.shape)
        out = self.layer4(out)
        # print(out.shape)
        out = F.avg_pool2d(out, 8)
        # print(out.shape)
        out = out.view(out.size(0), -1)
        # print(out.shape)
        out = self.fc(out)
        # print(out.shape)
        return out


if __name__ == '__main__':
    model = ResNet34()
    input = t.randn((1, 3, 256, 256))
    #to onnx
    # input_names = ['input']
    # output_names = ['output']
    # t.onnx.export(model, input, './resnet34.onnx',
    #               input_names=input_names, output_names=output_names,
    #               verbose=True, opset_version=11)
    out = model(input)
