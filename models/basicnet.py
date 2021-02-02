import torch as t
import torch.nn as nn


class basicnet(nn.Module):
    def __init__(self):
        super(basicnet, self).__init__()
        self.layer1 = nn.Sequential(nn.Conv2d(3, 64, 3, 1, 1, bias=False),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU(inplace=True))
        self.layer2 = nn.Sequential(nn.Conv2d(64, 128, 3, 1, 1, bias=False),
                                    nn.Conv2d(128, 128, 3, 2, 1, bias=False),
                                    nn.BatchNorm2d(128),
                                    nn.ReLU(inplace=True))
        self.layer3 = nn.Sequential(nn.MaxPool2d(2),
                                    nn.Linear(32, 10))

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        return out

if __name__ == '__main__':
    model = basicnet()
    input = t.randn((1, 3, 128, 128))
    # to onnx
    input_names = ['input']
    output_names = ['output']
    t.onnx.export(model, input, './basicnet.onnx',
                  input_names=input_names, output_names=output_names,
                  verbose=True, opset_version=11)
    out = model(input)
    print(out.shape)