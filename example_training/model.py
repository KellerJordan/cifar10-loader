# https://github.com/libffcv/ffcv/blob/6c3be0cabf1485aa2b6945769dbd1c2d12e8faa7/examples/cifar/train_cifar.py#L130
import torch
from torch import nn

def create_model(w=1.0):

    class Mul(nn.Module):
        def __init__(self, weight):
            super().__init__()
            self.weight = weight
        def forward(self, x):
            return x * self.weight

    class Flatten(nn.Module):
        def forward(self, x):
            return x.view(x.size(0), -1)

    class Residual(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, x):
            return x + self.module(x)

    def conv_bn(channels_in, channels_out, kernel_size=3, stride=1, padding=1):
        return nn.Sequential(
                nn.Conv2d(channels_in, channels_out,
                          kernel_size=kernel_size, stride=stride, padding=padding,
                          bias=False),
                nn.BatchNorm2d(channels_out),
                nn.ReLU(inplace=True)
        )

    NUM_CLASSES = 10
    w0 = int(w*64)
    w1 = int(w*128)
    w2 = int(w*256)
    model = nn.Sequential(
        conv_bn(3, w0, kernel_size=3, stride=1, padding=1),
        conv_bn(w0, w1, kernel_size=5, stride=2, padding=2),
        Residual(nn.Sequential(conv_bn(w1, w1), conv_bn(w1, w1))),
        conv_bn(w1, w2, kernel_size=3, stride=1, padding=1),
        nn.MaxPool2d(2),
        Residual(nn.Sequential(conv_bn(w2, w2), conv_bn(w2, w2))),
        conv_bn(w2, w1, kernel_size=3, stride=1, padding=0),
        nn.AdaptiveMaxPool2d((1, 1)),
        Flatten(),
        nn.Linear(w1, NUM_CLASSES, bias=False),
        Mul(0.2)
    )
    model = model.to(memory_format=torch.channels_last)
    return model

