# baseline resnet blocks with expansion = 1. Bottleneck resnets have expansion = 4 (check architecture)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchsummary import summary

#################################################################


def make_layer(block, in_channels, channels, num_blocks, stride = 1, dilation = 1):

    strides = [stride] + [1]*(num_blocks - 1)
    # stride = 2, num_blocks = 4, then strides = [2,1,1,1]

    blocks = []
    for stride in strides :
        blocks.append(block(in_channels = in_channels, channels = channels, stride = stride, dilation = dilation))
        in_channels = block.expansion * channels

    layer = nn.Sequential(*blocks) # will unpack list of blocks as arguments
    return layer

#################################################################################

class BasicBlock(nn.Module):

    expansion = 1

    def __init__(self, in_channels, channels, stride = 1, dilation = 1):
        super(BasicBlock, self).__init__()

        out_channels = self.expansion * channels
        self.conv1 = nn.Conv2d(in_channels, channels, kernel_size = 3, stride = stride, padding = dilation, dilation = dilation, bias = False)
        self.bn1   = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size = 3, stride = 1, padding = dilation, dilation = dilation, bias = False)
        self.bn2   = nn.BatchNorm2d(channels)

        if (stride != 1) or (in_channels != out_channels):
            #
            #print("input and output channels are : ")
            #print(in_channels, out_channels)
            conv = nn.Conv2d(in_channels, out_channels, kernel_size = 1, stride = stride, bias = False)
            bn = nn.BatchNorm2d(out_channels)
            self.downsample = nn.Sequential(conv,bn)
        else :
            self.downsample = nn.Sequential()

    def forward(self, x):

        # if stride == 2 then h/2 and w/2 for the operations below

        # x has shape : batch_size * in_channels * h * w
        out = F.relu(self.bn1(self.conv1(x))) # batch_size * channels * h * w and  same size output when pad = (Filter_size - 1)/2
        out = self.bn2(self.conv2(out)) # batch_size * channels * h * w
        out = out + self.downsample(x) # batch_size * channels * h * w
        out = F.relu(out) # batch_size * channels * h * w
        return out

###########################################################################

# os8 is output stride 8 : (h/8 and w/8)

class Resnet_BasicBlock_OS8(nn.Module):

    def __init__(self, num_layers):
        super(Resnet_BasicBlock_OS8, self).__init__()

        if num_layers == 18 :
            resnet = models.resnet18()
            #resnet.load_state_dict(torch.load('pretrained_models/resnet18-5c106cde.pth'))
            self.resnet = nn.Sequential(*list(resnet.children())[:-4])
            num_blocks_layer_4 = 2
            num_blocks_layer_5 = 2

        elif num_layers == 34 :
            resnet = models.resnet34()
            #resnet.load_state_dict(torch.load('pretrained_models/resnet34-333f7ec4.pth'))
            self.resnet = nn.Sequential(*list(resnet.children())[:-4])

        else :
            raise Exception("number of layers must be 18 or 34")

        self.layer4 = make_layer(BasicBlock,in_channels=128, channels = 256, num_blocks=num_blocks_layer_4, stride = 1, dilation = 2)
        self.layer5 = make_layer(BasicBlock, in_channels=256, channels = 512, num_blocks=num_blocks_layer_5, stride = 1, dilation = 4)


    def forward(self, x):

        # x has shape batch_size * 3 * h * w
        out = self.resnet(x) # batch_size * 128 * h/8 * w/8
        out = self.layer4(out) # batch_size * 256 * h/8 * w/8
        out = self.layer5(out) # batch_size * 512 * h/8 * w/8

        return out


def ResNet18_OS8():
    return Resnet_BasicBlock_OS8(num_layers=18)

def ResNet34_OS8():
    return Resnet_BasicBlock_OS8(num_layers=34)

# a = ResNet18_OS8()
# print(a.children)
