import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

# written based on assumption that Resnet is 18 layer or 34 layer
# the input contains 512 channels
class ASPP(nn.Module):

    def __init__(self, num_classes):
        super(ASPP, self).__init__()

        self.num_classes = num_classes

        self.conv_1x1_1 = nn.Conv2d(512,256, kernel_size = 1)
        self.bn_conv_1x1_1 = nn.BatchNorm2d(256)
        self.conv_3x3_1 = nn.Conv2d(512, 256, kernel_size = 3, stride = 1, padding = 6, dilation = 6)
        self.bn_conv_3x3_1 = nn.BatchNorm2d(256)
        self.conv_3x3_2 = nn.Conv2d(512,256, kernel_size = 3, stride = 1, padding = 12, dilation = 12)
        self.bn_conv_3x3_2 = nn.BatchNorm2d(256)
        self.conv_3x3_3 = nn.Conv2d(512,256, kernel_size = 3, stride = 1, padding = 18, dilation = 18)
        self.bn_conv_3x3_3 = nn.BatchNorm2d(256)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # conv to use after concatenating

        self.conv_1x1_2 = nn.Conv2d(512, 256, kernel_size = 1)
        self.bn_conv_1x1_2 = nn.BatchNorm2d(256)

        self.conv_1x1_3 = nn.Conv2d(1280, 256, kernel_size = 1)
        self.bn_conv_1x1_3 = nn.BatchNorm2d(256)

        self.conv_1x1_4 = nn.Conv2d(256,self.num_classes,kernel_size = 1)


    def forward(self, feature_map):

        # feature_map : batch_size * 512 * h/8 * h/8

        feature_map_h = feature_map.size()[2] # h/8
        feature_map_w = feature_map.size()[3] # w/8
        # later used for upsampling

        out_1x1_1 = F.relu(self.bn_conv_1x1_1(self.conv_1x1_1(feature_map))) # batch_size * 256 * h/8 * h/8
        out_3x3_1 = F.relu(self.bn_conv_3x3_1(self.conv_3x3_1(feature_map))) # batch_size * 256 * h/8 * h/8
        out_3x3_2 = F.relu(self.bn_conv_3x3_2(self.conv_3x3_2(feature_map))) # batch_size * 256 * h/8 * h/8
        out_3x3_3 = F.relu(self.bn_conv_3x3_3(self.conv_3x3_3(feature_map))) # batch_size * 256 * h/8 * h/8

        out_pool = self.avg_pool(feature_map) # batch_size * 512 * 1 * 1
        out = F.relu(self.bn_conv_1x1_2(self.conv_1x1_2(out_pool))) # batch_size * 256 * 1 * 1
        out = F.upsample(out, size = (feature_map_h, feature_map_w), mode = 'bilinear') # batch_size * 256 * h/8 * h/8

        out = torch.cat([out_1x1_1, out_3x3_1, out_3x3_2, out_3x3_3, out], 1) # batch_size * 1280 * h/8 * w/8
        out = F.relu(self.bn_conv_1x1_3(self.conv_1x1_3(out))) # batch_size * 256 * h/8 * h/8
        out = self.conv_1x1_4(out) # batch_size * num_classes * h/8 * w/8

        return out








