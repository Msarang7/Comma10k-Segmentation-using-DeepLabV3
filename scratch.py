import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchsummary import summary
from aspp import ASPP
from resnet import BasicBlock

################################################################################################
# gray_class_values = [41,76,90,124,161]
# classes = [0,1,2,3,4]
# mask = cv2.imread('masks/0000_0085e9e41513078a_2018-08-19--13-26-08_11_864.png',0)
# mask = cv2.resize(mask, (437,582),interpolation = cv2.INTER_NEAREST)
# a,b = np.unique(mask, return_counts = True)
# print(a)
# print(b)
# class_values = [41,76,90,124,161]
# for i in range(len(classes)):
#     mask[mask == gray_class_values[i]] = classes[i]
# a,b = np.unique(mask, return_counts = True)
# print(a)
# print(b)



####################################################################################################

# resnet = models.resnet18()
# resnet.load_state_dict(torch.load('pretrained_models/resnet18-5c106cde.pth'))
# resnet = nn.Sequential(*list(resnet.children())[:-4])

#####################################################################################

# loss = nn.CrossEntropyLoss()
# input = torch.randn(4,5,256,256, requires_grad=True) # batch_size * num_classes * h * w
# target = torch.empty(4,256,256, dtype=int).random_(5) # batch_size * h * w
# print("input")
# print(input)
# print("target")
# print(target)
#
# print(torch.argmax(input,1))
# print(np.argmax(input.detach().numpy(),1))
#
# output = loss(input, target)
# output.backward()
#
# print("output")
# print(output)

###############################################################################################

# checking summary of model
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#model = BasicBlock(3,256) # first block
#summary(model, (3,256,256)) # checking for first block. in_channels = 3 and out_channels = 256

#model = BasicBlock(3,256) # after first block
#summary(model, (3,256,256)) # checking after first block when in_channels = out_channels

################################################################################################

# checking summary of model ASPP
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
# model = ASPP(5)
# summary(model, (512,64,64))

####################################################################################

# a = np.load('weights.npy', allow_pickle = True)
# print(a)

#####################################################################################

# primary device
# if torch.cuda.is_available():
#     device = torch.device('cuda')
#     print("gpu detected")
# else :
#     device = torch.device('cpu')
#     print("gpu undetected")

########################################################################

img = plt.imread('scartch_test.png')
# print(img)
# plt.imshow(img)
# plt.show()

for c in img:
    print(c)