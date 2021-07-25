# encode the labels into

import numpy as np
import random
import pandas as pd
from torchvision import utils
from torch.utils.data import DataLoader, Dataset
import cv2
import torch
import matplotlib.pyplot as plt
import os
import albumentations as A
import warnings
warnings.filterwarnings('ignore')

# getting the names of images and masks for training and validation
def get_names():


    # getting names of masks

    mask_names = os.listdir('masks')
    train_masks = ['masks/'+ str(x) for x in mask_names if not x.endswith('9.png')]
    val_masks = ['masks/'+str(x) for x in mask_names if x.endswith('9.png')]


    # getting names of images

    image_names = os.listdir('imgs')
    train_image_names = ['imgs/'+str(x) for x in image_names if not x.endswith('9.png')]
    val_image_names = ['imgs/'+str(x) for x in image_names if x.endswith('9.png')]

    # saving as csv files
    df = pd.DataFrame()
    df['images'] = train_image_names
    df['masks']  = train_masks
    df.to_csv('train.csv')

    df = pd.DataFrame()
    df['images'] = val_image_names
    df['masks']  = val_masks
    df.to_csv('validation.csv')


    print(str(len(train_image_names))+ " images available for training")
    print(str(len(val_image_names))+ " images available for validation")



#
# transform = A.Compose([
#
#     A.RandomBrightnessContrast(p=0.2),
#     A.IAAAdditiveGaussianNoise(p=0.2),
#
#     A.OneOf([A.GridDistortion(border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0, p=1.0),
#             A.ElasticTransform(alpha_affine=10, border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0, p=1.0),
#             A.ShiftScaleRotate(
#                 shift_limit=0,
#                 scale_limit=0,
#                 rotate_limit=10,
#                 border_mode=cv2.BORDER_CONSTANT,
#                 value=0,
#                 mask_value=0,
#                 p=1.0),
#             A.OpticalDistortion(border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0, p=1.0)], p = 0.1),
#
#     A.OneOf(
#         [A.CLAHE(p=1.0),
#         A.RandomBrightness(p=1.0),
#          A.RandomGamma(p=1.0),
#          A.ISONoise(p=1.0)], p = 0.5),
#
#     A.OneOf(
#         [A.IAASharpen(p=1.0),
#          A.Blur(blur_limit=3, p=1.0),
#          A.MotionBlur(blur_limit=3, p=1.0),
#          ], p=0.5),
#
#     A.OneOf(
#         [A.RandomContrast(p=1.0),
#          A.HueSaturationValue(p=1.0),
#          ], p=0.5),
#
#
#
#     A.Cutout(p=0.3)
#
#
#
#
#
# ])


class Comma10K(Dataset):

    def __init__(self, csv_file, h = 256, w = 256, flip_rate = 0.5, crop = None):

        self.data = pd.read_csv(csv_file)
        self.flip_rate = flip_rate
        self.h = h
        self.w = w
        self.crop = crop
        self.gray_class_values = [41,76,90,124,161]
        self.classes = [0,1,2,3,4]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        image_name = self.data.iloc[idx,1]
        image = cv2.imread(image_name) # read in BGR
        image = cv2.resize(image, (self.h, self.w))
        label_name = self.data.iloc[idx,2]
        label = cv2.imread(label_name,0) # read in gray scale
        label = cv2.resize(label, (self.h, self.w), interpolation = cv2.INTER_NEAREST)

        #if random.random() < self.flip_rate:
        #    image = np.fliplr(image)
        #    label = np.fliplr(label)

        #if self.crop == True :

        #    start_x = np.random.randint(low = 0, high = (self.h-256))
        #    end_x = start_x + 256
        #    start_y = np.random.randint(low = 0, high = (self.w - 256))
        #    end_y = start_y + 256

        #    image = image[start_y:end_y, start_x:end_x]
        #    label = label[start_y:end_y, start_x:end_x]

        # image = transform(image = image)["image"]






        # one hot encdoing masks (not needed for CCEL)
        #label = np.stack([(label == v) for v in self.gray_class_values], axis=-1).astype('uint8')
        for i in range(len(self.classes)):
            label[label == self.gray_class_values[i]] = self.classes[i]


        # normalizing image with pretrained mean and std for pretraiend resnet

        image = image/255.0
        #image = image - np.array([0.485,0.456,0.406])
        #image = image / np.array([0.229, 0.224, 0.225])
        image = np.transpose(image, (2,0,1))
        image = image.astype('float32')


        # convert to tensor
        image = torch.from_numpy(image.copy()).float()
        label = torch.from_numpy(label.copy()).long()



        return image, label



def show_batch(batch):

    img_batch = batch[0]
    grid = utils.make_grid(img_batch)
    plt.imshow(grid.numpy()[::-1].transpose(1,2,0))
    plt.title('Batch from DataLoader')

if __name__ == "__main__" :

    train_data = Comma10K(csv_file='train.csv')

    # showing a batch
    batch_size = 4


    dataloader = DataLoader(train_data, batch_size = batch_size, shuffle = True)

    for i, (image,label) in enumerate(dataloader):
        print(i, image.size(), label.size())
        #print(batch[0])
        #print(batch[1])

        # observe 4th batch
        if i == 3:
            plt.figure()
            show_batch((image,label))
            plt.axis('off')
            plt.ioff()
            plt.show()
            break