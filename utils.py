import numpy as np
import cv2
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd


# https://raberrytv.wordpress.com/2017/10/29/pytorch-weight-decay-made-easy/
def add_weight_decay(net, l2_value, ski_list = ()):

    decay = []
    no_decay = []
    for name, param in net.named_parameters():
        if not param.requires_grad :
            continue
        if len(param.shape) == 1 or name.endswith('bias'):
            no_decay.append(param)
        else :
            decay.append(param)

    return [{'params': no_decay, 'weight_decay':0.0,}, {'params': decay, 'weight_decay' : l2_value}]


# computing class weights for the dataset
def get_class_weights():

    num_classes = 5
    gray_class_values = [41,76,90,124,161]
    classes = [0,1,2,3,4]

    data = pd.read_csv('train.csv')
    img_paths = data['masks'].tolist()
    train_id_to_count = {}

    for train_id in range(num_classes):
        train_id_to_count[train_id] = 0

    for i in range(len(img_paths)):

        mask = cv2.imread(img_paths[i],0)
        mask = cv2.resize(mask, (256,256), interpolation = cv2.INTER_NEAREST)



        for i in range(len(classes)):
            mask[mask == gray_class_values[i]] = classes[i]


            for train_id in range(num_classes):


                train_id_mask = np.equal(mask, train_id)
                train_id_count = np.sum(train_id_mask)

                train_id_to_count[train_id] += train_id_count

        weights = []
        total_count = sum(train_id_to_count.values())

        for train_id, count in train_id_to_count.items():

            train_id_prob = float(count) / float(total_count)
            train_id_weight = 1 / np.log(1.02 + train_id_prob)
            weights.append(train_id_weight)


    return weights

# weights = get_class_weights()
# np.array(weights).dump(open('weights.npy', 'wb'))





