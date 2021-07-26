import sys
from data import Comma10K
from deeplabv3 import DeepLabV3
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from utils import add_weight_decay
import numpy as np
import pickle
import matplotlib.pyplot as  plt
import cv2
import time
import warnings
warnings.filterwarnings('ignore')

model_id = "1"
num_epochs = 1000
batch_size = 4
learning_rate = 0.0001

model = DeepLabV3(model_id, project_dir = "/root/deeplabv3")
train_data = Comma10K(csv_file='train.csv')
val_data = Comma10K(csv_file='validation.csv')

num_train_batches = int(len(train_data)/batch_size)
num_val_batches = int(len(val_data)/batch_size)

train_loader = DataLoader(dataset = train_data, batch_size = batch_size, shuffle = True)
val_loader = DataLoader(dataset = val_data, batch_size = batch_size, shuffle = True)

print("Data loading completed for training and validation")

#model.load_state_dict(torch.load('model/model_1_epoch_204.pth'))

if torch.cuda.is_available():
    device = torch.device('cuda')
    print("gpu detected")
else :
    device = torch.device('cpu')
    print("gpu undetected")

model = model.to(device)

params = add_weight_decay(model, l2_value = 0.0001) # regularization
optimizer = torch.optim.Adam(params, lr = learning_rate)

class_weights = np.load('weights.npy', allow_pickle = True)
class_weights = torch.from_numpy(class_weights)
class_weights = class_weights.type(torch.FloatTensor)
class_weights = class_weights.to(device)

loss_fn = nn.CrossEntropyLoss()

epoch_losses_train = []
epoch_losses_val = []


for epoch in range(205,num_epochs):

    model.train()
    batch_losses = []

    for i,(images, labels) in enumerate(train_loader):

        images = images.to(device) # batch_size * 3 * h * w
        labels = labels.type(torch.LongTensor).to(device) # batch_size * h * w
        outputs = model(images) # batch_size * num_classes * h * w

        loss_batch = loss_fn(outputs, labels)
        batch_losses.append(loss_batch.detach().cpu().numpy())

        optimizer.zero_grad()
        loss_batch.backward()
        optimizer.step()

        # detaching more variables from gpu
        labels = labels.detach().cpu().numpy()
        images = images.detach().cpu().numpy()
        outputs = outputs.detach().cpu().numpy()
        loss_batch = loss_batch.detach().cpu().numpy()

        print("Epoch " + str(epoch) + "/" +str(num_epochs)  + ", train batch " + str(i) + "/" + str(len(train_loader)) + ", loss : " + str(loss_batch))

    epoch_loss = np.mean(batch_losses)
    epoch_losses_train.append(epoch_loss)

    with open("results/epoch_losses_train.pkl", "wb") as file :
        pickle.dump(epoch_losses_train, file)

    print("train loss : " + str(epoch_loss))
    plt.figure(1)
    plt.plot(epoch_losses_train,"k^")
    plt.plot(epoch_losses_train,"k")
    plt.xlabel("loss")
    plt.ylabel("epoch")
    plt.title("training loss per epoch")
    plt.savefig("epoch_losses_train.png")
    plt.close(1)



    ###############################

    print("For Validation")

    model.eval()
    batch_losses = []
    for j, (images, labels) in enumerate(val_loader):


        with torch.no_grad():
            images = images.to(device)  # batch_size * 3 * h * w
            labels = labels.type(torch.LongTensor).to(device)  # batch_size * h * w
            outputs = model(images)
            loss = loss_fn(outputs, labels)

            # detaching more variables from gpu
            labels = labels.detach().cpu().numpy()
            images = images.detach().cpu().numpy()
            outputs = outputs.detach().cpu().numpy()
            loss = loss.detach().cpu().numpy()

            batch_losses.append(loss)

        print("Epoch " + str(epoch) + "/" +str(num_epochs) + ", val batch " + str(j) + "/" + str(len(val_loader))  + "loss : " + str(loss))

    epoch_loss = np.mean(batch_losses)
    epoch_losses_val.append(epoch_loss)
    print("Epoch " + str(epoch) + " val loss : " + str(epoch_loss))

    with open("results/epoch_losses_val.pkl", "wb") as file :
        pickle.dump(epoch_losses_val, file)

    plt.figure(1)
    plt.plot(epoch_losses_val,"k^")
    plt.plot(epoch_losses_val,"k")
    plt.xlabel("loss")
    plt.ylabel("epoch")
    plt.title("validation loss per epoch")
    plt.savefig("epoch_losses_val.png")
    plt.close(1)

    # saving the model
    checkpoint_path = "model/" + "/model_" + model_id + "_epoch_" + str(epoch) + ".pth"
    torch.save(model.state_dict(), checkpoint_path)














