# evaluating the validation data

from data import Comma10K
from deeplabv3 import DeepLabV3
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

model_id = "1"
batch_size = 1 # 1 so that each image can be evaluated individually
model = DeepLabV3(model_id, project_dir = None)
eval_data = Comma10K(csv_file = 'validation.csv')
num_eval_batches = int(len(eval_data)/ batch_size)
eval_loader = DataLoader(dataset = eval_data, batch_size = batch_size, shuffle = False)
print("data ready for evaluation")

if torch.cuda.is_available():
    device = torch.device('cuda')
    print("gpu detected")
else :
    device = torch.device('cpu')
    print("gpu undetected")

model.load_state_dict(torch.load('model/model_1_epoch_206.pth'))
model = model.to(device)

eval_losses = []
loss_fn = nn.CrossEntropyLoss()
model.eval()

def evaluate():

    print("evaluating, have patience :)")
    for i,(images,labels) in enumerate(eval_loader):

        with torch.no_grad():
            images = images.to(device)
            labels = labels.type(torch.LongTensor).to(device)
            outputs = model(images)
            loss = loss_fn(outputs, labels)

            labels = labels.detach().cpu().numpy()
            images = images.detach().cpu().numpy()
            outputs = outputs.detach().cpu().numpy()
            loss = loss.detach().cpu().numpy()
            eval_losses.append(loss)

    print("mean evaluation loss : " + str(np.mean(eval_losses)))

    least_loss_index = np.argmin(eval_losses)
    highest_loss_index = np.argmax(eval_losses)

    print(least_loss_index) # 971
    print(highest_loss_index) # 272

#evaluate()


def segmap_to_colormap(pred):

    #print(pred)
    classes = [0, 1, 2, 3, 4]
    label_to_color = {
        0 : [0.2509804, 0.1254902, 0.1254902], #
        1 : [1,0,0], #
        2 : [0.8,0,1.0], # self car
        3 : [0.5019608, 0.5019608, 0.3764706], # undrivable (can be objects too like house)
        4 : [0,1,0.5] #

    }
    h,w = pred.shape
    img_color = np.zeros((h,w,3))
    for row in range(h):
        for col in range(w):
            label = pred[row, col]

            img_color[row, col] = label_to_color[label]

    #print(img_color)
    plt.imshow(img_color)
    plt.show()


# def show_input_image(image):
#
#     # image : tensor 3*h*w
#


least_error_image = eval_data[272][0]
least_error_image = least_error_image.to(device)
least_error_image = torch.reshape(least_error_image,(1,3,256,256)) # insert dimensions of image used for training here

least_error_label = eval_data[272][1]
output_least_error_image = model(least_error_image)
output_least_error_image = output_least_error_image.detach().cpu().numpy()
output_least_error_image = np.argmax(output_least_error_image, axis = 1)
output_least_error_image = np.reshape(output_least_error_image,(256,256))


# showing input image
least_error_image = least_error_image.detach().cpu().numpy()
least_error_image = np.reshape(least_error_image,(3,256,256))
least_error_image = np.transpose(least_error_image,(1,2,0))
plt.imshow(least_error_image)
plt.show()

# converting segmented map to rgb pixels
segmap_to_colormap(output_least_error_image)












