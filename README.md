# Comma10k-Segmentation-using-DeepLabV3

- [Comma10k](https://github.com/commaai/comma10k) has been used to train [DeepLabV3](https://arxiv.org/abs/1706.05587) architecture. Motivation behing using DeepLabV3 is [atrous convolutions] with varying rates. egs. For a 3x3 filter with rate r, there will be r-1 zeros placed between every element in the filter. rate = 1 corresponds to a regular convolution filter. This allows to capture contexts aross the image at a larger scale. The other 2 significant concepts in the paper are the encoder architecture and the ASPP pooling which can be understood from the paper.

# Encoder (Resnet)

- Resnet18 architecture was used for extracting the features from input images. From Resnet18, all the blocks (last 3 blocks) were removed except the first convolutional layer followed by batch normalization and a max pooling layer. On top of that, 2 basic blocks of resnet were added where each block contained  2 atrous convolutional filters with batch normalization after each convolution layer. Code can be seen in [resnet.py](https://github.com/Msarang7/Comma10k-Segmentation-using-DeepLabV3/blob/main/resnet.py). Resnet34 has also been implemented which can be used in place of Resnet18. The only difference is that Resnet34 will have more layeres left after removing the last 3 layers from the original layer.

- Important point to notice here is that both the archtiectures mentioned above have outputstride of 8. This makes the height and width of the feature channels h/8 and w/8 after passing them through the network where h and w are the original height and width of the images used for tranining.

# ASPP (pyramid pooling)

- After passing the image through the encoder, various rates are used for atrous convolution and the output from each convoluion layer is concatenated followed by concatenatio with a max pooling layer. This large band of channels of features are passed through another convolution to reduce the number of channels to number of classes in the masks. During this process, the height and width of the feature channels obtained from encoder is maintained to h/8 and w/8. Code for this is present in [aspp.py](https://github.com/Msarang7/Comma10k-Segmentation-using-DeepLabV3/blob/main/aspp.py).

# Output at the end of the Architecture (upsampling)

- The output from ASPP is upsampled to h and w from h/8 adnd w/8 keeping the number of channels equivalent to the number of labels in the masks. Code for this is present in [deeplabv3.py](https://github.com/Msarang7/Comma10k-Segmentation-using-DeepLabV3/blob/main/deeplabv3.py). I personally think upsampling the feature map by times 8 is too rapid and the upsampling should be done at slower pace as the model learns about the where abouts of the objects in image during upsampling.


# Training Details and Results

- Model was trained for 70 hours on a RTX-2060. Values of training parameters can be found in [train.py](https://github.com/Msarang7/Comma10k-Segmentation-using-DeepLabV3/blob/main/train.py).

- Using Categorical Cross Entropy as the metric, validation loss of 0.0505 and training loss  0.045120 was observed. The model can converge more probably but due to limited computing resources it was trained for limited time.


# Visualization and weight for labels

![temp](https://github.com/Msarang7/Comma10k-Segmentation-using-DeepLabV3/blob/main/segmented%20results/1.jpg)
![temp](https://github.com/Msarang7/Comma10k-Segmentation-using-DeepLabV3/blob/main/segmented%20results/2.jpg)

- Most of the segmentation looks good for the two randomly chosen images. For the first image, the car farthest coming from the opposite direction can be segmented better. Lane segmentation can also be improved. For the second image, the trees between the two trucks are identified as part of the truck.

- The reason for this can the less amount of fragments of the image are dedicated to lane lines and moving objects as compared to the drivers car, undrivable fragments and road. To tackle this problem weighted cross entropy is used. Weights are assigned for each class in the network based on the [ENet paper](https://arxiv.org/abs/1606.02147).

# Best and Worst

- The validation loss for each image was sorted and the best and worst predicted segmaps are shown below. 

![temp](https://github.com/Msarang7/Comma10k-Segmentation-using-DeepLabV3/blob/main/segmented%20results/best.jpg)
![temp](https://github.com/Msarang7/Comma10k-Segmentation-using-DeepLabV3/blob/main/segmented%20results/worst.jpg)

The first image is the easiest to classify therefore it gave the least error while the second image gave the most error. In second image, the front of the car is classified as part of the road and lane lines further away are completely missing. Model can be trained by augmenting the data using the Albumentations library for the frames during the night time.

# Torchsummary

- [Torchsummary](https://github.com/sksq96/pytorch-summary) is used for getting the summary of architectures. Code for this is present in [scratch.py](https://github.com/Msarang7/Comma10k-Segmentation-using-DeepLabV3/blob/main/scratch.py). You can check it by uncommenting the required part.

# Execution

- *python train.py* for training the model.
- *python evaluate.py* for visualizing the results.


















