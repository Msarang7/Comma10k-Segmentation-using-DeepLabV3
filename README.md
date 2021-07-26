# Comma10k-Segmentation-using-DeepLabV3

- [Comma10k](https://github.com/commaai/comma10k) has been used to train [DeepLabV3](https://arxiv.org/abs/1706.05587) architecture. Motivation behing using DeepLabV3 is [atrous convolutions] with varying rates. egs. For a 3x3 filter with rate r, there will be r-1 zeros placed between every element in the filter. Rate = 1 corresponds to a regular convolution filter. This allows to capture contexts aross the image at a larger scale. The other 2 significant concepts in the paper are the encoder architecture and the ASPP pooling which can be understood from the paper.

# Encoder

- Resnet18 architecture was used for extracting the features from input images. From Resnet18, all the blocks (last 3 blocks) were removed except the first convolutional layer followed by batch normalization and a max pooling layer. On top of that, 2 basic blocks of resnet were added where each block contained  2 atrous convolutional filters with batch normalization after each convolution layer. Code can be seen in [resnet.py](https://github.com/Msarang7/Comma10k-Segmentation-using-DeepLabV3/blob/main/resnet.py). Resnet34 has also been implemented which can be used in place of Resnet18. The only difference is that Resnet34 will have more layeres left after removing the last 3 layers from the original layer.

- Important point to notice here is that both the archtiectures mentioned above have outputstride of 8. This height and width of the feature channels will be h/8 and w/8 after passng them through the network where h and w are the original height and width of the images used for tranining.

# ASPP

- After passing the image through the encoder, various rates are used for atrous convolution and the output from each convoluion layer is concatenated followed by concatenatio with a max pooling layer. This large channels of features are passed through another convolution to reduce the number of channels to number of classes in the masks. During this process, the height and width of the feature channels obtained from encoder is maintained to h/8 and w/8.






