import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    """Defines a single residual block for the ResNet architecture.

    A residual block consists of two convolutional layers, each followed by batch
    normalisation and a ReLU activation. The block also contains a residual (skip)
    connection that bypasses the two convolutional layers, allowing the input to be
    added directly to the output of the second convolutional layer. This helps to
    mitigate the vanishing gradient problem in deep networks during training.

    If the input and output dimensions of the residual block are different, the
    skip connection is modified to match the output dimensions by using a 1x1
    convolution and batch normalisation.

    Attributes:

    Methods

    Args:
        in_channels (int): The number of input channels to the block.
        out_channels (int): The number of output channels from the block.
        stride (int): The stride length for the convolutional layers. Default: 1.
    """
    def __init__(self, in_channels, out_channels, stride=1):
        pass

    def forward(self, x):
        pass

class ResidualNetwork(nn.Module):
    """Defines a residual network (ResNet) architecture"

    Attributes:

    Methods:

    Args:
    
    Reference:
        He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep Residual Learning for Image Recognition. 
        In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 
        770-778. https://doi.org/10.1109/CVPR.2016.90
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # NOTE: when methods are declared with an underscore in front of them, they are treated
    #       as "private" -- not to be used outside the class.

    def _make_layer():
        pass

    def forward():
        pass
