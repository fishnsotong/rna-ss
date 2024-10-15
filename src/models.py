import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    """Defines a single residual block for the network."""
    def __init__(self, in_channels, out_channels, stride=1):
        pass

    def forward(self, x):
        pass

class ResidualNetwork(nn.Module):
    """Defines a residual network (ResNet) architecture"
    
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
