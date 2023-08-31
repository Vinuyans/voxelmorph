import torch
from torch import nn
from neurite_pytorch.augment import resize
import pdb

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super().__init__()
        self.lambd = lambd
    def forward(self, x):
        return self.lambd(x)
    
class InputLayer(nn.Module):

    def __init__(self, shape : tuple, batch_size, channels):
        super().__init__()
        self.shape = shape
        self.batch_size = batch_size
        self.channels = channels

    def forward(self, x):
        # tensor_shape = tuple(x.shape[1:])
        tensor_shape = (self.batch_size,  self.channels, *list(x.shape[2:]))
        
        if tensor_shape != self.shape :
            raise ValueError(f"Input Tensor's shape must be {self.shape} but got {tensor_shape} instead")
        
        return torch.reshape(x, tensor_shape)
    
class RescaleValues(nn.Module):
    """ 
    Simple Custom Layer to rescale data values (e.g. intensities) by fixed factor
    """

    def __init__(self, resize):
        super().__init__()
        self.resize = resize

    def forward(self, x):
        return x * self.resize

class Resize(nn.Module):
    """
    N-D Resize Tensorflow / Keras Layer
    Note: this is not re-shaping an existing volume, but resizing, like scipy's "Zoom"

    If you find this class useful, please cite the original paper this was written for:
        Dalca AV, Guttag J, Sabuncu MR
        Anatomical Priors in Convolutional Networks for Unsupervised Biomedical Segmentation, 
        CVPR 2018. https://arxiv.org/abs/1903.03148
    """

    def __init__(self,
                 zoom_factor,
                 input_shape,
                 interp_method='linear',
                 **kwargs):
        """
        Parameters: 
            interp_method: 'linear' or 'nearest'
                'xy' indexing will have the first two entries of the flow 
                (along last axis) flipped compared to 'ij' indexing
        """
        super().__init__()
        
        if isinstance(input_shape[0], (list, tuple)) and len(input_shape) > 1:
            raise Exception('Resize must be called on a list of length 1.')

        if isinstance(input_shape[0], (list, tuple)):
            input_shape = input_shape[0]
            
        self.zoom_factor = zoom_factor
        self.interp_method = interp_method
                
        # set up number of dimensions
        self.ndims = len(input_shape) - 2
        self.inshape = input_shape
        if not isinstance(self.zoom_factor, (list, tuple)):
            self.zoom_factor = [self.zoom_factor] * self.ndims
        else:
            assert len(self.zoom_factor) == self.ndims, \
                'zoom factor length {} does not match number of dimensions {}'\
                .format(len(self.zoom_factor), self.ndims)
     
    def forward(self, inputs):
        """
        Parameters
            inputs: volume of list with one volume
        """

        # check shapes
        if isinstance(inputs, (list, tuple)):
            assert len(inputs) == 1, "inputs has to be len 1. found: %d" % len(inputs)
            vol = inputs[0]
        else:
            vol = inputs
        # necessary for multi_gpu models...
        vol = torch.reshape(vol, [-1, *self.inshape[1:]])

        # map transform across batch
        #TODO Check if this works
        return torch.stack([self._single_resize(x) for x in vol])
    
    def _single_resize(self, inputs):
            return resize(inputs, self.zoom_factor, interp_method=self.interp_method)
