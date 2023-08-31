# third party
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os

from neurite_pytorch.augment import draw_perlin
import neurite_pytorch.layers as layers
import neurite_pytorch.utils as utils
import pdb    

os.environ["VXM_BACKEND"] = "pytorch"
import voxelmorph as vxm


class BaseSynth(nn.Module):
    # TODO check batch size (original was None) 
    def __init__(self,
        in_shape: tuple,
        in_label_list,
        channels=2,
        batch_size=1,
        out_label_list=None,
        out_shape=None,
        num_chan=1,
        zero_background=0.2,
        warp_res=[16],
        warp_std=0.5,
        warp_modulate=True,
        bias_res=40,
        bias_std=0.3,
        bias_modulate=True,
        blur_std=1,
        blur_modulate=True,
        normalize=True,
        gamma_std=0.25,
        dc_offset=0,
        one_hot=True,
        # seeds={},
        return_vel=False,
        return_def=False,
        id=0,):
    
        super().__init__()
        # pdb.set_trace()
        if out_shape is None:
            out_shape = in_shape
        in_shape, out_shape = map(np.asarray, (in_shape, out_shape))
        num_dim = len(in_shape)
        # Transform labels into [0, 1, ..., N-1].
        in_label_list = np.int32(np.unique(in_label_list))
        num_in_labels = len(in_label_list)
        new_in_label_list = np.arange(num_in_labels)
        in_lut = np.zeros(np.max(in_label_list) + 1, dtype=np.float32)
        # pdb.set_trace()
        for i, lab in enumerate(in_label_list):
            in_lut[lab] = i
        
        self.num_in_labels = num_in_labels    
        self.in_lut = in_lut
        
        if warp_std > 0:
            # Velocity field.
            vel_shape = (*out_shape // 2, num_dim)
            vel_scale = np.asarray(warp_res) / 2
            vel_draw = lambda x: draw_perlin(
                vel_shape, scales=vel_scale,
                min_std=0 if warp_modulate else warp_std, max_std=warp_std,
            )
            
            self.vel_draw = vel_draw
        
        self.in_label_list= in_label_list
        self.batch_size= batch_size
        self.out_label_list= out_label_list
        self.num_dim = num_dim
        self.out_shape= out_shape
        self.num_chan= num_chan
        self.zero_background= zero_background
        self.warp_res= warp_res
        self.warp_std= warp_std
        self.warp_modulate= warp_modulate
        self.bias_res= bias_res
        self.bias_std= bias_std
        self.bias_modulate= bias_modulate
        self.blur_std= blur_std
        self.blur_modulate= blur_modulate
        self.normalize= normalize
        self.gamma_std= gamma_std
        self.dc_offset= dc_offset
        self.one_hot= one_hot
        self.return_vel= return_vel
        self.return_def= return_def
        self.id= id
        
        
        # Layers
        self.InputLayer = layers.InputLayer(shape=(self.batch_size, channels, *in_shape), batch_size=self.batch_size, channels=channels)
        self.VelDrawLambdaLayer = layers.LambdaLayer(self.vel_draw)
        # self.VecIntLayer = vxm.layers.VecInt(nsteps=5)
        self.RescaleValuesLayer = layers.RescaleValues(2)
        # self.ResizeLayer = layers.Resize(2, interp_method='linear')
        # self.SpatialTransformerLayer = vxm.layers.SpatialTransformer(mode='nearest')
        
    def forward(self, x: torch.Tensor):
        import pdb
        labels = self.InputLayer(x)
        if torch.is_floating_point(labels):
            labels = labels.to(torch.int32)
        
        #TODO Check
        labels = torch.tensor(self.in_lut[labels], dtype=torch.float32)
        # labels=labels[None,:,:,:]
        # labels = torch.unsqueeze(-1)
        
        # One per batch.
        def_field= self.VelDrawLambdaLayer(labels)
        
        # Deformation field.
        def_field = vxm.layers.VecInt(def_field.shape[1:], nsteps=5)(torch.unsqueeze(def_field, dim=0))
        def_field = self.RescaleValuesLayer(def_field)
        def_field = layers.Resize(2, def_field.shape, interp_method='linear')(def_field)
        pdb.set_trace()
        # # Resampling
        labels = vxm.layers.SpatialTransformer(def_field.shape[2:], mode='nearest')(labels, def_field)

        labels = labels.type(torch.int32)
        
        # Intensity means and standard deviations.
        mean_min = 25
        mean_max = 225
        std_min = 5
        std_max = 25
        # m0, m1, s0, s1 = map(np.asarray, (mean_min, mean_max, std_min, std_max))
        # pdb.set_trace()
        mean = torch.FloatTensor(size=(self.batch_size, self.num_chan, self.num_in_labels)).uniform_(mean_min,mean_max)
        std = torch.FloatTensor(size=(self.batch_size, self.num_chan, self.num_in_labels)).uniform_(std_min,std_max) 

        # Synthetic image.
        image = torch.normal(mean=0.0, std=1.0, size=labels.size())
        indices = torch.cat([labels + i * self.num_in_labels for i in range(self.num_chan)], dim=-1).to(torch.int32)
        gather = lambda x: torch.reshape(x[0], (-1,))[x[1]]
        mean = gather([mean, indices])
        std = gather([std, indices])
        image = image * std + mean
        
        # Zero background.
        if self.zero_background > 0:
            rand_flip = torch.rand(
                size=(self.batch_size, *[1] * self.num_dim, self.num_chan),
            )
            rand_flip = torch.less(rand_flip, self.zero_background)
            image *= 1. - torch.logical_and(labels == 0, rand_flip).to(image.dtype)

        # Blur.
        if self.blur_std > 0:
            kernels = utils.gaussian_kernel(
                [self.blur_std] * self.num_dim, separate=True, random=self.blur_modulate,
                dtype=image.dtype,
            )
            image = utils.separable_conv(image, kernels, batched=True)

        # Bias field.
        if self.bias_std > 0:
            bias_shape = (*self.out_shape, 1)
            bias_draw = lambda x: draw_perlin(
                bias_shape, scales=self.bias_res,
                min_std=0 if self.bias_modulate else self.bias_std, max_std=self.bias_std,
            )
            # bias_field = KL.Lambda(lambda x: tf.map_fn(
                # bias_draw, x, fn_output_signature='float32'))(labels)
            bias_field = bias_draw(labels)
            image *= torch.exp(bias_field)

        # Intensity manipulations.
        image = torch.clamp(image, min=0, max=255)
        if self.normalize:
            # image = KL.Lambda(lambda x: tf.map_fn(utils.minmax_norm, x))(image)
            image = utils.minmax_norm(image) # TODO checck minmax
        if self.gamma_std > 0:
            gamma = torch.normal(
                std=self.gamma_std,
                mean=0.0,
                size=(self.batch_size, *[1] * self.num_dim, self.num_chan)
            )
            image = torch.pow(image, torch.exp(gamma))
        
        # Not Called
        # if dc_offset > 0:
        #     image += tf.random.uniform(
        #         shape=(batch_size, *[1] * num_dim, num_chan),
        #         maxval=dc_offset,
        #         seed=seeds.get('dc_offset'),
        #     )

        # Lookup table for converting the index labels back to the original values,
        # setting unwanted labels to background. If the output labels are provided
        # as a dictionary, it can be used e.g. to convert labels to GM, WM, CSF.
        if self.out_label_list is None:
            self.out_label_list = self.in_label_list
        if isinstance(self.out_label_list, (tuple, list, np.ndarray)):
            self.out_label_list = {lab: lab for lab in self.out_label_list}
        out_lut = np.zeros(self.num_in_labels, dtype='int32')
        for i, lab in enumerate(self.in_label_list):
            if lab in self.out_label_list:
                out_lut[i] = self.out_label_list[lab]

        # For one-hot encoding, update the lookup table such that the M desired
        # output labels are rebased into the interval [0, M-1[. If the background
        # with value 0 is not part of the output labels, set it to -1 to remove it
        # from the one-hot maps.
        if self.one_hot:
            hot_label_list = np.unique(list(self.out_label_list.values()))  # Sorted.
            hot_lut = np.full(hot_label_list[-1] + 1, fill_value=-1, dtype='int32')
            for i, lab in enumerate(hot_label_list):
                hot_lut[lab] = i
            out_lut = hot_lut[out_lut]

        # Convert indices to output labels only once.
        # labels = tf.gather(out_lut, labels, name=f'labels_back_{id}')
        # pdb.set_trace()
        labels = torch.tensor(out_lut[labels]).to(torch.int32)
        if self.one_hot:
            labels = F.one_hot(labels[..., 0].long(), num_classes=len(hot_label_list))

        outputs = [image, labels]
        return outputs
    
        # Not Called
        # if return_vel:
        #     outputs.append(vel_field)
        # if return_def:
        #     outputs.append(def_field)
        
class InputModel(nn.Module):
    def __init__(self, gen_model_1, gen_model_2):
        super().__init__()
        self.gen_model_1 = gen_model_1
        self.gen_model_2 = gen_model_2
    def forward(self, x):
        ima_1, map_1 = self.gen_model_1(x)
        ima_2, map_2 = self.gen_model_2(x)     
        

def labels_to_image(
    in_shape,
    in_label_list,
    batch_size=1,
    out_label_list=None,
    out_shape=None,
    num_chan=1,
    mean_min=None,
    mean_max=None,
    std_min=None,
    std_max=None,
    zero_background=0.2,
    warp_res=[16],
    warp_std=0.5,
    warp_modulate=True,
    bias_res=40,
    bias_std=0.3,
    bias_modulate=True,
    blur_std=1,
    blur_modulate=True,
    normalize=True,
    gamma_std=0.25,
    dc_offset=0,
    one_hot=True,
    # seeds={},
    return_vel=False,
    return_def=False,
    id=0,
):
    """
    Generative model for augmenting label maps and synthesizing images from them.

    Parameters:
        in_shape: List of the spatial dimensions of the input label maps.
        in_label_list: List of all possible input labels.
        out_label_list (optional): List of labels in the output label maps. If
            a dictionary is passed, it will be used to convert labels, e.g. to
            GM, WM and CSF. All labels not included will be converted to
            background with value 0. If 0 is among the output labels, it will be
            one-hot encoded. Defaults to the input labels.
        out_shape (optional): List of the spatial dimensions of the outputs.
            Inputs will be symmetrically cropped or zero-padded to fit.
            Defaults to the input shape.
        num_chan (optional): Number of image channels to be synthesized.
            Defaults to 1.
        mean_min (optional): List of lower bounds on the means drawn to generate
            the intensities for each label. Defaults to 0 for the background and
            25 for all other labels.
        mean_max (optional): List of upper bounds on the means drawn to generate
            the intensities for each label. Defaults to 225 for each label.
        std_min (optional): List of lower bounds on the SDs drawn to generate
            the intensities for each label. Defaults to 0 for the background and
            5 for all other labels.
        std_max (optional): List of upper bounds on the SDs drawn to generate
            the intensities for each label. Defaults to 25 for each label.
            25 for all other labels.
        zero_background (float, optional): Probability that the background is set
            to zero. Defaults to 0.2.
        warp_res (optional): List of factors N determining the
            resultion 1/N relative to the inputs at which the SVF is drawn.
            Defaults to 16.
        warp_std (float, optional): Upper bound on the SDs used when drawing
            the SVF. Defaults to 0.5.
        warp_modulate (bool, optional): Whether to draw the SVF with random SDs.
            If disabled, each batch will use the maximum SD. Defaults to True.
        bias_res (optional): List of factors N determining the
            resultion 1/N relative to the inputs at which the bias field is
            drawn. Defaults to 40.
        bias_std (float, optional): Upper bound on the SDs used when drawing
            the bias field. Defaults to 0.3.
        bias_modulate (bool, optional): Whether to draw the bias field with
            random SDs. If disabled, each batch will use the maximum SD.
            Defaults to True.
        blur_std (float, optional): Upper bound on the SD of the kernel used
            for Gaussian image blurring. Defaults to 1.
        blur_modulate (bool, optional): Whether to draw random blurring SDs.
            If disabled, each batch will use the maximum SD. Defaults to True.
        normalize (bool, optional): Whether the image is min-max normalized.
            Defaults to True.
        gamma_std (float, optional): SD of random global intensity
            exponentiation, i.e. gamma augmentation. Defaults to 0.25.
        dc_offset (float, optional): Upper bound on global DC offset drawn and
            added to the image after normalization. Defaults to 0.
        one_hot (bool, optional): Whether output label maps are one-hot encoded.
            Only the specified output labels will be included. Defaults to True.
        seeds (dictionary, optional): Integers for reproducible randomization.
        return_vel (bool, optional): Whether to append the half-resolution SVF
            to the model outputs. Defaults to False.
        return_def (bool, optional): Whether to append the combined displacement
            field to the model outputs. Defaults to False.
        id (int, optional): Model identifier used to create unique layer names
            for including this model multiple times. Defaults to 0.
    """
    
    

    # return tf.keras.Model(labels_input, outputs, name=f'synth_{id}')
    return BaseSynth(
        in_shape=in_shape, 
        in_label_list=in_label_list,
        batch_size=batch_size, 
        out_label_list=out_label_list, 
        out_shape=out_shape, 
        num_chan=num_chan, 
        zero_background=zero_background, 
        warp_res=warp_res, 
        warp_std=warp_std, 
        warp_modulate=warp_modulate, 
        bias_res=bias_res, 
        bias_std=bias_std, 
        bias_modulate=bias_modulate, 
        blur_std=blur_std, 
        blur_modulate=blur_modulate, 
        normalize=normalize, 
        gamma_std=gamma_std, 
        dc_offset=dc_offset, 
        one_hot=one_hot, 
        # seeds, 
        return_vel=return_vel, 
        return_def=return_def, 
        id=id)