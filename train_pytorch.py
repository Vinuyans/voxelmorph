#!/usr/bin/env python

"""
Example script to train a VoxelMorph model.

You will likely have to customize this script slightly to accommodate your own data. All images
should be appropriately cropped and scaled to values between 0 and 1.

If an atlas file is provided with the --atlas flag, then scan-to-atlas training is performed.
Otherwise, registration will be scan-to-scan.

If you use this code, please cite the following, and read function docs for further info/citations.

    VoxelMorph: A Learning Framework for Deformable Medical Image Registration G. Balakrishnan, A.
    Zhao, M. R. Sabuncu, J. Guttag, A.V. Dalca. IEEE TMI: Transactions on Medical Imaging. 38(8). pp
    1788-1800. 2019. 

    or

    Unsupervised Learning for Probabilistic Diffeomorphic Registration for Images and Surfaces
    A.V. Dalca, G. Balakrishnan, J. Guttag, M.R. Sabuncu. 
    MedIA: Medical Image Analysis. (57). pp 226-236, 2019 

Copyright 2020 Adrian V. Dalca

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in
compliance with the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is
distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
implied. See the License for the specific language governing permissions and limitations under the
License.
"""
import os
import importlib
import argparse
import time
import numpy as np
import torch

os.environ["NEURITE_BACKEND"] = "pytorch"
os.environ["VXM_BACKEND"] = "pytorch"
from neurite_pytorch.models import labels_to_image
import voxelmorph as vxm  
import pdb
# parse the commandline
parser = argparse.ArgumentParser()

# data organization parameters
parser.add_argument(
    "--label-dir", nargs="+", help="path or glob pattern pointing to input label maps"
)
parser.add_argument("--model-dir", default="models", help="model output directory")
parser.add_argument("--log-dir", help="optional TensorBoard log directory")
parser.add_argument("--sub-dir", help="optional subfolder for logs and model saves")

# generation parameters
parser.add_argument(
    "--same-subj", action="store_true", help="generate image pairs from same label map"
)
parser.add_argument(
    "--blur-std", type=float, default=1, help="maximum blurring std. dev."
)
parser.add_argument("--gamma", type=float, default=0.25, help="std. dev. of gamma")
parser.add_argument("--vel-std", type=float, default=0.5, help="std. dev. of SVF")
parser.add_argument("--vel-res", type=float, nargs="+", default=[16], help="SVF scale")
parser.add_argument(
    "--bias-std", type=float, default=0.3, help="std. dev. of bias field"
)
parser.add_argument(
    "--bias-res", type=float, nargs="+", default=[40], help="bias scale"
)
parser.add_argument(
    "--out-shape", type=int, nargs="+", help="output shape to pad to" ""
)
parser.add_argument(
    "--out-labels", default="fs_labels.npy", help="labels to optimize, see README"
)

# training parameters
parser.add_argument(
    "--gpu", default="0", help="GPU ID number(s), comma-separated (default: 0)"
)
parser.add_argument("--batch-size", type=int, default=1, help="batch size (default: 1)")
parser.add_argument(
    "--epochs", type=int, default=1500, help="number of training epochs (default: 1500)"
)
parser.add_argument(
    "--steps-per-epoch",
    type=int,
    default=100,
    help="frequency of model saves (default: 100)",
)
parser.add_argument("--load-model", help="optional model file to initialize with")
parser.add_argument(
    "--initial-epoch", type=int, default=0, help="initial epoch number (default: 0)"
)
parser.add_argument(
    "--lr", type=float, default=1e-4, help="learning rate (default: 1e-4)"
)
parser.add_argument(
    "--cudnn-nondet",
    action="store_true",
    help="disable cudnn determinism - might slow down training",
)

# network architecture parameters
parser.add_argument(
    "--enc",
    type=int,
    nargs="+",
    help="list of unet encoder filters (default: 16 32 32 32)",
)
parser.add_argument(
    "--dec",
    type=int,
    nargs="+",
    help="list of unet decorder filters (default: 32 32 32 32 32 16 16)",
)
parser.add_argument(
    "--int-steps", type=int, default=7, help="number of integration steps (default: 7)"
)
parser.add_argument(
    "--int-downsize",
    type=int,
    default=2,
    help="flow downsample factor for integration (default: 2)",
)
parser.add_argument(
    "--bidir", action="store_true", help="enable bidirectional cost function"
)

# loss hyperparameters
parser.add_argument(
    "--image-loss",
    default="mse",
    help="image reconstruction loss - can be mse or ncc (default: mse)",
)
parser.add_argument(
    "--lambda",
    type=float,
    dest="weight",
    default=0.01,
    help="weight of deformation loss (default: 0.01)",
)
args = parser.parse_args()

# -----------------------------------------------------------------------------------------------------------
#                                            Shape generation
# Config options taken from https://github.com/ivadomed/multimodal-registration/blob/main/train_synthmorph.py
# -----------------------------------------------------------------------------------------------------------
# in_shape = [160, 160, 192]
# num_dim = len(in_shape)
# num_label = 26
# num_maps = 100
# label_maps = []
# for _ in range(num_maps):
#     # Draw image and warp.
#     im = ne.utils.augment.draw_perlin(
#         out_shape=(*in_shape, num_label),
#         scales=[16, 32, 64],
#         max_std=1,
#     )
#     warp = ne.utils.augment.draw_perlin(
#         out_shape=(*in_shape, num_label, num_dim),
#         scales=[8, 16, 32],
#         max_std=3,
#     )

#     # Transform and create label map.
#     im = vxm.utils.transform(im, warp)
#     lab = torch.argmax(im, axis=-1)
#     label_maps.append(np.uint8(lab))
# labels_in = np.unique(label_maps)
labels_in, label_maps = vxm.py.utils.load_labels(args.label_dir)
gen = vxm.generators.synthmorph(
    label_maps,
    batch_size=args.batch_size,
    same_subj=args.same_subj,
    flip=True,
)
in_shape = label_maps[0].shape
labels_out = labels_in

# model configuration
gen_args = dict(
    in_shape=in_shape,
    out_shape=args.out_shape,
    batch_size=args.batch_size,
    in_label_list=labels_in,
    out_label_list=labels_out,
    warp_std=args.vel_std,
    warp_res=args.vel_res,
    blur_std=args.blur_std,
    bias_std=args.bias_std,
    bias_res=args.bias_res,
    gamma_std=args.gamma,
)

# prepare model folder
model_dir = args.model_dir
os.makedirs(model_dir, exist_ok=True)

gen_model_1 = labels_to_image(**gen_args, id=0)
gen_model_2 = labels_to_image(**gen_args, id=1)
# local_rank = os.environ["LOCAL_RANK"]
# device = torch.device("cpu", int(local_rank))
device = torch.device("cpu")

# unet architecture
enc_nf = args.enc if args.enc else [16, 32, 32, 32]
dec_nf = args.dec if args.dec else [32, 32, 32, 32, 32, 16, 16]

model = vxm.networks.VxmDense(
    inshape=in_shape,
    input_model=[gen_model_1, gen_model_2],
    nb_unet_features=[enc_nf, dec_nf],
    int_steps=args.int_steps,
    int_downsize=args.int_downsize,
)
# torchinfo.summary(model, input_data=next(gen))

print("Created model")
# TODO PARALLELSE
# if nb_gpus > 1:
#     # use multiple GPUs via DataParallel
#     model = torch.nn.DataParallel(model)
#     model.save = model.module.save

# prepare the model for training and send to device
model.to(device)
model.train()

# set optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)


# prepare deformation loss
# losses = [vxm.losses.Grad("l2", loss_mult=args.int_downsize).loss]
# weights = [1, args.weight]

# print(next(gen))
# pdb.set_trace()

# training loops
for epoch in range(args.initial_epoch, args.epochs):
    # save model checkpoint
    if epoch % 20 == 0:
        # model.save(os.path.join(model_dir, "%04d.pt" % epoch))
        torch.save(model.state_dict(), os.path.join(model_dir, "%04d.pt" % epoch))

    epoch_loss = []
    epoch_total_loss = []
    epoch_step_time = []

    for step in range(args.steps_per_epoch):
        step_start_time = time.time()

        # generate inputs (and true outputs) and convert them to tensors
        inputs, y_true = next(gen)
        # print(inputs, step)
        # .copy() function called to remove negative stride in numpy arrays which aren't supported in torch
        inputs = [
            torch.from_numpy(d.copy()).to(device).float().permute(0, 4, 1, 2, 3)
            for d in inputs
        ]
        y_true = [
            torch.from_numpy(d.copy()).to(device).float().permute(0, 4, 1, 2, 3)
            for d in y_true
        ]

        # run inputs through the model to produce a warped image and flow field
        y_pred, flow = model(*inputs)
        
        # calculate total loss
        loss = 0
        # maybe add // nb_devices
        const = torch.ones(args.batch_size)
        # Check if map_2 or y_true to use
        loss_dice = vxm.losses.Dice().loss(y_true, y_pred) + const 
        loss_grad =vxm.losses.Grad("l2").loss(None, flow) * args.weight
        loss += loss_dice
        loss += loss_grad 
        loss_list = [loss_dice, loss_grad]
        # for n, loss_function in enumerate(losses):
        #     curr_loss = loss_function(y_true[n], y_pred[n]) * weights[n]
        #     loss_list.append(curr_loss.item())
        #     loss += curr_loss

        epoch_loss.append(loss_list)
        epoch_total_loss.append(loss.item())

        # backpropagate and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # get compute time
        epoch_step_time.append(time.time() - step_start_time)

    # print epoch info
    epoch_info = "Epoch %d/%d" % (epoch + 1, args.epochs)
    time_info = "%.4f sec/step" % np.mean(epoch_step_time)
    losses_info = ", ".join(["%.4e" % f for f in np.mean(epoch_loss, axis=0)])
    loss_info = "loss: %.4e  (%s)" % (np.mean(epoch_total_loss), losses_info)
    print(" - ".join((epoch_info, time_info, loss_info)), flush=True)

# final model save
model.save(os.path.join(model_dir, "%04d.pt" % args.epochs))
