#!/usr/bin/env python

"""
Script to train Synthmorph with pytorch.
Based on the /scripts/tf/train_synthmoprh.py and /scripts/torch/train.py scripts found on the voxelmorph repo (https://github.com/voxelmorph/voxelmorph)

Citations: 

    VoxelMorph: A Learning Framework for Deformable Medical Image Registration G. Balakrishnan, A.
    Zhao, M. R. Sabuncu, J. Guttag, A.V. Dalca. IEEE TMI: Transactions on Medical Imaging. 38(8). pp
    1788-1800. 2019. 
"""
import os
import argparse
import importlib
import time
import datetime
import logging
import csv

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.distributed.optim import ZeroRedundancyOptimizer
from torch.distributed.elastic.multiprocessing.errors import record
from torch.utils.data import Dataset, DataLoader

import numpy as np

# Set vxm library to use pytorch backend
os.environ["NEURITE_BACKEND"] = "pytorch"
os.environ["VXM_BACKEND"] = "pytorch"

import voxelmorph as vxm

# sys.path.append("/voxelmorph/synchroLib")
# from verrouPyBinding import bindingVerrouCLib

# verrouCBindingLib = "/voxelmorph/synchroLib/verrouCBindingLib.so"
# bindVerrou = bindingVerrouCLib(verrouCBindingLib)
# bindVerrou.verrou_stop_instrumentation()
# bindVerrou.verrou_display_counters()
# slurm_id = os.environ['SLURM_ARRAY_TASK_ID']
# logging.basicConfig(filename='dist_verrou_%s.log'%slurm_id, level=logging.INFO)

torch.set_num_threads(1)

# parse the commandline
parser = argparse.ArgumentParser()

# data organization parameters
parser.add_argument("--label-dir", nargs="+", help="path or glob pattern pointing to input label maps")
parser.add_argument("--train-data", help="npz file with training data")
parser.add_argument("--model-dir", default="models", help="model output directory")
parser.add_argument("--log-dir", help="optional TensorBoard log directory")
parser.add_argument("--sub-dir", help="optional subfolder for logs and model saves")

# generation parameters
parser.add_argument("--same-subj", action="store_true", help="generate image pairs from same label map")
parser.add_argument("--blur-std", type=float, default=1, help="maximum blurring std. dev.")
parser.add_argument("--gamma", type=float, default=0.25, help="std. dev. of gamma")
parser.add_argument("--vel-std", type=float, default=0.5, help="std. dev. of SVF")
parser.add_argument("--vel-res", type=float, nargs="+", default=[16], help="SVF scale")
parser.add_argument("--bias-std", type=float, default=0.3, help="std. dev. of bias field")
parser.add_argument("--bias-res", type=float, nargs="+", default=[40], help="bias scale")
parser.add_argument("--out-shape", type=int, nargs="+", help="output shape to pad to")
parser.add_argument("--out-labels", default="fs_labels.npy", help="labels to optimize, see README")

# training parameters
parser.add_argument("--batch-size", type=int, default=1, help="batch size (default: 1)")
parser.add_argument("--epochs", type=int, default=1500, help="number of training epochs (default: 1500)")
parser.add_argument("--steps", type=int, default=100, help="number of steps per epochs (default: 100)")
parser.add_argument("--load-model", help="optional model file to initialize with")
parser.add_argument("--initial-epoch", type=int, default=0, help="initial epoch number (default: 0)")
parser.add_argument("--lr", type=float, default=1e-4, help="learning rate (default: 1e-4)")
parser.add_argument("--cudnn-nondet", action="store_true", help="disable cudnn determinism - might slow down training")
parser.add_argument("--reg-param", type=float, default=1.0, help="regularization weight")

# network architecture parameters
parser.add_argument("--enc", type=int, nargs="+", help="list of unet encoder filters")
parser.add_argument("--dec", type=int, nargs="+", help="list of unet decorder filters")
parser.add_argument("--int-steps", type=int, default=5, help="number of integration steps (default:5)")
parser.add_argument("--int-downsize", type=int, default=2, help="flow downsample factor for integration (default: 2)")
parser.add_argument("--bidir", action="store_true", help="enable bidirectional cost function")

# loss hyperparameters
parser.add_argument("--image-loss", default="mse", help="image reconstruction loss - can be mse or ncc (default: mse)")
parser.add_argument(
    "--lambda", type=float, dest="weight", default=0.01, help="weight of deformation loss (default: 0.01)"
)

# Data loading
parser.add_argument("--data-dir", required=True, help="Path to folder containing cpu data directories")

args = parser.parse_args()


@record
def main():
    setup()
    world_size = os.environ["WORLD_SIZE"]
    local_rank = os.environ["LOCAL_RANK"]
    rank = os.environ["RANK"]
    # device = torch.device("cpu")
    device = torch.device("cpu", int(local_rank))
    print(f"{datetime.datetime.now()} Running synthmorph on {device}; local rank:{local_rank}; global rank {rank}")

    # Load labels & create Synthmorph generator
    labels_in, label_maps = vxm.py.utils.load_labels(args.label_dir)
    # gen = vxm.generators.synthmorph(
    #     label_maps,
    #     batch_size=args.batch_size,
    #     same_subj=args.same_subj,  # Should be true according to paper TODO Check
    #     flip=True,
    # )
    in_shape = label_maps[0].shape

    # # model configuration
    # gen_args = dict(
    #     in_shape=in_shape,
    #     out_shape=args.out_shape,
    #     in_label_list=labels_in,
    #     warp_std=args.vel_std,
    #     warp_res=args.vel_res,
    #     blur_std=args.blur_std,
    #     bias_std=args.bias_std,
    #     bias_res=args.bias_res,
    #     gamma_std=args.gamma,
    # )

    # # Create TF model to preprocess the data
    # gen_model = None
    # strategy = "get_strategy"
    # with getattr(tf.distribute, strategy)().scope():
    #     import neurite as ne

    #     # generation
    #     gen_model_1 = ne.models.labels_to_image(**gen_args, id=0)
    #     gen_model_2 = ne.models.labels_to_image(**gen_args, id=1)
    #     ima_1, map_1 = gen_model_1.outputs
    #     ima_2, map_2 = gen_model_2.outputs

    #     # registration
    #     inputs = gen_model_1.inputs + gen_model_2.inputs
    #     gen_model = tf.keras.Model(inputs, outputs=(ima_1, ima_2))

    # Setup Sampler and DataLoader
    # generator = SynthGenerator(gen, gen_model, args.steps, device)
    generator = SynthGenerator(rank, args.data_dir, args.steps, device)

    sampler = DistributedSampler(
        generator,
        # num_replicas=int(world_size),
        # rank=int(local_rank),
        shuffle=False,
        drop_last=False,
    )

    dataloader = DataLoader(generator, shuffle=False, sampler=sampler)

    # Prepare model for training
    # unet architecture
    enc_nf = args.enc if args.enc else [64] * 4
    dec_nf = args.dec if args.dec else [64] * 6

    # sample_image = gen_model(next(gen)[0])[0]
    model = vxm.networks.VxmDense(
        inshape=in_shape,
        nb_unet_features=[enc_nf, dec_nf],
        int_steps=args.int_steps,
        int_downsize=args.int_downsize,
    )

    logging.info(f"{datetime.datetime.now()} Created a new model , device {device}")

    model.to(device)
    model = DDP(model, gradient_as_bucket_view=True)
    model.train()

    # set optimizer
    # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    optimizer = ZeroRedundancyOptimizer(model.parameters(), optimizer_class=torch.optim.Adam, lr=args.lr)
    # bindVerrou.verrou_display_counters()
    training_time = time.time()
    logging.info(f"{datetime.datetime.now()} Training Stared")
    # training loops
    for epoch in range(args.initial_epoch, args.epochs):
        # sampler.set_epoch(epoch)
        dataloader.sampler.set_epoch(epoch)
        # save model checkpoint
        if epoch % 20 == 0:
            torch.save(model.state_dict(), os.path.join(args.model_dir, "%04d.pt" % epoch))

        epoch_loss = []
        epoch_total_loss = []
        epoch_step_time = []

        for step, data in enumerate(dataloader):
            step_start_time = time.time()

            # Pass the synthesized data through the TF model and turn the results into pytorch tensors

            # bindVerrou.verrou_start_instrumentation()
            # run inputs through the model to produce a warped image and flow field
            src, trg = [tensor.squeeze(1) for tensor in data]

            # zero gradients
            optimizer.zero_grad()

            # forward pass
            forward_start_time = time.time()
            y_pred, flow = model(src, trg)
            print(f"Compute time for forward: f{time.time() - forward_start_time }s")

            # calculate total loss
            # maybe add // nb_devices
            loss_start_time = time.time()
            # const = torch.ones(args.batch_size // int(world_size)).squeeze()
            # Dice loss defined as 1 - Dice, vxm Dice returns a negative number so the constant represented the 1
            loss_dice = vxm.losses.Dice().loss(trg, y_pred) + torch.ones(1)
            loss_grad = vxm.losses.Grad("l2", loss_mult=args.reg_param).loss(None, flow)
            loss = loss_dice + loss_grad
            loss_list = [loss_dice, loss_grad]
            epoch_loss.append(loss_list)
            epoch_total_loss.append(loss.item())
            print(f"Compute time for loss: f{time.time() - loss_start_time }s")

            # backpropagate and optimize
            optim_start_time = time.time()
            loss.backward()
            optimizer.step()
            print(f"Compute time for loss: f{time.time() - optim_start_time }s")
            # bindVerrou.verrou_stop_instrumentation()
            # get compute time
            print(f"Compute time for step: f{time.time() - step_start_time}s")
            epoch_step_time.append(time.time() - step_start_time)

        # print epoch info
        epoch_info = "Epoch %d/%d" % (epoch + 1, args.epochs)
        time_info = "%.4f sec/step" % np.mean(epoch_step_time)
        epoch_loss_ungrad = list()
        for step in epoch_loss:
            epoch_loss_ungrad.append([loss.detach().numpy() for loss in step])
        losses_info = ", ".join(["%.4e" % f for f in np.mean(epoch_loss_ungrad, axis=0)])
        loss_info = "loss: %.4e  (%s)" % (np.mean(epoch_total_loss), losses_info)
        if os.environ["RANK"] == "0":
            with open("loss/loss_data.csv", "a") as out_file:
                writer = csv.writer(out_file)
                writer.writerow((epoch_info, time_info, loss_info))
        print(" - ".join((epoch_info, time_info, loss_info)), flush=True)

    # final model save
    dist.barrier()
    if os.environ["RANK"] == "0":
        torch.save(model, os.path.join(args.model_dir, "%04d.pt" % args.epochs))
        logging.info(f"{datetime.datetime.now()} Training Finished: total time:{time.time() - training_time}")
    dist.barrier()

    logging.info(f"{datetime.datetime.now()} final done")

    dist.destroy_process_group()


@record
def setup():
    # prepare model folder
    os.makedirs(args.model_dir, exist_ok=True)
    # initialize the process group
    print(f'Local Rank {os.environ["LOCAL_RANK"]}, Rank {os.environ["RANK"]}, World Size {os.environ["WORLD_SIZE"]}')
    # dist.init_process_group("gloo", world_size=(int(["WORLD_SIZE"])))
    dist.init_process_group("gloo")
    if os.environ["RANK"] == "0":
        with open("loss/loss_data.csv", "w") as out_file:
            writer = csv.writer(out_file)
            writer.writerow(("Epoch", "Sec/Epoch", "Loss"))
    dist.barrier()


@record
class SynthGenerator(Dataset):
    def __init__(self, rank, path, steps, device):
        self.rank = rank
        self.path = os.path.join(path, f"cpu{self.rank}")
        # self.gen = generator
        # self.gen_model = gen_model
        self.epoch_len = steps
        self.device = device

    def __len__(self):
        return self.epoch_len

    def __getitem__(self, index):
        # data = self.gen_model(next(self.gen)[0])
        # data = [tensor.numpy() for tensor in data]
        wait_time = time.time()
        while not os.path.exists(os.path.join(self.path, "data.npz.lock")):
            time.sleep(1)
        data = np.load(os.path.join(self.path, "data.npz"))
        print(f"Rank: {self.rank} waited {time.time() - wait_time}s for data")
        src = torch.from_numpy(data["src"]).to(self.device).float().permute(0, 4, 1, 2, 3)
        trg = torch.from_numpy(data["trg"]).to(self.device).float().permute(0, 4, 1, 2, 3)
        os.remove(os.path.join(self.path, "data.npz"))
        os.remove(os.path.join(self.path, "data.npz.lock"))
        return src, trg


if __name__ == "__main__":
    os.environ["TORCH_CPP_LOG_LEVEL"] = "INFO"
    os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
    main()
