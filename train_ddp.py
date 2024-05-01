#!/usr/bin/env python

"""
Script to train Synthmorph with pytorch.
Based on the /scripts/tf/train_synthmoprh.py and /scripts/torch/train.py scripts found on the voxelmorph repo (https://github.com/voxelmorph/voxelmorph)

Citations: 

    VoxelMorph: A Learning Framework for Deformable Medical Image Registration G. Balakrishnan, A.
    Zhao, M. R. Sabuncu, J. Guttag, A.V. Dalca. IEEE TMI: Transactions on Medical Imaging. 38(8). pp
    1788-1800. 2019. 
"""
# import scipy
# import scipy.linalg
import os
import threading

from matplotlib.pylab import f

print(f"Rank: {os.environ['RANK']}, PID:{os.getpid()}, TID:{threading.get_native_id()}")

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

# from torch.utils.tensorboard import SummaryWriter

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
parser.add_argument("--model-dir", default="/voxelmorph/models", help="model output directory")
parser.add_argument("--log-dir", help="optional TensorBoard log directory")
parser.add_argument("--sub-dir", help="optional subfolder for logs and model saves")
parser.add_argument("--load", help="Path to model to continue training from")

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
    # board
    # if os.environ["RANK"] == "0":
    #     # Setup Tensorboard
    #     board = SummaryWriter()
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = os.environ["RANK"]
    # device = torch.device("cpu")
    print(f" Rank {torch.distributed.get_rank()}", flush=True)
    # local_rank = os.environ['LOCAL_RANK'] #alternative option
    # print(torch.device("cpu", local_rank))  # should all return same thing
    device = torch.device("cpu", local_rank)
    print(
        f"{datetime.datetime.now()} Running synthmorph on {device}; local rank:{local_rank}; global rank {rank}",
        flush=True,
    )

    # Load labels & create Synthmorph generator
    labels_in, label_maps = vxm.py.utils.load_labels(args.label_dir)
    print(f"{datetime.datetime.now()} Loaded labels and created Synthmorph generator [Rank {rank}]", flush=True)

    in_shape = label_maps[0].shape

    # Setup Sampler and DataLoader
    generator = SynthGenerator(rank, args.data_dir, args.steps, device)
    print(f"{datetime.datetime.now()} Created generator [Rank {rank}]", flush=True)

    sampler = DistributedSampler(
        generator,
        num_replicas=world_size,
        rank=local_rank,
        shuffle=False,
        drop_last=False,
    )
    print(f"{datetime.datetime.now()} Created sampler [Rank {rank}]", flush=True)

    dataloader = DataLoader(generator, shuffle=False, sampler=sampler)
    print(f"{datetime.datetime.now()} Created dataloader [Rank {rank}]", flush=True)

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

    print(f"{datetime.datetime.now()} Created a new model , device {device}", flush=True)

    model.to(device)
    print("Model moved to device", flush=True)
    # , device_ids=[dist.get_rank()], output_device=dist.get_rank(),
    model = DDP(model, gradient_as_bucket_view=True)
    print("Model wrapped in DDP", flush=True)

    # set optimizer
    # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    optimizer = ZeroRedundancyOptimizer(model.parameters(), optimizer_class=torch.optim.Adam, lr=args.lr)
    print(f"{datetime.datetime.now()} Created optimizer", flush=True)

    # bindVerrou.verrou_display_counters()
    initial_epoch = 0
    # Load model params
    if args.load:
        checkpoint = torch.load(args.load)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        initial_epoch = checkpoint["epoch"]

    training_time = time.time()
    print(f"{datetime.datetime.now()} Training Stared", flush=True)

    model.train()
    print(f"{datetime.datetime.now()} Model set to training mode", flush=True)

    # TODO Can add inputs to graph
    # if os.environ["RANK"] == "0":
    #     board.add_graph(model)
    # training loops
    for epoch in range(initial_epoch, args.epochs):
        print(f"{datetime.datetime.now()} Starting epoch {epoch}", flush=True)
        # sampler.set_epoch(epoch)
        dataloader.sampler.set_epoch(epoch)
        print(f"{datetime.datetime.now()} Set sampler to epoch {epoch}", flush=True)
        # save model checkpoint
        if epoch % 10 == 0:
            optimizer.consolidate_state_dict(to=0)
            print(f"{datetime.datetime.now()} Consolidated state dict", flush=True)
            dist.barrier()
            print(f"{datetime.datetime.now()} First barrier passed", flush=True)
            if os.environ["RANK"] == "0":
                print(f"{datetime.datetime.now()} Saving model", flush=True)
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                    },
                    os.path.join(args.model_dir, "%04d.pt" % epoch),
                )
            print(f"{datetime.datetime.now()} Model saved", flush=True)
            dist.barrier()
            print(f"{datetime.datetime.now()} Second barrier passed", flush=True)

        step_losses = []
        step_total_loss = []
        epoch_step_time = []

        for step, data in enumerate(dataloader):
            print(f"{datetime.datetime.now()} Starting step {step}", flush=True)
            step_start_time = time.time()

            src, trg, map_2 = [tensor.squeeze(1) for tensor in data]
            print(f"{datetime.datetime.now()} Loaded data", flush=True)

            # zero gradients
            optimizer.zero_grad()
            # print(f"{datetime.datetime.now()} Zeroed gradients", flush=True)

            # forward pass
            y_pred, flow = model(src, trg)
            # print(f"{datetime.datetime.now()} Forward pass done", flush=True)

            # calculate total loss
            # const = args.steps // world_size
            const = 1
            loss_dice = vxm.losses.Dice().loss(map_2, y_pred) + const
            # print(f"{datetime.datetime.now()} Calculated Dice loss", flush=True)
            loss_grad = vxm.losses.Grad("l2", loss_mult=args.reg_param).loss(None, flow)
            # print(f"{datetime.datetime.now()} Calculated Gradient Descent loss", flush=True)
            # See https://stackoverflow.com/questions/53994625/how-can-i-process-multi-loss-in-pytorch
            loss = loss_dice + loss_grad
            loss_list = [loss_dice, loss_grad]
            step_losses.append(loss_list)
            step_total_loss.append(loss.item())

            # backpropagate and optimize
            loss.backward()
            # print(f"{datetime.datetime.now()} Backpropagated and optimized", flush=True)
            optimizer.step()
            print(f"{datetime.datetime.now()} Step done", flush=True)

            # get compute time
            print(f"Rank {rank}: Compute time for step: f{time.time() - step_start_time}s", flush=True)
            epoch_step_time.append(time.time() - step_start_time)

        # print epoch info
        epoch_info = "Epoch %d/%d" % (epoch + 1, args.epochs)
        time_info = "%.4f sec/step" % np.mean(epoch_step_time)
        epoch_loss_ungrad = list()
        for step in step_losses:
            epoch_loss_ungrad.append([loss.detach().numpy() for loss in step])
        losses_info = ", ".join(["%.4e" % f for f in np.mean(epoch_loss_ungrad, axis=0)])
        loss_info = "loss: %.4e  (%s)" % (np.mean(step_total_loss), losses_info)
        if os.environ["RANK"] == "0":
            with open("/voxelmorph/loss/loss_data.csv", "a") as out_file:
                writer = csv.writer(out_file)
                writer.writerow(
                    (
                        epoch_info,
                        time_info,
                        np.mean(step_total_loss),
                        step_losses[-1][0].item(),
                        step_losses[-1][1].item(),
                    )
                )
        print(" - ".join((epoch_info, time_info, loss_info)), flush=True)
        dist.barrier(group=torch.distributed.group.WORLD)

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
    dist.init_process_group("gloo")
    if os.environ["RANK"] == "0":
        # Create flie to write loss
        with open("/voxelmorph/loss/loss_data.csv", "a") as out_file:
            writer = csv.writer(out_file)
            # writer.writerow(("Epoch", "Sec/Epoch", "Loss (Total)", "Loss (Dice)", "Loss (Grad)"))
    dist.barrier()


@record
class SynthGenerator(Dataset):
    def __init__(self, rank: str, path: str, steps: int, device: torch.device):
        self.rank = rank
        self.path = os.path.join(path, f"cpu{self.rank}")
        self.epoch_len = steps
        self.device = device

    def __len__(self):
        return self.epoch_len

    def __getitem__(self, index):
        """
        Gets the npz file created by the data generation script and loads it.
        index is never used since rank is used to determine which data to take,
        but it is needed for the function definition

        Args:
            index : Not used

        Returns:
            tuple[Tensor. Tensor, Tensor]: returns the src, trg and map_2 that was generated as Tensors
        """
        wait_time = time.time()
        print(f"Rank: {self.rank} waiting for data", flush=True)
        while not os.path.exists(os.path.join(self.path, "data.npz.lock")):
            print(f"Rank: {self.rank} waiting for data", flush=True)
            time.sleep(1)
        # print(f"Rank: {self.rank} waited {time.time() - wait_time}s for lock", flush=True)
        data = np.load(os.path.join(self.path, "data.npz"))
        # print(f"Rank: {self.rank} waited {time.time() - wait_time}s for data", flush=True)
        try:
            src = torch.from_numpy(data["src"]).to(self.device).float().permute(0, 4, 1, 2, 3)
            trg = torch.from_numpy(data["trg"]).to(self.device).float().permute(0, 4, 1, 2, 3)
            map_2 = torch.from_numpy(data["map_2"]).to(self.device).float().permute(0, 4, 1, 2, 3)
            os.remove(os.path.join(self.path, "data.npz"))
            os.remove(os.path.join(self.path, "data.npz.lock"))
            return src, trg, map_2
        except:
            os.remove(os.path.join(self.path, "data.npz"))
            os.remove(os.path.join(self.path, "data.npz.lock"))


if __name__ == "__main__":
    os.environ["TORCH_CPP_LOG_LEVEL"] = "INFO"
    os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
    torch.manual_seed(0)
    main()
