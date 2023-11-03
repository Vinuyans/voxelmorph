import os
import argparse
import shutil
from pathlib import Path

import numpy as np
import tensorflow as tf
import neurite as ne
import voxelmorph as vxm

# parse command line
p = argparse.ArgumentParser()
# data organization parameters
p.add_argument("--label-dir", required=True, nargs="+", help="path or glob pattern pointing to input label maps")
# generation parameters
p.add_argument("--same-subj", action="store_true", help="generate image pairs from same label map")
p.add_argument("--blur-std", type=float, default=1, help="maximum blurring std. dev.")
p.add_argument("--gamma", type=float, default=0.25, help="std. dev. of gamma")
p.add_argument("--vel-std", type=float, default=0.5, help="std. dev. of SVF")
p.add_argument("--vel-res", type=float, nargs="+", default=[16], help="SVF scale")
p.add_argument("--bias-std", type=float, default=0.3, help="std. dev. of bias field")
p.add_argument("--bias-res", type=float, nargs="+", default=[40], help="bias scale")
p.add_argument("--out-shape", type=int, nargs="+", help="output shape to pad to" "")
p.add_argument("--out-labels", default="fs_labels.npy", help="labels to optimize, see README")
p.add_argument("--epochs", type=int, default=1500, help="training epochs")
p.add_argument("--batch-size", type=int, default=1, help="batch size")
# File saving params
p.add_argument("--data-dir", required=True, help="Path to folder containing cpu data directories")
p.add_argument("--filename", type=str, default="data.npz", help="Name of the data file")
p.add_argument("--ncpus", type=int, help="Number of total cpus")
# p.add_argument("--cpu", type=int, help="Designated CPU to generate data for")
p.add_argument("--gpu", type=str, default="0", help="ID of GPU to use")

args = p.parse_args()


def build_generators():
    # Load labels & create Synthmorph generator
    labels_in, label_maps = vxm.py.utils.load_labels(args.label_dir)
    gen = vxm.generators.synthmorph(
        label_maps,
        batch_size=args.batch_size,
        same_subj=args.same_subj,  # Should be true according to paper TODO Check
        flip=True,
    )
    in_shape = label_maps[0].shape

    # model configuration
    gen_args = dict(
        in_shape=in_shape,
        out_shape=args.out_shape,
        in_label_list=labels_in,
        warp_std=args.vel_std,
        warp_res=args.vel_res,
        blur_std=args.blur_std,
        bias_std=args.bias_std,
        bias_res=args.bias_res,
        gamma_std=args.gamma,
    )

    # Create TF model to preprocess the data
    gen_model = None
    strategy = "get_strategy"
    with getattr(tf.distribute, strategy)().scope():
        # generation
        gen_model_1 = ne.models.labels_to_image(**gen_args, id=0)
        gen_model_2 = ne.models.labels_to_image(**gen_args, id=1)
        ima_1, map_1 = gen_model_1.outputs
        ima_2, map_2 = gen_model_2.outputs

        # registration
        inputs = gen_model_1.inputs + gen_model_2.inputs
        gen_model = tf.keras.Model(inputs, outputs=(ima_1, ima_2))
    return gen_model, gen


def save_results(model, gen, path: str):
    src, trg = model(next(gen)[0])
    np.savez_compressed(os.path.join(path, args.filename), src=src.numpy(), trg=trg.numpy())
    with open(os.path.join(path, args.filename + ".lock"), "w") as file:
        pass


# TODO: If multiprocessing data generation, implement args.cpu
if __name__ == "__main__":
    print("Setup TF")
    device, nb_devices = vxm.tf.utils.setup_device(args.gpu)
    model, gen = build_generators()
    print("Finished Setup")
    # If data directory exists, empty it
    # if os.path.exists(args.data_dir):
    #     shutil.rmtree(args.data_dir)
    # os.mkdir(args.data_dir)

    # Initialize directores
    paths = []
    for cpu in range(args.ncpus):
        paths.append(os.path.join(args.data_dir, f"cpu{cpu}"))
        os.mkdir(paths[cpu])
    print("Initialized dirs")
    while True:
        for cpu in range(args.ncpus):
            if not any(Path(paths[cpu]).iterdir()):
                print(f"Writting to CPU:{cpu}")
                save_results(model, gen, paths[cpu])
                print(f"Data written for CPU:{cpu}")


