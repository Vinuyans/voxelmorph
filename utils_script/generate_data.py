import os
import lzma
import pickle
import argparse
import numpy as np
import tensorflow as tf
import voxelmorph as vxm
import pdb
import h5py
import neurite as ne
import tensorflow.keras.layers as KL
# parse command line
bases = (argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter)
p = argparse.ArgumentParser(
    formatter_class=type("formatter", bases, {}),
    description=f"Train a SynthMorph model on images synthesized from label maps. ",
)

# data organization parameters
p.add_argument(
    "--label-dir", nargs="+", help="path or glob pattern pointing to input label maps"
)
# generation parameters
p.add_argument(
    "--same-subj", action="store_true", help="generate image pairs from same label map"
)
p.add_argument("--blur-std", type=float, default=1, help="maximum blurring std. dev.")
p.add_argument("--gamma", type=float, default=0.25, help="std. dev. of gamma")
p.add_argument("--vel-std", type=float, default=0.5, help="std. dev. of SVF")
p.add_argument("--vel-res", type=float, nargs="+", default=[16], help="SVF scale")
p.add_argument("--bias-std", type=float, default=0.3, help="std. dev. of bias field")
p.add_argument("--bias-res", type=float, nargs="+", default=[40], help="bias scale")
p.add_argument("--out-shape", type=int, nargs="+", help="output shape to pad to" "")
p.add_argument(
    "--out-labels", default="fs_labels.npy", help="labels to optimize, see README"
)

p.add_argument("--epochs", type=int, default=1500, help="training epochs")
p.add_argument("--batch-size", type=int, default=1, help="batch size")



arg = p.parse_args()

f = h5py.File("mytestfile.hdf5", "w")
src = f.create_group("src")
trg = f.create_group("trg")
labels_in, label_maps = vxm.py.utils.load_labels(arg.label_dir)
gen = vxm.generators.synthmorph(
    label_maps,
    batch_size=arg.batch_size,
    same_subj=arg.same_subj, # Should be true according to paper TODO Check
    flip=True,
)
in_shape = label_maps[0].shape

# model configuration
gen_args = dict(
    in_shape=in_shape,
    out_shape=arg.out_shape,
    in_label_list=labels_in,
    # out_label_list=labels_out,
    warp_std=arg.vel_std,
    warp_res=arg.vel_res,
    blur_std=arg.blur_std,
    bias_std=arg.bias_std,
    bias_res=arg.bias_res,
    gamma_std=arg.gamma,
)

gen_model = None
# build model
strategy = 'get_strategy'
with getattr(tf.distribute, strategy)().scope():

    # generation
    gen_model_1 = ne.models.labels_to_image(**gen_args, id=0)
    gen_model_2 = ne.models.labels_to_image(**gen_args, id=1)
    ima_1, map_1 = gen_model_1.outputs
    ima_2, map_2 = gen_model_2.outputs

    # registration
    inputs = gen_model_1.inputs + gen_model_2.inputs
    gen_model = tf.keras.Model(inputs, outputs=(ima_1, ima_2))

    steps_per_epoch = 1
    for epoch in range(arg.epochs):
        for step in range(steps_per_epoch):
            # Generator returns [src,trg], [void] * 2, the latter being the y_true
            data= next(gen)[0]
            x = gen_model(data)
            pickle.dump(x[0], lzma.open('data.pkl.lzma', 'wb'))    
#             src.create_dataset(f"{epoch * steps_per_epoch + step}",data=x[0],compression='gzip',compression_opts=9)
#             trg.create_dataset(f"{epoch * steps_per_epoch + step}",data=x[1],compression='gzip',compression_opts=9)
# f.close()

#             src.append(x[0])
#             trg.append(x[1])
# # np.savez_compressed(f"generated_data_batch{arg.batch_size}_epochs{arg.epochs}.npz", src=src, trg=trg)