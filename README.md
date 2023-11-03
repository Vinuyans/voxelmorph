# Numerical Stabiity of Synthmorph

This repo was forked from the [voxelmorph](https://github.com/voxelmorph/voxelmorph)

## Goal

The aim of this experiment is to test the stability of both the inference and training of the Synthmorph model.


### Inference
To test the stability of the inference, we used [Verrou](https://edf-hpc.github.io/verrou/vr-manual.html) which perturbs all floating-point operations by changing the random mode. We also used a functionality of Verrou called [Interlibmath](https://github.com/edf-hpc/verrou/tree/master/Interlibmath) to intercept libm calls and replace it with an equivalent rounding mode since perturbating libm directly would cause errors. Finally, we also utilized another functionality called [SynchroLib](https://github.com/edf-hpc/verrou/tree/master/synchroLib) that can programatically start and stop Verrou to avoid pertrubating uneccessary operations.

We used 35 subjects randomly picked from the Corr dataset.

For more information on how we ran our experiment, please look at [inference_info.md](./notes/inference_info.md)

To see details of our results please look at the [paper](https://arxiv.org/pdf/2308.01939v1.pdf)

### Training 
We convert the training portion to be done in Pytorch, but we keep the original processing model that was used on the training data before being passed to the core Unet in Tensorflow. We then convert the TensorFlow tensor from the processing model into a Pytorch and continue.
We switched to Pytorch because Tensorflow does not provide support to data paralelism with multiple cpus. For more [information](./notes/tf_parralelism_issues.md) 

We took an additional 5 subjects from the Corr dataset to more acurately follow the original paper's 40 subject.

## Structure
<pre>
. 
├── data                          # <b>From Voxelmorph:</b> Contains basic data to run scripts (models not included)
├── notes                   
    ├── model_summary             # Contains outputs of Tensorflow's summary function of the synthmorph model during training.
    ├── inference_info.md                   # File describing how to register with the provided  files and how to run Verrou.
    ├── tf_parralelism_issues.md  # Describes the issues regarding implementing data paralelism with Tensorflow using only cpus.
├── scripts                       # <b>From Voxelmorph:</b> Scripts to train and register the models
├── utils_scripts                 #
    ├── normalize.py              # Normalizes NIfTI files
    ├── view.ipynb                # Sample code to vizualize NIfTI files
├── voxelmorph                    # <b>From Voxelmorph:</b> Voxelmorph library
├── Dockerfile                    # Dockerfile to build environment containing Verrou 
├── install-corr.sh               # Gets the dataset using Datalad (Assumes you've already installed the Corr dataset).
├── train_single.py               # The Synthmorph training script converted to use PyTorch. Runs on a single machine
├── train_ddp.py                  # The Synthmorph training script converted to use PyTorch and Distribute Data Paralelism. Runs with torchrun on multiple cpus.
└── README.md
</pre>