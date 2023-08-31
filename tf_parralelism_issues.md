# Issues Encountered with TensorFlow 

This document aims to summarize issues encountered trying to make Distrubuted Strategies from the Tensorflow Library work on only CPUs and on a multi-nodal architecture

## Mirrored Strategy

Mirrored Strategy is intended to utilize multiple "devices" on one machine to do synchronous training distributed training. It normaly relies on multiple GPUs or TPUs, but the issue arises when you try to utilize multiple CPUs.
In Tensorflow, all CPUs on a machine are grouped together as one CPU device that Tensorflow can use to perform operations with. So no matter how many CPUs there are on a single machine it couldn't perform distributed training using this strategy, but there would be a speed up since there are more ressources.
This had similar runtimes to using the default strategy that doesn't implement parallelism.

## Multi Worker Mirrored Strategy

When working with this strategy, we need to provide a way to resolve which worker is which and way for them to communicate. Since we're working with Slurm and don't have access to physical machines, we should use the built-in SlurmClusterResolver. The issue is that this class was built with GPUs in mind and crashes when none are available. It has checks to see if gpus are available and locks the maximum number of worker to be equal to the maximum number of GPUs available.

[Similar Issue](https://github.com/tensorflow/tensorflow/issues/56894) (They're fix did not work for me)
