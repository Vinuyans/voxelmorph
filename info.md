# Notes

---

## SynthMorph Model

### Scripts

#### Register.py

Script to register 2 scans.

```bash
python scripts/tf/register.py --moving ./data/filename_normalized.nii.gz --fixed ./data/modified_average_205.nii.gz --model ./data/brains-dice-vel-0.5-res-16-256f.h5 --moved test.nii.gz --warp test-warp.nii.gz
```

- **--moving**: MRI scans: Scan to deform to fit on the fixed
- **--fixed**: MRI scans: Fixed scan that the moving scan will deform to match
- **--model**: The model in .h5 format to use when registering
- **--moved**: filepath to save registered scan
- **--warp**: Filepath to save warped field

#### test.py

```bash
python scripts/tf/test.py  --model model.h5  --pairs pairs.txt  --img-suffix /img.nii.gz  --seg-suffix /seg.nii.gz
```

- **--model**: The model in .h5 format to test
- **--pairs**: A text file with line-by-line space-seperated registration pairs.
- **--img-suffix**: filename/file extension
- **--seg-suffix**: filename/file extension
- **--img-prefix**: path to image
- **--seg-prefix**: path to segmentation

#### pairs.txt

```txt
Text file format
movingDir fixedDir # image pairs to register
```

prefix: /path/to/image
suffix: /image.nii.gz

The items in the text files refer to the directory where each moving/fixed file is

Useful GitHub issue (provides a format): <https://github.com/voxelmorph/voxelmorph/issues/373#issuecomment-1004200091>

### Data

The VoxelMorph repo provides a data folder containing:

- atlas.npz
  - vol.npy (160,192,224)  
  - seg.npy (160,192,224)  
  - train_avg.npy (256)  
- test_scan.npz
  - vol.npy (160,192,224)  
  - seg.npy (160,192,224)
- labels.npz (30)

Both models are also available:

- shapes
- brains

---

## Verrou

Verrou is an instrumentation tool that uses MCA arithmetics to perturbe floating point operations. It can be used to test teh accuracy and stability of applications. The source code can be found [here](https://github.com/edf-hpc/verrou) and the docs [here](https://edf-hpc.github.io/verrou/vr-manual.html#idm6469)

Running the SynthMorph model with Verrou produced fatal exception due to Verrou perturbing functions that are too critical. Therefore we had to find what these functions are and where they're located to be able to exclude them

### Running Verrou

```bash
valgrind --tool=verrou --rounding-mode=random --exclude=<exclusion.tex>\
      python /voxelmorph/scripts/tf/register.py\
      --moving <scan>\
      --fixed <atlas>\
      --model <model>\
      --moved <moved filename>\
      --warp <warped filename>
```

The exclusion file should be in the following format inside a file

```txt
Function Object
```

### Gen-Exclude

You can run verrou  with the `--gen-exclude` flag to see what functions are being called while the application is running.

```bash
valgrind --tool=verrou --rounding-mode=random --gen-exclude=<path>\
      python /voxelmorph/scripts/tf/register.py\
      --moving <scan>\
      --fixed <atlas>\
      --model <model>\
      --moved <moved filename>\
      --warp <warped filename>
```

### Delta Debug

Verrou comes with a debugging tool called verrou_dd_line/verrou_dd_sym which looks for instable symbols and instable lines of compiled code respectively.

To run, you must give a `run_script` and a `cmd_script`. The former runs your application with Verrou and saves the output somewhere. The latter verifies if this output is valid. This must be determined by the user and could be a simple `diff`.

Here is how we ran verrou_dd_line on Compute Canada. **Note: Our cmd script failed but verrou_dd outputs a FullPerturbation sym link for debug purposes which we were able to use to find the function causing the issues**

```bash
module load singularity/3.8

singularity exec --writable-tmpfs\
      --bind /synthmorph/scripts/dd:/voxelmorph/verrou\
      --bind /synthmorph/scans:/voxelmorph/scans\
      synthmorph.sif verrou_dd_sym /voxelmorph/verrou/run_script.sh /voxelmorph/verrou/cmd_script.sh
```

```bash
#!/bin/bash
valgrind --tool=verrou --rounding-mode=random \
 python /voxelmorph/load_model.py --moving /voxelmorph/scans/norm-sub-0003002.mgz\
 --fixed /voxelmorph/data/norm-average_mni305.mgz --model /voxelmorph/data/brains-dice-vel-0.5-res-16-256f.h5\
 --moved /voxelmorph/verrou/sub-0003002-moved.nii.gz --warp /voxelmorph/verrou/sub-0003002-warped.nii.gz 
```

```bash
# Since there were no output, we couldn't compare anything.
# If you need to run this yourself, save the weights of the model when loading and compare it with the baseline results.
```
