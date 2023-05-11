# Register.py

Script to register 2 scans.

```bash
python scripts/tf/register.py --moving ./data/test_scan.npz --fixed ./data/modified_average_205.nii.gz -
-model ./data/brains-dice-vel-0.5-res-16-256f.h5 --moved test.nii.gz --warp test-warp.nii.gz
```


python /voxelmorph/scripts/tf/register.py --moving /voxelmorph/scans/sub-0003002.nii.gz --fixed /voxelmorph/data/modified_average_205.nii.gz --model /voxelmorph/data/brains-dice-vel-0.5-res-16-256f.h5 --moved /voxelmorph/nifty/test.nii.gz --warp /voxelmorph/nifty/test-warp.nii.gz

- **--moving**: MRI scans: Scan to deform to fit on the fixed
- **--fixed**: MRI scans: Fixed scan that the moving scan will deform to match
- **--model**: The model in .h5 format to use when registering
- **--moved**: filepath to save registered scan
- **--warp**: Filepath to save warped field

# test.py

```bash
python scripts/tf/test.py  --model model.h5  --pairs pairs.txt  --img-suffix /img.nii.gz  --seg-suffix /seg.nii.gz
```

- **--model**: The model in .h5 format to test
- **--pairs**: A text file with line-by-line space-seperated registration pairs.
- **--img-suffix**: filename/file extension
- **--seg-suffix**: filename/file extension
- **--img-prefix**: path to image
- **--seg-prefix**: path to segmentation

## pairs.txt

```txt
Text file format
movingDir fixedDir # image pairs to register
```

prefix: /path/to/image
suffix: /image.nii.gz

The items in the text files refer to the directory where each moving/fixed file is

Useful GitHub issue (provides a format): <https://github.com/voxelmorph/voxelmorph/issues/373#issuecomment-1004200091>

**When we are going to be testing the model, we will need access to fixed and moving MRI scans. the pairs.txt file will most likely need to be made by hand. From reading online uses of voxelmorph, I saw OASIS come up a lot as a database for MRI scans.**

# Notes 

## Data

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

## Other

**What is *Image registration***: To register two images means to align them, so that common features overlap and differences, should there be any, between the two are
emphasized and readily visible to the naked eye. We refer to the process of
aligning two images as image registration

Atlases (also refered to as templates)

data-lad get 37 random samples
remove segmentation reference in test,py