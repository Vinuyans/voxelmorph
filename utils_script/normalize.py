import nibabel as nib
import click as ck
import torchio as tio
from pathlib import Path
import os


@ck.command()
@ck.option("--input", "-i", type=ck.Path(), help="Input dir to normalize")
@ck.option("--output", "-o", type=ck.Path(), help="Output directory. Where all the normalized data will be saved")
def main(input, output):
    """
    Normalize NIfTI images in a directory.

    Args:
        input (str): Input directory path.
        output(str): Output directory path
    Returns:
        None
    """
    dir_path = Path(input)
    for file_path in dir_path.iterdir():
        img = nib.load(file_path)
        img2 = tio.ScalarImage(file_path)
        RI_norm = tio.RescaleIntensity(out_min_max=(0, 1))
        normed_img = RI_norm(img2)
        neuro_img = nib.nifti1.Nifti1Image(normed_img.numpy().squeeze(), img.affine)
        nib.save(neuro_img, os.path.join(output, f"norm-{file_path.name}"))


if __name__ == "__main__":
    main()
