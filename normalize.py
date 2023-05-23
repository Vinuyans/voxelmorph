import nibabel as nib
import numpy as np
import click as ck
import torchio as tio
from pathlib import Path


@ck.command()
@ck.option("--input", "-i", type=ck.Path(), help="Input dir to normalize")
def main(input):
    dir_path = Path(input)
    for file_path in dir_path.iterdir():
        img = nib.load(file_path)
        img2 = tio.ScalarImage(file_path)
        RI_norm = tio.RescaleIntensity(out_min_max=(0, 1))
        normed_img = RI_norm(img2)
        neuro_img = nib.nifti1.Nifti1Image(normed_img.numpy().squeeze(), img.affine)
        nib.save(neuro_img, f"norm-{file_path.name}")


if __name__ == "__main__":
    main()
