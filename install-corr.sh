#!/bin/bash
# datalad install the corr dataset into a folder called /corr
# Dataset foudn at http://datasets.datalad.org/?dir=/corr/RawDataBIDS/

# Excluded due to orientation issues
# "./corr/RawDataBIDS/DC_1/sub-0027393/ses-1/anat/sub-0027393_ses-1_run-1_T1w.nii.gz"
# "./corr/RawDataBIDS/DC_1/sub-0027402/ses-1/anat/sub-0027402_ses-1_run-1_T1w.nii.gz"

file_paths=(
  "./corr/RawDataBIDS/BMB_1/sub-0003002/ses-1/anat/sub-0003002_ses-1_run-1_T1w.nii.gz"
  "./corr/RawDataBIDS/BNU_1/sub-0025879/ses-2/anat/sub-0025879_ses-2_run-1_T1w.nii.gz"
  "./corr/RawDataBIDS/BNU_2/sub-0025930/ses-2/anat/sub-0025930_ses-2_run-1_T1w.nii.gz"
  "./corr/RawDataBIDS/BNU_3/sub-0027094/ses-1/anat/sub-0027094_ses-1_run-1_T1w.nii.gz"
  "./corr/RawDataBIDS/HNU_1/sub-0025436/ses-9/anat/sub-0025436_ses-9_run-1_T1w.nii.gz"
  "./corr/RawDataBIDS/IACAS/sub-0025462/ses-1/anat/sub-0025462_ses-1_run-1_T1w.nii.gz"
  "./corr/RawDataBIDS/IBA_TRT/sub-0027236/ses-1/anat/sub-0027236_ses-1_run-1_T1w.nii.gz"
  "./corr/RawDataBIDS/IPCAS_1/sub-0025507/ses-2/anat/sub-0025507_ses-2_run-1_T1w.nii.gz"
  "./corr/RawDataBIDS/IPCAS_2/sub-0025531/ses-1/anat/sub-0025531_ses-1_run-1_T1w.nii.gz"
  "./corr/RawDataBIDS/IPCAS_3/sub-0025555/ses-1/anat/sub-0025555_ses-1_run-1_T1w.nii.gz"
  "./corr/RawDataBIDS/IPCAS_4/sub-0026196/ses-1/anat/sub-0026196_ses-1_run-1_T1w.nii.gz"
  "./corr/RawDataBIDS/IPCAS_5/sub-0027294/ses-1/anat/sub-0027294_ses-1_run-1_T1w.nii.gz"
  "./corr/RawDataBIDS/IPCAS_6/sub-0026044/ses-1/anat/sub-0026044_ses-1_run-1_T1w.nii.gz"
  "./corr/RawDataBIDS/IPCAS_7/sub-0026055/ses-1/anat/sub-0026055_ses-1_run-1_T1w.nii.gz"
  "./corr/RawDataBIDS/IPCAS_8/sub-0025598/ses-2/anat/sub-0025598_ses-2_run-1_T1w.nii.gz"
  "./corr/RawDataBIDS/JHNU/sub-0025604/ses-1/anat/sub-0025604_ses-1_run-1_T1w.nii.gz"
  "./corr/RawDataBIDS/LMU_1/sub-0025350/ses-1/anat/sub-0025350_ses-1_run-1_T1w.nii.gz"
  "./corr/RawDataBIDS/LMU_2/sub-0025362/ses-1/anat/sub-0025362_ses-1_run-1_T1w.nii.gz"
  "./corr/RawDataBIDS/LMU_3/sub-0025406/ses-2/anat/sub-0025406_ses-2_run-1_T1w.nii.gz"
  "./corr/RawDataBIDS/MPG_1/sub-0027434/ses-1/anat/sub-0027434_ses-1_acq-uni_run-1_T1w.nii.gz"
  "./corr/RawDataBIDS/MRN/sub-0027012/ses-1/anat/sub-0027012_ses-1_run-1_T1w.nii.gz"
  "./corr/RawDataBIDS/NKI_TRT/sub-2842950/ses-1/anat/sub-2842950_ses-1_run-1_T1w.nii.gz"
  "./corr/RawDataBIDS/NYU_1/sub-0027110/ses-1/anat/sub-0027110_ses-1_run-1_T1w.nii.gz"
  "./corr/RawDataBIDS/NYU_2/sub-0025011/ses-1/anat/sub-0025011_ses-1_run-1_T1w.nii.gz"
  "./corr/RawDataBIDS/SWU_1/sub-0027214/ses-2/anat/sub-0027214_ses-2_run-1_T1w.nii.gz"
  "./corr/RawDataBIDS/SWU_2/sub-0027191/ses-1/anat/sub-0027191_ses-1_run-1_T1w.nii.gz"
  "./corr/RawDataBIDS/SWU_3/sub-0027174/ses-2/anat/sub-0027174_ses-2_run-1_T1w.nii.gz"
  "./corr/RawDataBIDS/SWU_4/sub-0025631/ses-1/anat/sub-0025631_ses-1_run-1_T1w.nii.gz"
  "./corr/RawDataBIDS/UM/sub-0026175/ses-1/anat/sub-0026175_ses-1_run-1_T1w.nii.gz"
  "./corr/RawDataBIDS/UPSM_1/sub-0025248/ses-1/anat/sub-0025248_ses-1_run-1_T1w.nii.gz"
  "./corr/RawDataBIDS/Utah_1/sub-0026036/ses-1/anat/sub-0026036_ses-1_run-1_T1w.nii.gz"
  "./corr/RawDataBIDS/Utah_2/sub-0026017/ses-8/anat/sub-0026017_ses-8_run-1_T1w.nii.gz"
  "./corr/RawDataBIDS/UWM/sub-0027268/ses-1/anat/sub-0027268_ses-1_run-1_T1w.nii.gz"
  "./corr/RawDataBIDS/XHCUMS/sub-0025993/ses-4/anat/sub-0025993_ses-4_run-1_T1w.nii.gz"
  "./corr/RawDataBIDS/BMB_1/sub-0003010/ses-1/anat/sub-0003010_ses-1_run-1_T1w.nii.gz"
  "./corr/RawDataBIDS/HNU_1/sub-0025429/ses-2/anat/sub-0025429_ses-2_run-1_T1w.nii.gz"
  "./corr/RawDataBIDS/BNU_2/sub-0025946/ses-1/anat/sub-0025946_ses-1_run-1_T1w.nii.gz"
  "./corr/RawDataBIDS/BNU_3/sub-0027085/ses-1/anat/sub-0027085_ses-1_run-1_T1w.nii.gz"
  "./corr/RawDataBIDS/LMU_1/sub-0025356/ses-1/anat/sub-0025356_ses-1_run-1_T1w.nii.gz"
  "./corr/RawDataBIDS/JHNU/sub-0025612/ses-1/anat/sub-0025612_ses-1_run-1_T1w.nii.gz"
  "./corr/RawDataBIDS/NYU_2/sub-0025114/ses-1/anat/sub-0025114_ses-1_run-1_T1w.nii.gz"
  "./corr/RawDataBIDS/IPCAS_1/sub-0025499/ses-1/anat/sub-0025499_ses-1_run-1_T1w.nii.gz"
  "./corr/RawDataBIDS/IPCAS_2/sub-0025535/ses-1/anat/sub-0025535_ses-1_run-1_T1w.nii.gz"
)
for file in "${file_paths[@]}"; do
  :
  datalad get "$file"
done
