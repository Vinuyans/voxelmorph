# Data
The original paper used the Buckner40 dataset, but it is not publicly available due to legal reasons. We decided to use the [CoRR dataset](http://datasets.datalad.org/?dir=/corr/RawDataBIDS) instead. We initially took 35 subjects across 35 of CoRR's datasets when we were testing inference. We then added 5 more to match the size of the Buckner40 training dataset. 

## Subjects

| Dataset | Subject Id | Session |
|---------|------------|---------|
| BMB_1   | sub-0003002|  ses-1  |
|**BMB_1**\*| sub-0003010|  ses-1  |
| BNU_1   | sub-0025879|  ses-2  |
| BNU_2   | sub-0025930|  ses-2  |
|**BNU_2**\*| sub-0025946|  ses-1  |
| BNU_3   | sub-0027094|  ses-1  |
|**BNU_3**\*| sub-0027085|  ses-1  |
| DC_1    | sub-0027393|  ses-1  |
|**DC_1**\*| sub-0027402|  ses-1  |
| HNU_1   | sub-0025436|  ses-9  |
|**HNU_1**\*| sub-0025429|  ses-1  |
| IACAS   | sub-0025462|  ses-1  |
| IBA_TRT | sub-0027236|  ses-1  |
| IPCAS_1 | sub-0025507|  ses-2  |
| IPCAS_2 | sub-0025531|  ses-1  |
| IPCAS_3 | sub-0025555|  ses-1  |
| IPCAS_4 | sub-0026196|  ses-1  |
| IPCAS_5 | sub-0027294|  ses-1  |
| IPCAS_6 | sub-0026044|  ses-1  |
| IPCAS_7 | sub-0026055|  ses-1  |
| IPCAS_8 | sub-0025598|  ses-2  |
| JHNU    | sub-0025604|  ses-1  |
| LMU_1   | sub-0025350|  ses-1  |
| LMU_2   | sub-0025362|  ses-1  |
| LMU_3   | sub-0025406|  ses-2  |
| MPG_1   | sub-0027434|  ses-1  |
| MRN     | sub-0027012|  ses-1  |
| NKI_TRT | sub-2842950|  ses-1  |
| NYU_1   | sub-0027110|  ses-1  |
| NYU_2   | sub-0025011|  ses-1  |
| SWU_1   | sub-0027214|  ses-2  |
| SWU_2   | sub-0027191|  ses-1  |
| SWU_3   | sub-0027174|  ses-2  |
| SWU_4   | sub-0025631|  ses-1  |
| UM      | sub-0026175|  ses-1  |
| UPSM_1  | sub-0025248|  ses-1  |
| Utah_1  | sub-0026036|  ses-1  |
| Utah_2  | sub-0026017|  ses-8  |
| UWM     | sub-0027268|  ses-1  |
| XHCUMS  | sub-0025993|  ses-4  |

## Pre-processing

### Inference
The subjects were processed using Freesurfer's recon-all command:
```bash
recon-all -verbose -sd /root -subjid sub-01 \
	-motioncor \
	-talairach \
	-nuintensitycor \
	-normalization \
	-skullstrip \
	-gcareg \
	-canorm
```
And then cropped as such:
```python
img = nib.load('subject.mgz')
#160,192,224,
data = img.get_fdata()
data = data[40:200, 20:212, 20:244]
neuro_img = nib.nifti1.Nifti1Image(data, img.affine) 
nib.save(neuro_img, 'subject-cropped.nii.gz')
```
### Training

For training, we needed the brain segments of the subjects. We do this by running Freesurfer's run_samseg command on the cropped subject data:
```bash
run_samseg --input /scans/subject.nii.gz --output /output 
```
