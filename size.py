import numpy as np

# Load the .npz file
data = np.load("data/atlas.npz")

# Get the keys of the arrays stored in the file
keys = list(data.keys())

# Initialize minimum and maximum values to None
amin = None
amax = None

# Iterate over the keys and find the minimum and maximum values for each array
for key in keys:
    array = data[key]
    if amin is None:
        amin = np.amin(array)
    else:
        amin = min(amin, np.amin(array))
    if amax is None:
        amax = np.amax(array)
    else:
        amax = max(amax, np.amax(array))
    if amax == 85.0:
        print("here")
# Print the results
print(type(amin))
print("Minimum value:", amin)
print("Maximum value:", amax)

# import numpy as np
# import nibabel as nib
# # Load the .npz file
# img = nib.load("norm_av.nii.gz")
# data = img.get_fdata()
# amax = data.max()
# amin = data.min()

# # Print the results
# print(type(amin))
# print("Minimum value:", amin)
# print("Maximum value:", amax)
