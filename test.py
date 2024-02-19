import h5py
import plot_utility as pu
import numpy as np


from sigpy import dcf as dcf
import load_data

import orthoslicer as ort

#with h5py.File('/home/turbotage/Documents/4DRecon/results/reconed_framed_w0.5000_l0.0010_i149.h5', 'r') as hf:
#    image_found = np.array(hf['image'])

#with h5py.File('/home/turbotage/Documents/4DRecon/reconed_full.h5', 'r') as hf:
#    image = np.array(hf['image'])

with h5py.File('/home/turbotage/Documents/4DRecon/background_corrected.h5', 'r') as hf:
    image_true = np.array(hf['corrected_img'])
    image_cd = np.array(hf['cd'])
    image_vel = np.array(hf['vel'])

print(1)

