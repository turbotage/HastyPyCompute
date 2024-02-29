import h5py
import plot_utility as pu
import numpy as np


from sigpy import dcf as dcf
import load_data

import orthoslicer as ort
import image_creation as ic

#with h5py.File('/home/turbotage/Documents/4DRecon/results/reconed_framed_w0.5000_l0.0010_i149.h5', 'r') as hf:
#    image_found = np.array(hf['image'])

#with h5py.File('/home/turbotage/Documents/4DRecon/background_corrected_cropped.h5', 'r') as hf:
#    image = np.array(hf['img'])

nspokes = 400
nsamp_per_spoke = 85
im_size = (128, 128, 128)
crop_factor = 1.0
method = 'PCVIPR'

xangles = np.random.rand(nspokes).astype(np.float32)
xangles = np.arccos(1.0 - 2.0*xangles).astype(np.float32)
zangles = 2 * np.pi * np.random.rand(nspokes).astype(np.float32)

coord = np.ascontiguousarray(ic.create_coords(nspokes, nsamp_per_spoke, 
    im_size, method, False, crop_factor, xangles, zangles))

pu.scatter_3d(coord, title='PC-VIPR Sampling Pattern', axis_labels=None)


#with h5py.File('/home/turbotage/Documents/4DRecon/background_corrected.h5', 'r') as hf:
#    image_true = np.array(hf['corrected_img'])
#    image_cd = np.array(hf['cd'])
#    image_vel = np.array(hf['vel'])

print(1)

