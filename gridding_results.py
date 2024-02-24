import cufinufft
import numpy as np
import cupy as cp
import simulate_mri as simri
import h5py
from sigpy import dcf as dcf
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from matplotlib import colormaps

import orthoslicer as ort
import plot_utility as pu

from direct_sinc import cuda_direct_sinc3_weights

def color_weights(coords, weights, frame, enc):
    size = 100
    wimg = np.zeros((size,size,size), dtype=np.float32)
    crd = size * (np.pi + coords[frame][enc]) / (2 * np.pi)
    w = weights[frame][enc]
    for i in range(crd.shape[1]):
        wimg[int(crd[0,i]), int(crd[1,i]), int(crd[2,i])] += w[i]
    return wimg
    

base_path = '/home/turbotage/Documents/4DRecon/'

coords, kdatas, nframes, nenc, kdatamax = simri.load_coords_kdatas(base_path)
true_image, vessel_mask, smaps = simri.load_true_and_smaps(base_path)




weights = []
for frame in range(nframes):
    wl = []
    w = cuda_direct_sinc3_weights(coords[frame][0])
    for enc in range(nenc):
        wl.append(w)
    print('Weights: ', frame)
    weights.append(wl)

grid_image = np.empty_like(true_image)
imgout = cp.empty(smaps.shape, dtype=np.complex64)
for frame in range(nframes):
    for enc in range(nenc):
    
        kd = cp.array(kdatas[frame][enc] * weights[frame][enc])
        crd = cp.array(coords[frame][enc])

        cufinufft.nufft3d1(x=crd[0], y=crd[1], z=crd[2], data=kd, n_modes=imgout.shape, out=imgout)
        grid_image[frame, enc,...] = cp.sum(smaps.conj() * imgout.get(), axis=0) / cp.sum(smaps.conj() * smaps, axis=0)

ort.image_nd(grid_image)


viewshare_coords = []
viewshare_kdatas = []
viewshare_weights = []

for frame in range(nframes):
    
    crd = np.concatenate([
        coords[(frame - 1) % nframes][enc], 
        coords[(frame) % nframes][enc],
        coords[(frame + 1) % nframes][enc]
        ], axis=1)
    
    wd = cuda_direct_sinc3_weights(crd)

    crd_list = []
    wd_list = []
    kd_list = []

    for enc in range(nenc):
        
        crd_list.append(crd)
        wd_list.append(wd)

        kd = np.concatenate([
            kdatas[(frame - 1) % nframes][enc], 
            kdatas[(frame) % nframes][enc],
            kdatas[(frame + 1) % nframes][enc]
            ], axis=1)
        
        kd_list.append(kd)

    viewshare_coords.append(crd_list)
    viewshare_weights.append(wd_list)
    viewshare_kdatas.append(kd_list)

del crd, wd, kd, crd_list, wd_list, kd_list

imgout = cp.empty(smaps.shape, dtype=np.complex64)
for frame in range(nframes):
    for enc in range(nenc):
    
        kd = cp.array(viewshare_kdatas[frame][enc] * viewshare_weights[frame][enc])
        crd = cp.array(viewshare_coords[frame][enc])

        cufinufft.nufft3d1(x=crd[0], y=crd[1], z=crd[2], data=kd, n_modes=imgout.shape, out=imgout)
        grid_image[frame, enc,...] = cp.sum(smaps.conj() * imgout.get(), axis=0) / cp.sum(smaps.conj() * smaps, axis=0)

ort.image_nd(grid_image)


