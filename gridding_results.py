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

from direct_sinc import cuda_direct_sinc3_weights, cuda_stupid_distance

def color_weights(coords, weights, frame, enc):
    size = 100
    wimg = np.zeros((size,size,size), dtype=np.float32)
    crd = size * (np.pi + coords[frame][enc]) / (2 * np.pi)
    w = weights[frame][enc]
    for i in range(crd.shape[1]):
        wimg[int(crd[0,i]), int(crd[1,i]), int(crd[2,i])] += w[i]
    return wimg

base_path = '/home/turbotage/Documents/4DRecon/run_nspoke200_samp100_noise2.00e-04/'

coords, kdatas, nframes, nenc, kdatamax = simri.load_coords_kdatas(base_path)
true_image, vessel_mask, smaps = simri.load_true_and_smaps(base_path)

gauss_factor = 4.0

plot_wc = True
if plot_wc:
    w_ds = 1e-5 + cuda_direct_sinc3_weights(coords[0][0])

    #w_sd = cuda_stupid_distance(coords[0][0])

    rr = np.sum(np.square(coords[0][0]), axis=0)
    w_r = (1e-4 + rr) * np.exp(-rr / ((gauss_factor * np.pi*np.pi)))

    w_pm = dcf.pipe_menon_dcf(256*coords[0][0]/np.pi, (512,512,512))

    #maxmax = np.max([np.max(w_r), np.max(w_ds), np.max(w_pm), np.max(w_sd)])

    #w_r /= np.max(w_r)
    #w_ds /= np.max(w_ds)
    #w_pm /= np.max(w_pm)
    #w_sd /= np.max(w_sd)

    #w_r /= maxmax
    #w_ds /= maxmax
    #w_pm /= maxmax
    #w_sd /= maxmax

    pu.scatter_3d(coords[0][0], color=colormaps['YlOrRd'](w_ds), axis_labels=None)
    pu.scatter_3d(coords[0][0], color=colormaps['YlOrRd'](w_r), axis_labels=None)
    pu.scatter_3d(coords[0][0], color=colormaps['YlOrRd'](w_pm), axis_labels=None)
    #pu.scatter_3d(coords[0][0], color=colormaps['YlOrRd'](w_sd), axis_labels=None)

    rrs = np.sqrt(rr)
    aidx = np.argsort(rrs)

    rrs = rrs[aidx]

    plt.figure()
    plt.plot(rrs, w_ds[aidx], label='Direct Sinc')
    plt.plot(rrs, w_r[aidx], label='Radial Gaussian')
    plt.plot(rrs, w_pm[aidx], label='Pipe Menon')
    #plt.plot(rrs, w_sd[aidx], label='Stupid Distance')
    plt.plot(rrs, rr[aidx], label='Radial Distance')
    plt.plot(rrs, np.sqrt(w_r[aidx]), label='Radial Distance Sqrt')
    plt.plot(rrs, np.sqrt(np.sqrt(w_r[aidx])), label='Radial Distance Sqrt')
    plt.legend()
    plt.show()


weights = []
for frame in range(nframes):
    wl = []
    #w = cuda_direct_sinc3_weights(coords[frame][0])
    #w = cuda_stupid_distance(coords[frame][0])
    #w = np.sum(np.square(coords[frame][0]), axis=0)

    #w = dcf.pipe_menon_dcf(160*coords[frame][0]/np.pi, (320,320,320))
    w = np.sum(np.square(coords[frame][0]), axis=0)
    w = (1e-5 + w) * np.exp(-w / ((gauss_factor * np.pi*np.pi)))
    w = w / w.max()

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

with h5py.File(base_path + 'gridded_image.h5', "w") as f:
    f.create_dataset('image', data=grid_image)


viewshare_coords = []
viewshare_kdatas = []
viewshare_weights = []

for frame in range(nframes):
    
    crd = np.concatenate([
        coords[(frame - 1) % nframes][enc], 
        coords[(frame) % nframes][enc],
        coords[(frame + 1) % nframes][enc]
        ], axis=1)
    
    #wd = 1e-5 + cuda_direct_sinc3_weights(crd)
    wd = np.sum(np.square(crd), axis=0)
    wd = (1e-5 + wd) * np.exp(-wd / ((gauss_factor*np.pi*np.pi)))
    wd = wd / wd.max()

    print('Weights: ', frame)

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

with h5py.File(base_path + 'viewshare_gridded_image.h5', "w") as f:
    f.create_dataset('image', data=grid_image)
