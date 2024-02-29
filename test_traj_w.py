import h5py

import numpy as np
import plot_utility as pu

import matplotlib.pyplot as plt

base_path = '/home/turbotage/Documents/4DRecon/other_data/'

with h5py.File(base_path + 'MRI_Raw.h5', 'r') as f:

    fk = f['Kdata']

    shp = fk['KX_E1'][:].shape

    coord = np.stack([
        fk['KX_E1'][:].reshape(shp[-1]*shp[-2]), 
        fk['KY_E1'][:].reshape(shp[-1]*shp[-2]), 
        fk['KZ_E1'][:].reshape(shp[-1]*shp[-2])], 
        axis=0)

    weights = fk['KW_E1'][:].reshape(shp[-1]*shp[-2])

    rr = np.sum(np.square(coord), axis=0)
    wrr = rr * np.exp(-rr / (0.25*(160*160)))

    rrs = np.sqrt(rr)

    aidx = np.argsort(rr)

    rrs = rrs[aidx]
    weights = weights[aidx]
    wrr = wrr[aidx]
    rr = rr[aidx]

    print('HG')

plt.figure()
plt.plot(rrs, weights)
plt.plot(rrs, wrr)
plt.plot(rrs, rr)
plt.show()