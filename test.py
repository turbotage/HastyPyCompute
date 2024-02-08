import h5py
import plot_utility as pu
import numpy as np



with h5py.File('/media/buntess/OtherSwifty/Data/Garpen/Ena/SenseMaps.h5', 'r') as f:

    smaps = f['Maps']
    smap3 = np.array(smaps['SenseMaps_3'][:])

    print(1)