import h5py
import plot_utility as pu

with h5py.File('/home/turbotage/Documents/4DRecon/smaps_320.h5', 'r') as f:
    smaps = f['smaps'][()]

with h5py.File('/home/turbotage/Documents/4DRecon/image_320.h5', 'r') as f:
    images = f['image'][()]


print('Hello')