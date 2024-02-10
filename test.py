import h5py
import plot_utility as pu
import numpy as np


from sigpy import dcf as dcf
import load_data

print(1)

im_size=(256,256,256)

dataset = load_data.load_processed_dataset('/media/buntess/OtherSwifty/Data/Garpen/Ena/dataset_framed.h5')

for i in range(len(dataset['weights'])):
    print(f'Frame {i}')
    dataset['weights'][i] = dcf.pipe_menon_dcf(dataset['coords'][i]/np.pi*im_size[0]/2, im_size)

print(1)