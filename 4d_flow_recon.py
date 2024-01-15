import cupy as cp
import cufinufft


def gradient_step(smaps, image, coords, kspace, weights):
    num_smaps = smaps.shape[0]
    num_frames = image.shape[0]
    num_encodes = image.shape[1]

    for i in range(num_frames):
        for j in range(num_encodes):
            
