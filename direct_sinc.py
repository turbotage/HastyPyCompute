import numpy as np
import finufft
import numba as nb
from numba import cuda
import math

@nb.jit(nopython=True, cache=True, parallel=True, nogil=True)
def direct_sinc3_weights(coord):
    x = coord[0,...]
    y = coord[1,...]
    z = coord[2,...]

    weights = np.zeros(x.size, dtype=np.float32)
    for i in nb.prange(x.size):
        for j in range(x.size):
            weights[i] += np.square(
                np.sinc(x[j] - x[i]) * 
                np.sinc(y[j] - y[i]) * 
                np.sinc(z[j] - y[i]))
    return 1 / weights
        

@cuda.jit
def numba_cuda_direct_sinc3_weights(coord, weights):
    x = coord[0,...]
    y = coord[1,...]
    z = coord[2,...]
    pos = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    if pos < x.size:
        for j in range(x.size):
            wx = abs(x[j] - x[pos])
            wx = math.sin(wx + 1e-6) / (wx + 1e-6)

            wy = abs(y[j] - y[pos])
            wy = math.sin(wy + 1e-6) / (wy + 1e-6)

            wz = abs(z[j] - z[pos])
            wz = math.sin(wz + 1e-6) / (wz + 1e-6)

            w = wx * wy * wz
            w *= w

            weights[pos] += w

        weights[pos] = 1 / weights[pos]
        
def cuda_direct_sinc3_weights(coord):

    weights = np.zeros(coord.shape[1], dtype=np.float32)

    coord_cu = cuda.to_device(coord)
    weights_cu = cuda.to_device(weights)

    threadsperblock = 32
    blockspergrid = (coord.shape[1] + (threadsperblock - 1)) // threadsperblock
    numba_cuda_direct_sinc3_weights[blockspergrid, threadsperblock](coord_cu, weights_cu)
    return weights_cu.copy_to_host()