import numpy as np
import finufft
import numba as nb
from numba import cuda
import math

@cuda.jit
def numba_cuda_stupid_distance(coord, weights):
    x = coord[0,...]
    y = coord[1,...]
    z = coord[2,...]
    pos = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x

    if pos < x.size:
        for j in range(x.size):
            if pos == j:
                continue

            wx = (x[j] - x[pos])

            wy = (y[j] - y[pos])


            wz = (z[j] - z[pos])

            w = wx*wx + wy*wy + wz*wz

            weights[pos] += 1 / w

        weights[pos] = (1.0 / weights[pos])
      

@cuda.jit
def numba_cuda_direct_sinc3_weights(coord, weights):
    x = coord[0,...]
    y = coord[1,...]
    z = coord[2,...]
    pos = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x

    k = 16 #2*np.pi
    if pos < x.size:
        for j in range(x.size):
            wx = k * (x[j] - x[pos])
            if abs(wx) < 1e-6:
                wx = 1
            else:
                wx = math.sin(wx) / (wx)

            wy = k * (y[j] - y[pos])
            if abs(wy) < 1e-6:
                wy = 1
            else:
                wy = math.sin(wy) / (wy)

            wz = k * (z[j] - z[pos])
            if abs(wz) < 1e-6:
                wz = 1
            else:
                wz = math.sin(wz) / (wz)

            w = wx * wy * wz
            w *= w

            weights[pos] += w

        weights[pos] = (1.0 / weights[pos])
        
def cuda_direct_sinc3_weights(coord):

    weights = np.zeros(coord.shape[1], dtype=np.float32)

    coord_cu = cuda.to_device(coord)
    weights_cu = cuda.to_device(weights)

    threadsperblock = 32
    blockspergrid = (coord.shape[1] + (threadsperblock - 1)) // threadsperblock
    numba_cuda_direct_sinc3_weights[blockspergrid, threadsperblock](coord_cu, weights_cu)
    return weights_cu.copy_to_host()


def cuda_stupid_distance(coord):
    weights = np.zeros(coord.shape[1], dtype=np.float32)

    coord_cu = cuda.to_device(coord)
    weights_cu = cuda.to_device(weights)

    threadsperblock = 32
    blockspergrid = (coord.shape[1] + (threadsperblock - 1)) // threadsperblock
    numba_cuda_stupid_distance[blockspergrid, threadsperblock](coord_cu, weights_cu)
    return weights_cu.copy_to_host()