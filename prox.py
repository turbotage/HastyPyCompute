import numpy as np
import cupy as cp
import cupyx
import cupyx.scipy as cpxsp

import svt

def dctprox(base_alpha):

    async def dctprox_ret(image, alpha, scratchmem):

        lamda = base_alpha * alpha

        cp.fuse(kernel_name='softmax')
        def softmax(img, lam):
            return cp.exp(1j*cp.angle(img)) * cp.maximum(0, (cp.abs(img) - lam))

        for i in range(image.shape[0]):
            gpuimg = cpxsp.fft.dctn(cp.array(image[i,...]))
            gpuimg = softmax(gpuimg, lamda)
            gpuimg = cpxsp.fft.idctn(gpuimg)
            cp.copyto(image[i,...], gpuimg)

    return dctprox_ret

def fftprox(base_alpha):

    async def fftprox_ret(image, alpha, scratchmem):

        lamda = base_alpha * alpha

        def softmax(img, lam):
            return cp.exp(1j*cp.angle(img)) * cp.maximum(0, (cp.abs(img) - lam))
        
        for i in range(image.shape[0]):
            gpuimg = cp.fft.fftn(cp.array(image[i,...]), norm="ortho")
            gpuimg = softmax(gpuimg, lamda)
            gpuimg = cp.fft.ifftn(gpuimg, norm="ortho")
            cp.copyto(image[i,...], gpuimg)

    return fftprox_ret


def svtprox(base_alpha, blk_shape, blk_strides, block_iter):

    async def svtprox_ret(image, alpha, scratchmem):

        lamda = base_alpha * alpha

        scratchmem.fill(0.0)
        await svt.my_svt3(scratchmem, image,  lamda, blk_shape, blk_strides, block_iter, 5)

        np.copyto(image, scratchmem)

    return svtprox_ret


def spatial_svtprox(base_alpha, blk_shape, blk_strides, block_iter):

    async def spatial_svtprox_ret(image, alpha, scratchmem):

        lamda = base_alpha * alpha

        scratchmem.fill(0.0)
        await svt.my_spatial_svt3(scratchmem, image, lamda, blk_shape, blk_strides, block_iter)

        np.copyto(image, scratchmem)

    return spatial_svtprox_ret
