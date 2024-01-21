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


def svtprox(base_alpha, blk_shape, blk_strides, block_iter):

    def svtprox_ret(image, alpha, scratchmem):

        lamda = base_alpha * alpha

        scratchmem.fill(0.0)
        svt.svt_numba3(scratchmem, image,  lamda, blk_shape, blk_strides, block_iter, 5)

        np.copyto(image, scratchmem)

    return svtprox_ret
        
