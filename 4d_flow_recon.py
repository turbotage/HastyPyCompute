import numpy as np
import cupy as cp
import cupyx as cpx
import cupyx.scipy as cpxsp
import cufinufft

import svt

import math
import time

import util
import load_data
import asyncio


def dctprox(base_alpha):

    def dctprox_ret(image, alpha, scratchmem):

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
        

def gradient_step(smaps, image, coords, kdata, weights, device, alpha, full=False):
    with device:
        num_smaps = smaps.shape[0]
        num_slices = image.shape[0]

        smaps_gpu = cp.array(smaps)

        imshape = image.shape[1:]

        if full:
            forward_plan = cufinufft.Plan(nufft_type=2, n_modes=imshape, 
                n_trans=num_smaps, eps=1e-4, dtype="complex64", upsampfac=2.0,
                gpu_method=1, gpu_sort=1, gpu_kerevalmeth=0)
            
            backward_plan = cufinufft.Plan(nufft_type=1, n_modes=imshape,
                n_trans=num_smaps, eps=1e-4, dtype="complex64", upsampfac=2.0,
                gpu_method=2)
        else:
            forward_plan = cufinufft.Plan(nufft_type=2, n_modes=imshape, 
                n_trans=num_smaps, eps=1e-4, dtype="complex64", upsampfac=1.25,
                gpu_method=1, gpu_sort=1, gpu_kerevalmeth=0)
            
            backward_plan = cufinufft.Plan(nufft_type=1, n_modes=imshape,
                n_trans=num_smaps, eps=1e-4, dtype="complex64", upsampfac=2.0,
                gpu_method=2)

        imagemem = cp.zeros_like(image[0,...])

        cp.fuse(kernel_name='weights_and_kdata_func')
        def weights_and_kdata_func(kdmem, kd, w):
            return w*(kdmem - kd)
        
        cp.fuse(kernel_name='sum_smaps_func')
        def sum_smaps_func(imgmem, s, alpha):
            return alpha * cp.sum(imgmem * cp.conj(s))

        #stupid_frame = cp.zeros_like(image[0,...])

        for i in range(num_slices):
            image_frame = cp.array(image[i,...])
            kdata_frame = cp.array(kdata[i])
            weights_frame = cp.array(weights[i])
            coord_frame = cp.array(coords[i])

            kdatamem = cp.empty_like(kdata_frame)

            imagemem = smaps_gpu * image_frame

            forward_plan.setpts(x=coord_frame[0,...], y=coord_frame[1,...], z=coord_frame[2,...])
            backward_plan.setpts(x=coord_frame[0,...], y=coord_frame[1,...], z=coord_frame[2,...])

            forward_plan.execute(imagemem, out=kdatamem)

            if kdata is not None:
                kdatamem = weights_and_kdata_func(kdatamem, kdata_frame, weights_frame)
                backward_plan.execute(kdatamem, imagemem)
            else:
                backward_plan.execute(kdata * weights, imagemem)

            image[i,...] -= sum_smaps_func(imagemem, smaps_gpu, alpha).get()

            
def fista(smaps, image, coords, kdata, weights, numiter, alpha, gradstep, prox):
    
    if image.is_pinned != True:
        raise RuntimeError('image must be in pinned memory')
    if kdata.is_pinned != True:
        raise RuntimeError('kdata must be in pinned memory')
    if smaps.device == cp.cuda.current_device():
        raise RuntimeError('smaps must be on device')

    t = 1

    resids = []

    image_old = cp.empty_like(image)
    image_z = cp.empty_like(image)
    cp.copyto(image_z, image)

    def update():
        cp.copyto(image_old, image)
        cp.copyto(image, image_z)

        gradstep(smaps, image, coords, kdata, weights, cp.cuda.Device(0), alpha)

        prox(image, alpha, image_z)

        t_old = t
        t = 0.5 * (1.0 + math.sqrt(1.0 + 4.0*t_old*t_old)) 

        cp.subtract(image, image_old, out=image_z)
        resids.append(np.linalg.norm(image_z))
        cp.add(image, ((t_old - 1.0) / t) * image_z, out=image_z)

    for i in range(numiter):
        update()
    





nx = 160
ny = 160
nz = 160
nframe = 80*5
ncoil = 32
nupts = 110000

image = util.complex_rand((nframe, nx, ny, nz))
smaps = util.complex_rand((ncoil, nx, ny, nz))
kdata = util.rand_vector((ncoil, nupts), nframe, dtype=np.complex64)
coords = util.rand_vector((3,nupts), nframe, [-3.1415, 3.1415], np.float32)
weights = util.rand_vector((nupts,), nframe)




if True:
    start = time.time()

    output = np.empty_like(image)
    asyncio.run(svt.my_svt3(output, image, 0.1, np.array([16,16,16]), np.array([16,16,16]), 4, 5))
    #svt.svt_numba3(output, image, 0.1, np.array([16,16,16]), np.array([16,16,16]), 4, 5)

    end = time.time()

    print(f"Time: {end - start}")


if False:
    start = time.time()

    gradient_step(smaps, image, coords, kdata, weights, cp.cuda.Device(0), 0.1)
    cp.cuda.stream.get_current_stream().synchronize()

    end = time.time()

    print(f"Time: {end - start}")


print('Hello')


