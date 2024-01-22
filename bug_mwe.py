import numpy as np
import cupy as cp
import cupyx as cpx
import cupyx.scipy as cpxsp
import cufinufft
import math

import asyncio
import concurrent
from functools import partial
import time

# Inputs shall be on CPU
async def gradient_step_x(smaps, image, coords, kdata, weights, device, alpha, ntransf = None, streams=[],full=False):
    with device:
        
        if len(streams) == 0:
            streams = [cp.cuda.get_current_stream()]

        num_smaps = smaps.shape[0]
        num_slices = image.shape[0]

        if ntransf is None:
            ntransf = num_smaps
        if num_smaps % ntransf != 0:
            raise ValueError(f"Number of smaps ({num_smaps}) must be divisible by ntransf ({ntransf})")

        smaps_gpu = cp.array(smaps)

        imshape = image.shape[1:]

        forward_plans = []
        backward_plans = []

        if full:
            for stream in streams:
                forward_plans.append(cufinufft.Plan(nufft_type=2, n_modes=imshape, 
                    n_trans=ntransf, eps=1e-4, dtype="complex64", upsampfac=2.0,
                    gpu_method=1, gpu_sort=1, gpu_kerevalmeth=0,
                    gpu_device_id=device.id, gpu_stream=stream.ptr))
                
                backward_plans.append(cufinufft.Plan(nufft_type=1, n_modes=imshape,
                    n_trans=ntransf, eps=1e-4, dtype="complex64", upsampfac=2.0,
                    gpu_method=2,
                    gpu_device_id=device.id, gpu_stream=stream.ptr))
        else:
            for stream in streams:
                forward_plans.append(cufinufft.Plan(nufft_type=2, n_modes=imshape, 
                    n_trans=ntransf, eps=1e-4, dtype="complex64", upsampfac=1.25,
                    gpu_method=1, gpu_sort=1, gpu_kerevalmeth=0,
                    gpu_device_id=device.id, gpu_stream=stream.ptr))
                
                backward_plans.append(cufinufft.Plan(nufft_type=1, n_modes=imshape,
                    n_trans=ntransf, eps=1e-4, dtype="complex64", upsampfac=2.0,
                    gpu_method=2,
                    gpu_device_id=device.id, gpu_stream=stream.ptr))

        cp.fuse(kernel_name='weights_and_kdata_func')
        def weights_and_kdata_func(kdmem, kd, w):
            return w*(kdmem - kd)
        
        cp.fuse(kernel_name='sum_smaps_func')
        def sum_smaps_func(imgmem, s, alpha):
            return alpha * cp.sum(imgmem * cp.conj(s), axis=0)

        loop = asyncio.get_event_loop()
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=len(streams))

        def execute_one_gradient(slice):
            modi = slice % len(streams)
            with streams[modi]:
                stream = streams[modi]
                stream.synchronize()

                image_frame = cp.array(image[slice,...])
                weights_frame = cp.array(weights[slice])
                coord_frame = cp.array(coords[slice])
                
                forward_plans[modi].setpts(x=coord_frame[0,...], y=coord_frame[1,...], z=coord_frame[2,...])
                backward_plans[modi].setpts(x=coord_frame[0,...], y=coord_frame[1,...], z=coord_frame[2,...])

                runs = math.ceil(num_smaps / ntransf)
                for run in range(runs):
                    start = run * ntransf

                    kdata_frame = cp.array(kdata[slice][start:start+ntransf,...])

                    kdatamem = cp.empty_like(kdata_frame)

                    locals = smaps_gpu[start:start+ntransf,...]

                    imagemem = locals * image_frame

                    forward_plans[modi].execute(imagemem, out=kdatamem)
                    if kdata is not None:
                        kdatamem = weights_and_kdata_func(kdatamem, kdata_frame, weights_frame)
                        backward_plans[modi].execute(kdatamem, imagemem)
                    else:
                        backward_plans[modi].execute(kdatamem * weights, imagemem)
                    
                    image[slice,...] -= sum_smaps_func(imagemem, locals, alpha).get()


        for outer_slice in range(0,num_slices,len(streams)):
            futures = []

            for inner_slice in range(len(streams)):
                if outer_slice + inner_slice < num_slices:
                    futures.append(loop.run_in_executor(executor, execute_one_gradient, outer_slice + inner_slice))
                else:
                    break
                print(f"Slice = {outer_slice + inner_slice}")
            
            for fut in futures:
                await fut




def complex_rand(shape, dtype=np.float32):
    return np.random.rand(*shape).astype(dtype) + 1j*np.random.rand(*shape).astype(dtype)

def rand_vector(shape, num, bounds=[0.0, 1.0], dtype=np.float32):
    vec = []
    for i in range(num):
        if dtype == np.complex64:
            vec.append(bounds[0] + (bounds[1] - bounds[0])*complex_rand(shape, np.float32))
        elif dtype == np.complex128:
            vec.append(bounds[0] + (bounds[1] - bounds[0])*complex_rand(shape, np.float64))
        else:
            vec.append(bounds[0] + (bounds[1] - bounds[0])*np.random.rand(*shape).astype(dtype))
    return vec

async def main():
    
    ncoil = 16
    nframes = 5
    imsize = (80,80,80)
    nupts = 50000

    loop = asyncio.get_event_loop()
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=5)

    smaps = loop.run_in_executor(executor, partial(complex_rand, (ncoil,) + imsize))
    image = loop.run_in_executor(executor, partial(complex_rand, (nframes*5,) + imsize))
    kdata = loop.run_in_executor(executor, partial(rand_vector, (ncoil,nupts), nframes*5, dtype=np.complex64))
    coord = loop.run_in_executor(executor, partial(rand_vector, (3,nupts), nframes*5, dtype=np.float32))
    weights = loop.run_in_executor(executor, partial(rand_vector, (nupts,), nframes*5, dtype=np.float32))

    smaps = await smaps
    image = await image
    kdata = await kdata
    coord = await coord
    weights = await weights

    start = time.time()
    #await gradient_step_x(smaps, image, coord, kdata, weights, cp.cuda.Device(0), 0.1, 8, 
    #                    [cp.cuda.Stream(non_blocking=True), cp.cuda.Stream(non_blocking=True)])
    
    #await gradient_step_x(smaps, image, coord, kdata, weights, cp.cuda.Device(0), 0.1, 8, 
    #					[cp.cuda.Stream(non_blocking=True)])
    
    await gradient_step_x(smaps, image, coord, kdata, weights, cp.cuda.Device(0), 0.1, 8)
    
    end = time.time()
    print(f"Grad Time={end - start} s")
    print('Ran to Complete!')

if __name__ == "__main__":
    asyncio.run(main())