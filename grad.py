import numpy as np
import cupy as cp
import cupyx as cpx
import cupyx.scipy as cpxsp
import cufinufft
import math

import asyncio
import concurrent

# Inputs shall be on CPU
def gradient_step_x(smaps, image, coords, kdata, weights, device, alpha, full=False):
    with device:
        num_smaps = smaps.shape[0]
        num_slices = image.shape[0]

        smaps_gpu = cp.array(smaps)

        imshape = image.shape[1:]

        streams = [cp.cuda.Stream(non_blocking=True), cp.cuda.Stream(non_blocking=True)]
        forward_plans = []
        backward_plans = []

        if full:
            forward_plans.append(cufinufft.Plan(nufft_type=2, n_modes=imshape, 
                n_trans=num_smaps, eps=1e-4, dtype="complex64", upsampfac=2.0,
                gpu_method=1, gpu_sort=1, gpu_kerevalmeth=0,
                gpu_device_id=device.id, gpu_stream=streams[0].ptr))
            
            forward_plans.append(cufinufft.Plan(nufft_type=2, n_modes=imshape, 
                n_trans=num_smaps, eps=1e-4, dtype="complex64", upsampfac=2.0,
                gpu_method=1, gpu_sort=1, gpu_kerevalmeth=0,
                gpu_device_id=device.id, gpu_stream=streams[1].ptr))

            backward_plans.append(cufinufft.Plan(nufft_type=1, n_modes=imshape,
                n_trans=num_smaps, eps=1e-4, dtype="complex64", upsampfac=2.0,
                gpu_method=2,
                gpu_device_id=device.id, gpu_stream=streams[0].ptr))
            
            backward_plans.append(cufinufft.Plan(nufft_type=1, n_modes=imshape,
                n_trans=num_smaps, eps=1e-4, dtype="complex64", upsampfac=2.0,
                gpu_method=2,
                gpu_device_id=device.id, gpu_stream=streams[1].ptr))
        else:
            forward_plans.append(cufinufft.Plan(nufft_type=2, n_modes=imshape, 
                n_trans=num_smaps, eps=1e-4, dtype="complex64", upsampfac=1.25,
                gpu_method=1, gpu_sort=1, gpu_kerevalmeth=0,
                gpu_device_id=device.id, gpu_stream=streams[0].ptr))
            
            forward_plans.append(cufinufft.Plan(nufft_type=2, n_modes=imshape, 
                n_trans=num_smaps, eps=1e-4, dtype="complex64", upsampfac=1.25,
                gpu_method=1, gpu_sort=1, gpu_kerevalmeth=0,
                gpu_device_id=device.id, gpu_stream=streams[1].ptr))
            
            backward_plans.append(cufinufft.Plan(nufft_type=1, n_modes=imshape,
                n_trans=num_smaps, eps=1e-4, dtype="complex64", upsampfac=2.0,
                gpu_method=2,
                gpu_device_id=device.id, gpu_stream=streams[0].ptr))
            
            backward_plans.append(cufinufft.Plan(nufft_type=1, n_modes=imshape,
                n_trans=num_smaps, eps=1e-4, dtype="complex64", upsampfac=2.0,
                gpu_method=2,
                gpu_device_id=device.id, gpu_stream=streams[1].ptr))

        imagemem = cp.zeros_like(image[0,...])

        cp.fuse(kernel_name='weights_and_kdata_func')
        def weights_and_kdata_func(kdmem, kd, w):
            return w*(kdmem - kd)
        
        cp.fuse(kernel_name='sum_smaps_func')
        def sum_smaps_func(imgmem, s, alpha):
            return alpha * cp.sum(imgmem * cp.conj(s))

        loop = asyncio.get_event_loop()
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)

        def execute_one_gradient(i):
            modi = i % 2
            with streams[modi]:
                image_frame = cp.array(image[i,...])
                kdata_frame = cp.array(kdata[i])
                weights_frame = cp.array(weights[i])
                coord_frame = cp.array(coords[i])

                kdatamem = cp.empty_like(kdata_frame)

                imagemem = smaps_gpu * image_frame

                forward_plans[modi].setpts(x=coord_frame[0,...], y=coord_frame[1,...], z=coord_frame[2,...])
                backward_plans[modi].setpts(x=coord_frame[0,...], y=coord_frame[1,...], z=coord_frame[2,...])

                forward_plans[modi].execute(imagemem, out=kdatamem)

                if kdata is not None:
                    kdatamem = weights_and_kdata_func(kdatamem, kdata_frame, weights_frame)
                    backward_plans[modi].execute(kdatamem, imagemem)
                else:
                    backward_plans[modi].execute(kdata * weights, imagemem)

                image[i,...] -= sum_smaps_func(imagemem, smaps_gpu, alpha).get()

        async def run():
            for i in range(0,num_slices,2):
                runner1 = loop.run_in_executor(executor, execute_one_gradient, i)
                if i + 1 < num_slices:
                    runner2 = loop.run_in_executor(executor, execute_one_gradient, i + 1)

                await runner1
                if i + 1 < num_slices:
                    await runner2

        asyncio.run(run())


# Inputs shall be on GPU
def gradient_step_s(smaps, image, coords, kdata, weights, device, alpha, full=True):
    with device:
        num_smaps = smaps.shape[0]
        num_slices = image.shape[0]

        smaps_gpu = cp.array(smaps)

        normfactor = 1.0 / np.sqrt(np.prod(smaps.shape[1:]))

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

        forward_plan.setpts(x=coords[0,:], y=coords[0,:], z=coords[0,:])
        backward_plan.setpts(x=coords[0,:], y=coords[0,:], z=coords[0,:])

        kdataout = cp.empty_like(kdata)
        smapsout = cp.empty_like(smaps)

        forward_plan.execute(image * smaps, out=kdataout)
        kdataout *= normfactor
        kdataout -= kdata
        kdataout *= weights
        backward_plan.execute(kdataout, out=smapsout)
        smapsout *= cp.conj(image) * normfactor

        smaps[:] -= alpha * smapsout
