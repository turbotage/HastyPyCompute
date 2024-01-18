import svt

import math
import time
import numpy as np

import util
import load_data
import asyncio

import load_data
import coil_est
import cupy as cp

import solvers


            
# def fista(smaps, image, coords, kdata, weights, numiter, alpha, gradstep, prox):
    
#     if image.is_pinned != True:
#         raise RuntimeError('image must be in pinned memory')
#     if kdata.is_pinned != True:
#         raise RuntimeError('kdata must be in pinned memory')
#     if smaps.device == cp.cuda.current_device():
#         raise RuntimeError('smaps must be on device')

#     t = 1

#     resids = []

#     image_old = cp.empty_like(image)
#     image_z = cp.empty_like(image)
#     cp.copyto(image_z, image)

#     def update():
#         cp.copyto(image_old, image)
#         cp.copyto(image, image_z)

#         gradstep(smaps, image, coords, kdata, weights, cp.cuda.Device(0), alpha)

#         prox(image, alpha, image_z)

#         t_old = t
#         t = 0.5 * (1.0 + math.sqrt(1.0 + 4.0*t_old*t_old)) 

#         cp.subtract(image, image_old, out=image_z)
#         resids.append(np.linalg.norm(image_z))
#         cp.add(image, ((t_old - 1.0) / t) * image_z, out=image_z)

#     for i in range(numiter):
#         update()
    









async def main():

    imsize = (160,160,160)

    start = time.time()
    dataset = await load_data.load_flow_data('/home/turbotage/Documents/4DRecon/MRI_Raw.h5', gating_names=['TIME_E0', 'ECG_E0'])
    end = time.time()
    print(f"Load Time={end - start} s")
    
    start = time.time()
    dataset = await load_data.gate_time(dataset)
    end = time.time()
    print(f"Gate Time={end - start} s")

    start = time.time()
    dataset = await load_data.flatten(dataset)
    end = time.time()
    print(f"Flatten Time={end - start} s")

    start = time.time()
    dataset = await load_data.crop_kspace(dataset, imsize)
    end = time.time()
    print(f"Crop Time={end - start} s")

    maxval = max(np.max(np.abs(kd)) for kd in dataset['kdatas'])
    for kd in dataset['kdatas']:
        kd[:] /= maxval

    smaps, image = coil_est.low_res_sensemap(dataset['coords'][0], dataset['kdatas'][0], dataset['weights'][0], imsize,
                                      tukey_param=(0.95, 0.95, 0.95), exponent=3)


    smaps, image = coil_est.isense(image, smaps, 
                        cp.array(dataset['coords'][0]), 
                        cp.array(dataset['kdatas'][0]), 
                        cp.array(dataset['weights'][0]))

    ncoil = 32
    imsize = (160,160,160)

    smaps = util.complex_rand((ncoil,) + imsize)
    image = util.complex_rand((80*5,) + imsize)
    kdata = util.complex_rand((ncoil,) + imsize)

    print('H')

    if False:
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


if __name__ == "__main__":
    asyncio.run(main())


