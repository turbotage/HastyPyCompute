import svt

import math
import time
import numpy as np

import util
import load_data
import asyncio
import concurrent
from functools import partial

import load_data
import coil_est
import cupy as cp

import solvers
import grad




async def main():

	imsize = (160,160,160)

	load_from_zero = False
	if load_from_zero:
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

		maxval = max(np.max(np.abs(wd)) for wd in dataset['weights'])
		for wd in dataset['weights']:
			wd[:] /= maxval

		start = time.time()
		load_data.save_processed_dataset(dataset, '/home/turbotage/Documents/4DRecon/dataset.h5')
		end = time.time()
		print(f"Save Dataset Time={end - start} s")
	else:
		start = time.time()
		dataset = load_data.load_processed_dataset('/home/turbotage/Documents/4DRecon/dataset.h5')
		end = time.time()
		print(f"Load Dataset Time={end - start} s")


	smaps, image = await coil_est.low_res_sensemap(dataset['coords'][0], dataset['kdatas'][0], dataset['weights'][0], imsize,
									  tukey_param=(0.95, 0.95, 0.95), exponent=3)

	
	devicectx = grad.DeviceCtx(cp.cuda.Device(0), 11, imsize, "full")

	smaps, image = await coil_est.isense(image, smaps, 
		cp.array(dataset['coords'][0]), 
		cp.array(dataset['kdatas'][0]), 
		cp.array(dataset['weights'][0]),
		devicectx, 
		iter=[7,[5,5]],
		lamda=[0.5, 0.00001])


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


