import svt

import math
import time
import numpy as np
import random
import h5py

import util
import gc
import load_data
import asyncio
import concurrent
from functools import partial

import load_data
import coil_est
import cupy as cp

import solvers
import grad

import prox



async def main(lx=0.5, ls=0.0002):

	imsize = (320,320,320)

	load_from_zero = False
	if load_from_zero:
		start = time.time()
		dataset = await load_data.load_flow_data('/media/buntess/OtherSwifty/Data/COBRA191/MRI_Raw.h5', gating_names=['TIME_E0', 'ECG_E0'])
		end = time.time()
		print(f"Load Time={end - start} s")
		
		start = time.time()
		dataset = await load_data.gate_time(dataset)
		end = time.time()
		print(f"Gate Time={end - start} s")

		start = time.time()
		dataset['kdatas'] = coil_est.coil_compress(dataset['kdatas'], 0, 32)
		end = time.time()
		print(f"Compress Time={end - start} s")

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
		load_data.save_processed_dataset(dataset, '/media/buntess/OtherSwifty/Data/COBRA191/dataset.h5')
		end = time.time()
		print(f"Save Dataset Time={end - start} s")
	else:
		start = time.time()
		dataset = load_data.load_processed_dataset('/media/buntess/OtherSwifty/Data/COBRA191/dataset.h5')
		end = time.time()
		print(f"Load Dataset Time={end - start} s")

	# smaps = await coil_est.walsh(list([dataset['coords'][0]]), list([dataset['kdatas'][0]]), list([dataset['weights'][0]]), imsize)


	smaps, image = await coil_est.low_res_sensemap(dataset['coords'][0], dataset['kdatas'][0], dataset['weights'][0], imsize,
									  tukey_param=(0.95, 0.95, 0.95), exponent=3)


	devicectx = grad.DeviceCtx(cp.cuda.Device(0), 2, imsize, "full")

	do_isense = True
	if do_isense:
		smaps, image, alpha_i = await coil_est.isense(image, smaps, 
			cp.array(dataset['coords'][0]), 
			cp.array(dataset['kdatas'][0]), 
			cp.array(dataset['weights'][0]),
			devicectx, 
			iter=[5,[4,7]],
			lamda=[lx, ls])

	image = np.repeat(image, 5, axis=0)

	async def gradx(ximg, a):
		await grad.gradient_step_x(smaps, ximg, dataset['coords'], dataset['kdatas'], dataset['weights'],
				a, [devicectx], calcnorm=False)
		
	proxx = prox.dctprox(lx)

	await solvers.fista(np, image, alpha_i, gradx, proxx, 15)

	filename = f'/media/buntess/OtherSwifty/Data/COBRA191/reconed_lx{lx}_ls{ls}.h5'
	print('Save')
	with h5py.File(filename, 'w') as f:
		f['image'] = image
		f['smaps'] = smaps

if __name__ == "__main__":
	for i in range(100):
		print(f'Iteration number: {i}')
		lambda_x = 10**(random.uniform(0, -4))
		lambda_s = 10**(random.uniform(-2, -6))

		asyncio.run(main(lambda_x, lambda_s))


