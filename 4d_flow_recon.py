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



async def get_smaps(lx=0.5, ls=0.0002, im_size=(320,320,320), load_from_zero = False):

	if load_from_zero:
		start = time.time()
		dataset = await load_data.load_flow_data('/media/buntess/OtherSwifty/Data/Garpen/Ena/MRI_Raw.h5', gating_names=['TIME_E0', 'ECG_E0'])
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
		load_data.save_processed_dataset(dataset, '/media/buntess/OtherSwifty/Data/Garpen/Ena/dataset.h5')
		end = time.time()
		print(f"Save Dataset Time={end - start} s")
	else:
		start = time.time()
		dataset = load_data.load_processed_dataset('/media/buntess/OtherSwifty/Data/Garpen/Ena/dataset.h5')
		end = time.time()
		print(f"Load Dataset Time={end - start} s")

	# smaps = await coil_est.walsh(list([dataset['coords'][0]]), list([dataset['kdatas'][0]]), list([dataset['weights'][0]]), imsize)

	#coil_images, ref_c = coil_est.create_coil_images(list([dataset['coords'][0]]), list([dataset['kdatas'][0]]), list([dataset['weights'][0]]), imsize)
	
	#start = time.time()
	#smaps = coil_est.walsh_cpu(coil_images, ref_c, np.array([9,9,9]))
	#end = time.time()
	#print(f"Walsh Took={end - start}")
	
	#filename = f'/media/buntess/OtherSwifty/Data/COBRA191/reconed_lx{lx}_ls{ls}.h5'
	#print('Save')
	#with h5py.File(filename, 'w') as f:
	#	f['smaps'] = smaps
		
		
	

	smaps, image = await coil_est.low_res_sensemap(dataset['coords'][0], dataset['kdatas'][0], dataset['weights'][0], imsize,
									  tukey_param=(0.95, 0.95, 0.95), exponent=3)

	#dataset['weights'] = [w**0.75 for w in dataset['weights']]

	devicectx = grad.DeviceCtx(cp.cuda.Device(0), 2, imsize, "full")

	do_isense = False
	if do_isense:
		smaps, image, alpha_i, resids = await coil_est.isense(image, smaps, 
			cp.array(dataset['coords'][0]), 
			cp.array(dataset['kdatas'][0]), 
			cp.array(dataset['weights'][0]),
			devicectx, 
			iter=[5,[4,7]],
			lamda=[lx, ls])
	else:
		async def inormal(imgnew):
			return await grad.gradient_step_x(smaps, imgnew, [dataset['coords'][0]], None, [dataset['weights'][0]], None, [devicectx])

		alpha_i = 0.25 / await solvers.max_eig(np, inormal, util.complex_rand(image[0,...][None,...].shape, xp=np), 8)


	image = np.repeat(image, 5, axis=0)

	async def gradx(ximg, a):
		await grad.gradient_step_x(smaps, ximg, dataset['coords'], dataset['kdatas'], dataset['weights'],
				a, [devicectx], calcnorm=False)
		
	proxx = prox.dctprox(lx)

	#proxx = prox.svtprox()

	await solvers.fista(np, image, alpha_i, gradx, proxx, 15)

	# del ....
	#cp.get_default_memory_pool().free_all_blocks()

	#filename = f'/media/buntess/OtherSwifty/Data/COBRA191/reconed_lx{lx:.5f}_ls{ls:.7f}_res{resids[-1]}.h5'
	filename = '/media/buntess/OtherSwifty/Data/Garpen/Ena/reconed_iSENSE.h5'
	print('Save')
	with h5py.File(filename, 'w') as f:
		f.create_dataset('image', data=image)
		f.create_dataset('smaps', data=smaps)

	del image, smaps, dataset


async def run_framed(niter, nframes, smapsPath, load_from_zero=True, imsize = (320,320,320)):
	

	if load_from_zero:
		start = time.time()
		dataset = await load_data.load_flow_data('/media/buntess/OtherSwifty/Data/Garpen/Ena/MRI_Raw.h5', gating_names=['TIME_E0', 'ECG_E0'])
		end = time.time()
		print(f"Load Time={end - start} s")

		start = time.time()
		dataset['kdatas'] = coil_est.coil_compress(dataset['kdatas'], 0, 32)
		end = time.time()
		print(f"Compress Time={end - start} s")
		
		start = time.time()
		dataset = await load_data.gate_ecg(dataset, nframes)
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
		load_data.save_processed_dataset(dataset, '/media/buntess/OtherSwifty/Data/Garpen/Ena/dataset_framed.h5')
		end = time.time()
		print(f"Save Dataset Time={end - start} s")
	else:
		start = time.time()
		dataset = load_data.load_processed_dataset('/media/buntess/OtherSwifty/Data/Garpen/Ena/dataset_framed.h5')
		end = time.time()
		print(f"Load Dataset Time={end - start} s")



	# Load smaps and full image
	smaps, image = load_data.load_smaps_image(smapsPath)

	#dataset['weights'] = [w**0.75 for w in dataset['weights']]


	devicectx = grad.DeviceCtx(cp.cuda.Device(0), 2, imsize, "full")
	image = np.repeat(image, nframes, axis=0)

	async def inormal(imgnew):
		return await grad.gradient_step_x(smaps, imgnew, [dataset['coords'][0]], None, [dataset['weights'][0]], None, [devicectx])

	alpha_i = 0.25 / await solvers.max_eig(np, inormal, util.complex_rand(image[0,...][None,...].shape, xp=np), 8)

	async def gradx(ximg, a):
		await grad.gradient_step_x(smaps, ximg, dataset['coords'], dataset['kdatas'], dataset['weights'],
				a, [devicectx], calcnorm=False)
		

	proxx = prox.svtprox(base_alpha=1e-3, blk_shape=np.array([8, 8, 8]), blk_strides=np.array([8, 8, 8]), block_iter=2)

	await solvers.fista(np, image, alpha_i, gradx, proxx, niter)

	# del ....
	#cp.get_default_memory_pool().free_all_blocks()

	filename = f'/media/buntess/OtherSwifty/Data/Garpen/Ena/reconed_framed{nframes}.h5'
	print('Save')
	with h5py.File(filename, 'w') as f:
		f.create_dataset('image', data=image)

	del image, smaps, dataset


async def test_svt(smapsPath, nframes=10):
	# Load smaps and full image
	smaps, image = load_data.load_smaps_image(smapsPath)

	#dataset['weights'] = [np.sqrt(w) for w in dataset['weights']]


	devicectx = grad.DeviceCtx(cp.cuda.Device(0), 2, imsize, "full")
	image = np.repeat(image, nframes, axis=0)

	proxx = prox.svtprox(base_alpha=1e-5, blk_shape=np.array([16, 16, 16]), blk_strides=np.array([16, 16, 16]), block_iter=2)

	import plot_utility as pu
	image_c = image.copy()
	await proxx(image_c, 1e-9, image_c.copy())

	print('Hej')


if __name__ == "__main__":
	imsize = (256,256,256)

	for i in range(1):
		print(f'Iteration number: {i}')
		lambda_x = round(10**(random.uniform(0, -4)), 5)
		lambda_s = round(10**(random.uniform(-2, -6)), 7)

		#asyncio.run(get_smaps(lambda_x, lambda_s, imsize, False))
		
		#cp.get_default_memory_pool().free_all_blocks()

		sPath = '/media/buntess/OtherSwifty/Data/Garpen/Ena/reconed_iSENSE.h5' #'/media/buntess/OtherSwifty/Data/COBRA191/reconed_lowres.h5'

		asyncio.run(run_framed(niter=10, nframes=20, smapsPath=sPath, load_from_zero=False, imsize=imsize))
