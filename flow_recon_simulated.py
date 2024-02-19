import svt

import math
import time
import numpy as np
import random
import h5py
from sigpy import dcf as dcf


import direct_sinc
import util
import gc
import load_data
import asyncio
import concurrent
from functools import partial

import simulate_mri as simri

import orthoslicer as ort

import load_data
import coil_est
import cupy as cp
import cufinufft

import solvers
import grad

import prox

#base_path = '/media/buntess/OtherSwifty/Data/Garpen/Ena/'
base_path = '/home/turbotage/Documents/4DRecon/'


async def find_alpha(smaps, coords, weights, imsize):
	devicectxdict = {"dev": cp.cuda.Device(0), "ntransf": 16, "imsize": imsize[1:]}

	async def inormal(imgnew):
		return await grad.gradient_step_x(smaps, imgnew, coords, None, weights, None, [devicectxdict])

	alpha_i = 0.5 / await solvers.max_eig(np, inormal, util.complex_rand(imsize, xp=np), 10)

	return alpha_i

async def run_full(smaps, coords, kdatas):

	nenc = 5
	nframes = len(coords) // nenc

	imsize = (nenc,) + smaps.shape[1:]

	full_coords = []
	full_kdatas = []
	for i in range(nenc):
		coord_stack = []
		kdata_stack = []
		for j in range(nframes):
			coord_stack.append(coords[j*nenc + i])
			kdata_stack.append(kdatas[j*nenc + i])
		full_coords.append(np.concatenate(coord_stack, axis=1))
		full_kdatas.append(np.concatenate(kdata_stack, axis=1))
	
	
	weights = dcf.pipe_menon_dcf(160*full_coords[0]/np.pi, (320,320,320), max_iter=50)
	weights = weights / weights.max()
	#weights = direct_sinc.cuda_direct_sinc3_weights(coords[0])
	#weights = direct_sinc.cuda_direct_sinc3_weights(full_coords[0])
	weights = [weights for _ in range(nenc)]
		
	#weights = [(w / w.max())**0.5 for w in weights]

	img_check = True
	if img_check:
		img = np.zeros(imsize[1:], dtype=np.complex64)
		crd = [cp.array(full_coords[2][0,:]), cp.array(full_coords[2][1,:]), cp.array(full_coords[2][2,:])]
		#crd = [cp.array(coords[0][0,:]), cp.array(coords[0][1,:]), cp.array(coords[0][2,:])]
		imgout = cp.empty(img.shape, dtype=np.complex64)
		for i in range(32):
			kd = cp.array(full_kdatas[2][i,...] * weights[2])
			#kd = cp.array(kdatas[0][i,...] * weights[0])
			cufinufft.nufft3d1(x=crd[0], y=crd[1], z=crd[2], data=kd, n_modes=imgout.shape, out=imgout)
			img += smaps[i,...].conj() * imgout.get() 

		img /= np.sum(smaps.conj() * smaps, axis=0)

		print('Hello')


	weights = [w**0.5 for w in weights]
	alpha_i = await find_alpha(smaps, full_coords, weights, imsize)

	devicectxdict = {"dev": cp.cuda.Device(0), "ntransf": 16, "imsize": imsize[1:]}

	async def gradx(ximg, a):
		normlist = await grad.gradient_step_x(smaps, ximg, full_coords, full_kdatas, weights,
				a, [devicectxdict], calcnorm=True)
		
		print(f'Error = {np.array([a.item() for a in normlist[0]]).sum()}')

	image = np.zeros(imsize, dtype=np.complex64)

	proxx = prox.dctprox(2.0)

	#async def proxx(x, a, z):
	#	pass

	def callback(x, i):
		#ort.image_nd(x)
		pass


	await solvers.fista(np, image, alpha_i, gradx, proxx, 100, callback=callback)

	return image


async def run_framed(smaps, image, coords, kdatas, weights, alpha_i, lamda, max_iter, callback):
	nenc = 5
	nframes = len(coords) // nenc

	imsize = (nenc,) + smaps.shape[1:]

	devicectxdict = {"dev": cp.cuda.Device(0), "ntransf": 16, "imsize": imsize[1:]}
	async def gradx(ximg, a):
		normlist = await grad.gradient_step_x(smaps, ximg, coords, kdatas, weights,
				a, [devicectxdict], calcnorm=True)
		
		print(f'Error = {np.array([a.item() for a in normlist[0]]).sum()}')

	proxx = prox.svtprox(lamda, np.array([8, 8, 8]), np.array([8, 8, 8]), 3)

	await solvers.fista(np, image, alpha_i, gradx, proxx, max_iter, callback)


async def runner(wexponent, lamda, max_iters, create_image_full, create_framed_weights, run_find_alpha, start_method, only_init, with_noise):

	base_path = '/home/turbotage/Documents/4DRecon/'

	coords, kdatas, nframes, nenc = simri.load_coords_kdatas(base_path)
	
	list = []
	for i in range(nframes):
		for j in range(nenc):
			list.append(coords[i][j])
	coords = list
	list = []

	if with_noise:
		noise_vec = np.load(base_path + 'noise_vec.npz')
		noise_vec = [noise_vec[a] for a in noise_vec]
	for i in range(nframes):
		for j in range(nenc):
			kd = kdatas[i][j]
			if with_noise:
				kd += noise_vec[i*nenc + j]
			list.append(kd)
	kdatas = list

	true_image, vessel_mask, smaps = simri.load_true_and_smaps(base_path)
	true_image = true_image.reshape((nframes * nenc,) + smaps.shape[1:])
	true_image /= np.abs(np.mean(true_image))
	true_image_norm = np.linalg.norm(true_image)

	true_phases = np.angle(true_image[...,vessel_mask])
	true_phase_norm = np.linalg.norm(true_phases)

	if create_image_full:
		print('Creating full image')
		image_full = await run_full(smaps, coords, kdatas)
		with h5py.File(base_path + 'reconed_full.h5', 'w') as hf:
			hf.create_dataset('image', data=image_full)
	else:
		print('Loading full image')
		with h5py.File(base_path + 'reconed_full.h5', 'r') as hf:
			image_full = np.array(hf['image'])
	if create_framed_weights:
		print('Creating framed weights')
		weights = []
		for i in range(nframes):
			w = dcf.pipe_menon_dcf(160*coords[nenc*i]/np.pi, (320,320,320), max_iter=50)
			w = w / w.max()
			for j in range(nenc):
				weights.append(w)
		with h5py.File(base_path + 'reconed_framed_weights.h5', 'w') as hf:
			for i, w in enumerate(weights):
				hf.create_dataset(f'weights_{i}', data=weights[i])
	else:
		print('Loading framed weights')
		weights = []
		with h5py.File(base_path + 'reconed_framed_weights.h5', 'r') as hf:
			for i in range(len(hf.keys())):
				weights.append(np.array(hf[f'weights_{i}']))

	weights = [w**wexponent for w in weights]

	imsize = (nframes*nenc,) + smaps.shape[1:]

	if run_find_alpha:
		print('Finding alpha')
		alpha_i = await find_alpha(smaps, coords, weights, imsize)
		np.save(base_path + f"alpha_{wexponent}.npy", alpha_i)
	else:
		print('Loading alpha')
		alpha_i = np.load(base_path + 'alpha_i.npy').item()

	if only_init:
		return

	img0 = np.empty(imsize, dtype=np.complex64)

	if start_method == "zero":
		img0[:] = 0
	elif start_method == "mean":
		for i in range(nframes):
			for j in range(nenc):
				img0[i*nenc + j,...] = image_full[j,...]
	elif start_method == "diff":
		img0[:] = 0
		smaps_cu = cp.array(smaps)
		for frame in range(nframes):
			for enc in range(nenc):
				coord = cp.array(coords[frame*nenc + enc])
				kd = cp.empty_like(kdatas[frame*nenc + enc])

				dimg = cp.array(image_full[enc, ...]) * smaps_cu

				cufinufft.nufft3d2(x=coord[0,:], y=coord[1,:], z=coord[2,:], data=dimg, out=kd)
				kd /= np.sqrt(dimg.size)

				kdatas[frame*nenc + enc] -= kd.get()

	cp.get_default_memory_pool().free_all_blocks()

	err_rel = []
	err_max = []

	err_phase_rel = []
	err_phase_max = []
	err_phase_mean = []

	def callback(x,iter):

		xscaled = x.copy()
		if start_method == "diff":
			for frame in range(nframes):
				for enc in range(nenc):
					xscaled[frame*nenc + enc,...] += image_full[enc,...]

		xscaled = xscaled / np.abs(np.mean(xscaled))

		xdiff = xscaled - true_image

		err_rel.append(np.linalg.norm(xdiff) / true_image_norm)
		err_max.append(np.max(np.abs(xdiff)))

		xphases = np.angle(xscaled[...,vessel_mask])
		phase_diff = xphases - true_phases
		phase_diff = np.arctan2(np.sin(phase_diff), np.cos(phase_diff))

		phase_diff = np.abs(phase_diff)
		err_phase_rel.append(np.linalg.norm(phase_diff) / true_phase_norm)
		err_phase_max.append(np.max(phase_diff))
		err_phase_mean.append(np.mean(phase_diff))
		

		if iter in max_iters:
			np.save(base_path + f"results/err_rel_w{wexponent:.3e}_l{lamda:.3e}_i{iter}_{start_method}.npy", np.array(err_rel))
			np.save(base_path + f"results/err_max_w{wexponent:.3e}_l{lamda:.3e}_i{iter}_{start_method}.npy", np.array(err_max))
			np.save(base_path + f"results/err_phase_rel_w{wexponent:.3e}_l{lamda:.3e}_i{iter}_{start_method}.npy", np.array(err_phase_rel))
			np.save(base_path + f"results/err_phase_max_w{wexponent:.3e}_l{lamda:.3e}_i{iter}_{start_method}.npy", np.array(err_phase_max))
			np.save(base_path + f"results/err_phase_mean_w{wexponent:.3e}_l{lamda:.3e}_i{iter}_{start_method}.npy", np.array(err_phase_mean))

			#with h5py.File(base_path + f"results/reconed_framed_w{wexponent:.3e}_l{lamda:.3e}_i{i}_{start_method}.h5", 'w') as hf:
			#	hf.create_dataset('image', data=x)

	callback(img0, 0)

	await run_framed(smaps, img0, coords, kdatas, weights, alpha_i, lamda, max_iters[-1], callback)

async def create_noise(noise_fraction):
	# calc power of signal
	coords, kdatas, nframes, nenc = simri.load_coords_kdatas(base_path)

	power = 0
	signal_length = 0
	shapes = []
	for frame in range(nframes):
		for enc in range(nenc):
			kd = kdatas[frame][enc]
			power += np.sum(np.square(np.abs(kd)))
			signal_length += kd.size
			shapes.append(kd.shape)

	power /= signal_length

	noise_power = np.sqrt(power * noise_fraction) / np.sqrt(2)

	noise = []
	for shape in shapes:
		noise.append(noise_power * (np.random.randn(*shape) + 1j*np.random.randn(*shape)))

	np.savez(base_path + 'noise_vec', *noise)

async def big_runner(wexps, lambdas, with_noise):
	max_iters = [200, 201]
	for w in wexps:
		await runner(w, 0, 0, False, False, True, "", True, with_noise)
		for l in lambdas:
			await runner(w, l, max_iters, False, False, False, "diff", False, with_noise)
			await runner(w, l, max_iters, False, False, False, "zero", False, with_noise)
			await runner(w, l, max_iters, False, False, False, "mean", False, with_noise)

if __name__ == "__main__":

	#asyncio.run(create_noise(0.001))
	asyncio.run(runner(0.5, 0, 0, True, False, False, "", True, True))