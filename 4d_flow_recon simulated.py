import svt

import math
import time
import numpy as np
import random
import h5py
from sigpy import dcf as dcf

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
	devicectxdict = {"dev": cp.cuda.Device(0), "ntransf": 32, "imsize": imsize[1:]}

	async def inormal(imgnew):
		return await grad.gradient_step_x(smaps, imgnew, coords, None, weights, None, [devicectxdict])

	alpha_i = 0.5 / await solvers.max_eig(np, inormal, util.complex_rand(imsize, xp=np), 8)

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
	
	
	weights = dcf.pipe_menon_dcf(full_coords[i]/np.pi*imsize[0]/2, imsize[1:], max_iter=30)
	weights = [weights for _ in range(nenc)]
		
	weights = [(w / w.max())**0.5 for w in weights]

	img = np.zeros((64,64,64), dtype=np.complex64)

	crd = [cp.array(full_coords[0][0,:]), cp.array(full_coords[0][1,:]), cp.array(full_coords[0][2,:])]
	for i in range(32):
		kd = cp.array(full_kdatas[0][i,...] * weights[0])
		img += cufinufft.nufft3d1(x=crd[0], y=crd[1], z=crd[2], data=kd, n_modes=imsize[1:]).get() / smaps[i,...]


	alpha_i = 0.01 #await find_alpha(smaps, full_coords, weights, imsize)

	devicectxdict = {"dev": cp.cuda.Device(0), "ntransf": 32, "imsize": imsize[1:]}

	async def gradx(ximg, a):
		normlist = await grad.gradient_step_x(smaps, ximg, full_coords, full_kdatas, weights,
				a, [devicectxdict], calcnorm=True)
		
		print(f'Error = {np.array([a.item() for a in normlist[0]]).sum()}')

	image = np.zeros(imsize, dtype=np.complex64)

	proxx = prox.dctprox(0.0)

	await solvers.fista(np, image, alpha_i, gradx, proxx, 30, callback=None)

	return image
	image_full = await run_full(smaps, coords, kdatas)



async def runner():
	imsize = (64,64,64)
	usePipeMenon = True

	lambda_x = 0.05 #round(10**(random.uniform(0, -4)), 5)
	lambda_s = round(10**(random.uniform(-2, -6)), 7)

	base_path = '/home/turbotage/Documents/4DRecon/'

	coords, kdatas, nframes, nenc = simri.load_coords_kdatas(base_path)
	
	list = []
	for i in range(nframes):
		for j in range(nenc):
			list.append(coords[i][j])
	coords = list
	list = []
	maxkd = []
	for i in range(nframes):
		for j in range(nenc):
			kd = kdatas[i][j]
			list.append(kd)
			maxkd.append(np.max(np.abs(kd)))
	kdatas = list

	maxkd = max(maxkd)
	kdatas = [kd / maxkd for kd in kdatas]

	true_image, smaps = simri.load_true_and_smaps(base_path)

	image_full = await run_full(smaps, coords, kdatas)

	run_framed(niter=100)

if __name__ == "__main__":
	asyncio.run(runner())