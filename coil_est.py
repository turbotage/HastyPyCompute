import cupy as cp
import cufinufft
import numpy as np
from scipy.signal import tukey
import math
import util

from jinja2 import Template

import grad
import solvers
import prox
import gjin
import re

import numba as nb

def coil_covariance(x):
    covmat = np.empty((x.shape[0], x.shape[0]))
    for i in range(x.shape[0]):
        for j in range(x.shape[0]):
            covmat[i,j] = np.corrcoef(x[i,...], x[j,...])[0,1]
    return covmat

def coil_compress(kdata, axis=0, target_channels=None):
    ncoils = kdata[0].shape[0]
    nspokes = kdata[0].shape[1]
    pnperspoke = kdata[0].shape[2]

    for e in range(len(kdata)):
        kdata[e] = np.ascontiguousarray(kdata[e]).reshape((ncoils, nspokes * pnperspoke))
    
    kdata_cc = kdata[0]
    # Pick out only 5% of the data for SVD

    mask = np.random.rand(kdata_cc.shape[1])<0.1

    kcc = np.empty((ncoils, np.sum(mask).item()), dtype=kdata_cc.dtype)

    for c in range(ncoils):
        kcc[c, :] = kdata_cc[c,...][mask]

    # SVD
    U, S, _ = np.linalg.svd(kcc, full_matrices=False)

    for e in range(len(kdata)):
        kdata[e] = np.squeeze(np.matmul(U, kdata[e]))
        kdata[e] = np.reshape(kdata[e], (ncoils, nspokes, pnperspoke))[:target_channels, ...]

    return kdata

async def low_res_sensemap(coord, kdata, weights, im_size, tukey_param=(0.95, 0.95, 0.95), exponent=3):

	dim = len(im_size)
	ncoil = kdata.shape[0]

	normfactor = 1.0 / math.sqrt(math.prod(im_size))

	coil_images = cp.zeros((ncoil,) + im_size, dtype=kdata.dtype)
	coil_images_filtered = cp.empty_like(coil_images)
	coordcu = cp.array(coord)
	weightscu = cp.array(weights)

	t1 = cp.array(tukey(im_size[0], tukey_param[0]))
	t2 = cp.array(tukey(im_size[1], tukey_param[1]))
	t3 = cp.array(tukey(im_size[2], tukey_param[2]))
	window_prod = cp.meshgrid(t1, t2, t3)
	window = (window_prod[0] * window_prod[1] * window_prod[2]).reshape(im_size)
	del window_prod, t1, t2, t3
	gc.collect()
	window **= exponent

	if dim == 3:

		for i in range(ncoil):
			kdatacu = cp.array(kdata[i,...]) * weightscu
			ci = coil_images[i,...]

			kdatacu *= normfactor

			cufinufft.nufft3d1(x=coordcu[0,:], y=coordcu[1,:], z=coordcu[2,:], data=kdatacu,
				n_modes=coil_images.shape[1:], out=ci, eps=1e-5)
			
			cif = coil_images_filtered[i,...]
			cif[:]	= cp.fft.fftshift(cp.fft.fftn(cp.fft.ifftshift(ci)))
			cif[:] *= window
			cif[:] = cp.fft.fftshift(cp.fft.ifftn(cp.fft.ifftshift(cif)))

		del window

		coil_images_filtered = coil_images_filtered.get()
		coil_images = coil_images.get()

		gc.collect()

		sos = np.sqrt(np.sum(np.square(np.abs(coil_images_filtered)), axis=0))
		sos += np.max(sos)*1e-5

		smaps = coil_images_filtered / sos
		del coil_images_filtered
		image = np.sum(np.conj(smaps) * coil_images, axis=0) / np.sum(np.conj(smaps)*smaps, axis=0)

		cp.get_default_memory_pool().free_all_blocks()

		return smaps, image[None,...]
	else:
		raise RuntimeError('Not Implemented Dimension')

async def isense(img, smp, coord, kdata, weights, devicectx: grad.DeviceCtx, iter=[10,[5,5]], lamda=[0.1, 0.1]):
	dev = cp.cuda.Device(0)

	async def snormal(smpnew):
		return await grad.gradient_step_s(smpnew, img, coord, None, weights, None, devicectx)

	async def inormal(imgnew):
		return await grad.gradient_step_x(smp, imgnew, [coord], None, [weights], None, [devicectx])

	async def gradx(ximg, a):
		norm = await grad.gradient_step_x(smp, ximg, [coord], [kdata], [weights], a, [devicectx], calcnorm=True)
		print(f"Data Error: {norm[0][0]}")
	
	async def grads(smp, a):
		norm = await grad.gradient_step_s(smp, img, coord, kdata, weights, a, devicectx, calcnorm=True)
		print(f"Data Error: {norm}")

	proxx = prox.dctprox(lamda[0])
	proxs = prox.spatial_svtprox(lamda[1], np.array([32,32,32]), np.array([32,32,32]), 2)

	#alpha_i = 0.5 / await solvers.max_eig(cp, inormal, cp.ones_like(img), 8)
	alpha_i = 0.25 / await solvers.max_eig(np, inormal, util.complex_rand(img.shape, xp=np), 8)

	img.fill(0.0)
	await solvers.fista(np, img, alpha_i, gradx, proxx, 10)

	alpha_s = 0.125 / await solvers.max_eig(np, snormal, util.complex_rand(smp.shape, xp=np), 8)

	async def update():
		print('S update:')		
		await solvers.fista(np, smp, alpha_s, grads, proxs, iter[1][1])
		print('I update:')
		await solvers.fista(np, img, alpha_i, gradx, proxx, iter[1][0])

	for it in range(iter[0]):
		await update()

	return smp, img, alpha_i
		


@nb.jit(nopython=True, cache=True, parallel=True, nogil=True)
def block_fetcher_3d_numba(coil_images, zrange, yrange, xrange, blk_size):

	nenc = coil_images.shape[0]
	ncoil = coil_images.shape[1]
	imsize = coil_images.shape[2;]

	nblock = (zrange[1] - zrange[0]) * (yrange[1] - yrange[0]) * (xrange[1] - xrange[0])

	large_block = np.empty((nblock,ncoil,nenc*math.prod(blk_size)), dtype=coil_images.dtype)

	for ny in nb.prange(yrange[0], yrange[1]):
		for nx in range(xrange[0], xrange[1]):
			for nz in range(zrange[0], zrange[1]):
				
				idx = [nx, ny, nz]

				start = [0,0,0]
				end = [0,0,0]
				for i in range(3):
					startl = idx[i] - blk_size[i] // 2
					endl = idx[i] + blk_size[i] // 2

					if startl < 0:
						startl = 0
						endl = startl + blk_size[i]
					elif endl > imsize[i]:
						startl = imsize[i] - blk_size[i]
						endl = imsize[i]

					start[i] = startl
					end[i] = endl

				for c in range(ncoil):
					count = 0
					for e in range(nenc):
						for x in range(start[0], end[0]):
							for y in range(start[1], end[1]):
								for z in range(start[2], end[2]):
									large_block[count,c,e*math.prod(blk_size):(e+1)*math.prod(blk_size)] = coil_images[c,e,x,y,z]
									count += 1

	return large_block



@nb.jit(nopython=True, cache=True, parallel=True, nogil=True)
def block_pusher_3d_numba(coil_images_out, U, S, xrange, yrange, zrange, refc):
	blockcount = 0
	for nx in nb.prange(xrange[0], xrange[1]):
		for ny in range(yrange[0], yrange[1]):
			for nz in range(zrange[0], zrange[1]):
				
				idx = [nx, ny, nz]

				slocal = S[blockcount,...]
				ulocal = U[blockcount,...]

				for c in range(coil_images_out.shape[0]):
					temp = np.sqrt(slocal[0]*ulocal[c,0]*np.conj(ulocal[refc,0]) / np.abs(U[refc,0]))
					coil_images_out[c, idx[0], idx[1], idx[2]] = temp
    

#def create coil_images 

async def walsh(coil_images, refc, blk_size):
	# loop over xrange, yrange and zrange in block_fetcher_3d_numba
	# in every loop, do svd stuff on large_block, and then output into output coil images

	nenc = coil_images.shape[0]
	ncoil = coil_images.shape[1]
	imsize = coil_images.shape[2:]



	xranges = [
				[0, 			 		imsize[0] // 4], 
				[imsize[0] // 4, 		imsize[0] // 2],
				[imsize[0] // 2, 		3 * (imsize[0] // 4)]
				[3 * (imsize[0] // 4), 	imsize[0]]
	]

	zranges = xranges
	coil_images_out = np.empty((ncoil, imsize[0], imsize[1], imsize[2]), dtype=coil_images.dtype)

	for xrange in xranges:
		for y in range(imsize[1]):
			for zranges in zranges:
				large_block = block_fetcher_3d_numba(coil_images, zranges, [y, y+1], xrange, blk_size)

				# Loop over blocks to not run out of memory on gpu
				large_block = cp.array(large_block)
				U, S, _ = cp.linalg.svd(large_block, full_matrices=False)
				U = U.get()
				S = S.get()
		
				block_pusher_3d_numba(coil_images_out, U, S, xrange, [y, y+1], zranges, refc)

	return coil_images_out

    
	
