import cupy as cp
import cufinufft
import numpy as np
from scipy.signal import tukey
import math
import util

import grad
import solvers
import prox
import gc


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
	await solvers.fista(np, img, alpha_i, gradx, proxx, 15)

	alpha_s = 0.125 / await solvers.max_eig(np, snormal, util.complex_rand(smp.shape, xp=np), 8)

	async def update():
		print('S update:')		
		await solvers.fista(np, smp, alpha_s, grads, proxs, iter[1][1])
		print('I update:')
		await solvers.fista(np, img, alpha_i, gradx, proxx, iter[1][0])

	for it in range(iter[0]):
		await update()

	return smp, img, alpha_i
		

async def walsh(coord, kdata, weights, im_size, blocksize=(8, 8, 8)):

	dim = len(im_size)
	ncoil = kdata[0].shape[0]
	nencs = len(kdata)
	Np = nencs*math.prod(blocksize)
	

	normfactor = 1.0 / math.sqrt(math.prod(im_size))

	coil_images = cp.zeros((nencs, ncoil,) + im_size, dtype=kdata[0].dtype)
	smaps = cp.zeros((ncoil,) + im_size, dtype=kdata[0].dtype)
	coordcu = cp.array(coord)
	weightscu = cp.array(weights)



	if dim == 3:

		# Calculate coil images
		for e in range(nencs):
			for i in range(ncoil):
				kdatacu = cp.array(kdata[e][i,...]) * weightscu[e,...]
				ci = coil_images[e,i,...]

				kdatacu *= normfactor

				cufinufft.nufft3d1(x=coordcu[e,0,:], y=coordcu[e,1,:], z=coordcu[e,2,:], data=kdatacu,
					n_modes=coil_images.shape[2:], out=ci, eps=1e-5)
			

		# Use coil with maximum signal as reference
		ref_c = 0
		max_i = 0
		for c in range(ncoil):
			intensity = 0
			for e in range(nencs):
				intensity += cp.sum(cp.linalg.norm(coil_images[e, c]))

			if intensity > max_i:
				ref_c = c
				max_i = intensity


		nblocks = math.prod(im_size)
		# Get Blocks and make SVD
		for b in range(nblocks):

			idx = getBlockidx(b, im_size)

			xstart, xstop = getStartStop(idx[0], blocksize[0], im_size[0])
			ystart, ystop = getStartStop(idx[1], blocksize[1], im_size[1])
			zstart, zstop = getStartStop(idx[2], blocksize[2], im_size[2])

			R = cp.zeros((ncoil, Np), dtype=kdata[0].dtype)
			for c in range(ncoil):
				count = 0
				for e in range(nencs):
					for z in range(zstart, zstop):
						for y in range(ystart, ystop):
							for x in range(xstart, xstop):
								R[c, count] = coil_images[e, c, x, y, z]
								count += 1

			U, S, Vh = cp.linalg.svd(R, full_matrices=False)

			for c in range(ncoil):
				temp = cp.sqrt(S[0]*U[c, 0]*cp.conj(U[ref_c, 0])/cp.abs(U[ref_c, 0]))

				smaps[c, idx[0], idx[1], idx[2]] = temp

		# SOS of coil images
		smaps = smaps.get()

		gc.collect()

		sos = np.sqrt(np.sum(np.square(np.abs(smaps)), axis=0))
		sos += np.max(sos)*1e-5

		smaps = smaps / sos
		


		return smaps
	else:
		raise RuntimeError('Not Implemented Dimension')
	


def getBlockidx(block, im_size):
	dims = len(im_size)
	idx = np.zeros(3)
	counti = block
	for dim in range(dims):
		idx[dim] = counti % im_size[dim]
		counti = (counti - idx[dim])/im_size[dim]
	return idx


def getStartStop(idxi, block_size_i, N):
	istart = idxi - block_size_i / 2
	istop = idxi + block_size_i / 2

	if (istart < 0):
		istart = 0
		istop = block_size_i
	
	if (istop > N):
		istop = N
		istart = N - block_size_i

	return int(istart), int(istop)