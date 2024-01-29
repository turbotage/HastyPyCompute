import cupy as cp
import cufinufft
import numpy as np
from scipy.signal import tukey
import math
import util

import grad
import solvers
import prox

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

		coil_images_filtered = coil_images_filtered.get()
		coil_images = coil_images.get()

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
	proxs = prox.spatial_svtprox(lamda[1], [16,16,16], [16,16,16], 4)

	#alpha_i = 0.5 / await solvers.max_eig(cp, inormal, cp.ones_like(img), 8)
	alpha_i = 0.25 / await solvers.max_eig(np, inormal, util.complex_rand(img.shape, xp=np), 8)

	img.fill(0.0)
	await solvers.fista(np, img, alpha_i, gradx, proxx, 15)

	alpha_s = 0.125 / await solvers.max_eig(np, snormal, util.complex_rand(smp.shape, xp=np), 8)

	async def update():
		print('S update:')		
		await solvers.fista(np, smp, alpha_s, grads, proxs, iter[1][0])
		print('I update:')
		await solvers.fista(np, img, alpha_i, gradx, proxx, iter[1][1])

	for it in range(iter[0]):
		await update()

	return smp, img
		
