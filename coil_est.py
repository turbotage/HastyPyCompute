import cupy as cp
import cufinufft
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

		sos = cp.sqrt(cp.sum(cp.square(cp.abs(coil_images_filtered)), axis=0))
		sos += cp.max(sos)*1e-5

		smaps = coil_images_filtered / sos
		del coil_images_filtered
		image = cp.sum(cp.conj(smaps) * coil_images, axis=0) / cp.sum(cp.conj(smaps)*smaps, axis=0)

		return smaps.get(), image[None,...].get()
	else:
		raise RuntimeError('Not Implemented Dimension')

async def isense(img, smp, coord, kdata, weights, devicectx: grad.DeviceCtx, iter=[10,[5,5]], lamda=[0.1, 0.1]):
	dev = cp.cuda.Device(0)

	async def snormal(smpnew):
		return await grad.device_gradient_step_s(smpnew, img, coord, None, weights, None, devicectx)

	async def inormal(imgnew):
		return await grad.device_gradient_step_x(smp, imgnew, [coord], None, [weights], None, devicectx)

	async def gradx(ximg, a):
		norm = await grad.device_gradient_step_x(smp, ximg, [coord], [kdata], [weights], a, devicectx, calcnorm=True)
		print(f"Data Error: {norm[0]}")
	
	async def grads(smp, a):
		norm = await grad.device_gradient_step_s(smp, img, coord, kdata, weights, a, devicectx, calcnorm=True)
		print(f"Data Error: {norm}")

	proxx = prox.dctprox(lamda[0])
	proxs = prox.fftprox(lamda[1])

	#alpha_i = 0.5 / await solvers.max_eig(cp, inormal, cp.ones_like(img), 8)
	alpha_i = 0.25 / await solvers.max_eig(cp, inormal, util.complex_rand(img.shape, xp=cp), 8)

	img.fill(0.0)
	await solvers.fista(cp, img, alpha_i, gradx, proxx, 15)

	alpha_s = 0.125 / await solvers.max_eig(cp, snormal, smp, 8)

	async def update():
		print('S update:')		
		await solvers.fista(cp, smp, alpha_s, grads, proxs, iter[1][0])
		print('I update:')
		await solvers.fista(cp, img, alpha_i, gradx, proxx, iter[1][1])

	for it in range(iter[0]):
		await update()

	return smp, img
		
