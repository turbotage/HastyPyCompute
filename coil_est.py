import cupy as cp
import cufinufft
from scipy.signal import tukey

import grad
import solvers

def low_res_sensemap(coord, kdata, weights, im_size, tukey_param=(0.95, 0.95, 0.95), exponent=3):

	dim = len(im_size)
	ncoil = kdata.shape[0]

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

		return smaps, image
	else:
		raise RuntimeError('Not Implemented Dimension')

def isense(i, s, coord, kdata, weights, iter=[10,[10,10]], lamda=[0.1, 0.1]):
	dev = cp.cuda.Device(0)

	def snormal(s):
		sup = cp.copy(s)
		sup *= i
		kout = cp.empty_like(kdata)
		sout = cp.empty_like(sup)
		cufinufft.nufft3d2(coord[0,:], coord[1,:], coord[2,:], sup, out=kout, eps=1e-5)
		kout *= weights
		cufinufft.nufft3d1(coord[0,:], coord[1,:], coord[2,:], kout, sup.shape, out=sout, eps=1e-5)
		sout *= cp.conj(i)
		return sout

	def inormal(i):
		iup = cp.copy(i)
		iup *= s
		kout = cp.empty_like(kdata)
		xout = cp.empty_like(i)
		cufinufft.nufft3d2(coord[0,:], coord[1,:], coord[2,:], iup, out=kout, eps=1e-5)
		kout *= weights
		cufinufft.nufft3d1(coord[0,:], coord[1,:], coord[2,:], kout, iup.shape, out=xout, eps=1e-5)
		xout *= cp.conj(s)
		return xout

	alpha_s = 1 / solvers.max_eig(snormal, s, 8)
	alpha_i = 1 / solvers.max_eig(inormal, i, 8)


	grads = lambda smp, a: grad.gradient_step_s(smp, i, coord, kdata, weights, dev, a)
	gradx = lambda ximg, a: grad.gradient_step_x(s, ximg, coord, kdata, weights, dev, a, True)

	def update():
		pass

	return i, s
		
