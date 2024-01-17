import cupy as cp
import cufinufft

from scipy.signal import tukey


def low_res_sensemap(coord, kdata, weights, im_size, tukey_param=(0.95, 0.95, 0.95), exponent=3):

    dim = len(im_size)
    ncoil = kdata.shape[0]

    coil_images = cp.zeros((ncoil,) + im_size, dtype=kdata.dtype)
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

            ci= cp.fft.fftshift(cp.fft.fftn(cp.fft.ifftshift(ci)))
            ci *= window
            ci = cp.fft.fftshift(cp.fft.ifftn(cp.fft.ifftshift(ci)))

        sos = cp.sqrt(cp.sum(cp.square(cp.abs(coil_images)), axis=0))
        sos += cp.max(sos)*1e-5

        return coil_images / sos
    else:
        raise RuntimeError('Not Implemented Dimension')

def isense():
    
