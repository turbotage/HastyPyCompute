import cufinufft
import cupy as cp



image = (cp.random.rand(1,8)+1j*cp.random.rand(1,8)).astype(cp.complex64)
kernel_start = cp.empty((15,), dtype=cp.complex64)
kdata = (cp.random.rand(100) + 1j*cp.random.rand(100)).astype(cp.complex64)
kernel = cp.zeros((16,), dtype=cp.complex64)

cx = -3.141592 + 2*3.141592*cp.random.rand(100).astype(cp.float32)

cufinufft.nufft1d1(cx,kdata,(1,15),isign=-1,out=kernel_start)
kernel[:15] = kernel_start
kernel = cp.fft.fftn(kernel)
kernel = kernel[None,:]

def finufft_roundtrip(x):
    y = cp.empty_like(x)
    k = cufinufft.nufft1d2(cx,x, isign=1)
    cufinufft.nufft1d1(cx,k,(1,8),isign=-1,out=y)
    return y

def toeplitz_roundtrip(x):
    #tot = cp.zeros((16,), dtype=cp.complex64)
    return cp.fft.ifftn(cp.fft.fftn(x,(16,)) * kernel, (8,))


back_finufft = finufft_roundtrip(image)
back_toeplitz = toeplitz_roundtrip(image)


print('Hello')

