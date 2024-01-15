import cufinufft
import cupy as cp



image = (cp.random.rand(32,32,32)+1j*cp.random.rand(32,32,32)).astype(cp.complex64)
kernel_start = cp.empty((63,63,63), dtype=cp.complex64)
kdata = cp.ones((10000,), dtype=cp.complex64)
kernel = cp.zeros((64,64,64), dtype=cp.complex64)

cx = -3.141592 + 2*3.141592*cp.random.rand(10000).astype(cp.float32)
cy = -3.141592 + 2*3.141592*cp.random.rand(10000).astype(cp.float32)
cz = -3.141592 + 2*3.141592*cp.random.rand(10000).astype(cp.float32)

cufinufft.nufft3d1(cx,cy,cz,kdata,(1,63,63,63),out=kernel_start)

kernel[:63,:63,:63] = kernel_start


def finufft_roundtrip(x):
    y = cp.empty_like(x)
    k = cufinufft.nufft3d2(cx,cy,cz,x)
    return cufinufft.nufft3d1(cx,cy,cz,k,(1,32,32,32),out=y) / (32*32*32)

def toeplitz_roundtrip(x):
    tot = cp.zeros((64,64,64), dtype=cp.complex64)
    tot[:32,:32,:32] = x
    return cp.fft.ifftn(cp.fft.fftn(tot) * kernel)


back_finufft = finufft_roundtrip(image)
back_toeplitz = toeplitz_roundtrip(image)


print('Hello')





