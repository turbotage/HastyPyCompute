import numpy as np
import cupy as cp


#def rand_like(shape, dtype):
#    if dtype == np.complex64:
#        
#    elif dtype == np.complex128:
#        
#    else:
        
def get_array_backend(arr):
    if isinstance(arr, cp.ndarray):
        return cp
    elif isinstance(arr, np.ndarray):
        return np
    else:
        raise ValueError(f"Unsupported array type: {type(arr)}")

def complex_rand(shape, dtype=np.float32, xp=np):
    return xp.random.rand(*shape).astype(dtype) + 1j*xp.random.rand(*shape).astype(dtype)

def rand_vector(shape, num, bounds=[0.0, 1.0], dtype=np.float32):
    vec = []
    for i in range(num):
        if dtype == np.complex64:
            vec.append(bounds[0] + (bounds[1] - bounds[0])*complex_rand(shape, np.float32))
        elif dtype == np.complex128:
            vec.append(bounds[0] + (bounds[1] - bounds[0])*complex_rand(shape, np.float64))
        else:
            vec.append(bounds[0] + (bounds[1] - bounds[0])*np.random.rand(*shape).astype(dtype))
    return vec