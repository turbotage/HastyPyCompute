import math
import cupy as cp
import numpy as np
import h5py

async def fista(xp, x, alpha, gradstep, prox, maxiter, callback):

    t = xp.array([1.0])
    resids = []

    x_old = xp.empty_like(x)
    z = xp.empty_like(x)
    xp.copyto(z, x)

    async def update():
        xp.copyto(x_old, x)
        xp.copyto(x, z)

        await gradstep(x, alpha)
        await prox(x, alpha, z)

        t_old = t
        t[:] = 0.5 * (1.0 + math.sqrt(1.0 + 4.0*t_old*t_old))

        xp.subtract(x, x_old, out=z)
        resids.append(xp.linalg.norm(z))
        xp.add(x, ((t_old - 1.0) / t) *z, out=z)

    for i in range(maxiter):
        await update()
        if callback is not None:
            callback(x, i)
            
    return resids

    


async def max_eig(xp, A, x, iter):
    xup = xp.copy(x)
    for i in range(iter):
        y = await A(xup)
        maxeig = xp.linalg.norm(y)
        print(maxeig)
        xup = y / maxeig
    return maxeig