import math
import cupy as cp
import numpy as np

def fista(xp, x, alpha, gradstep, prox, maxiter):

    t = xp.array([1.0])
    resids = []

    x_old = xp.empty_like(x)
    z = xp.empty_like(x)
    xp.copyto(z, x)

    def update():
        xp.copyto(x_old, x)
        xp.copyto(x, z)

        gradstep(x, alpha)

        prox(x, alpha, z)

        t_old = t
        t[:] = 0.5 * (1.0 + math.sqrt(1.0 + 4.0*t_old*t_old))

        xp.subtract(x, x_old, out=z)
        resids.append(xp.linalg.norm(z))
        xp.add(x, ((t_old - 1.0) / t) *z, out=z)

    for i in range(maxiter):
        update()

    


def max_eig(A, x, iter):
    xup = cp.copy(x)
    for i in range(iter):
        y = A(xup)
        max_eig = cp.linalg.norm(y)
        xup = y / max_eig
    return max_eig