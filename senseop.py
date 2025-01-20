from typing import TypeAlias
import threading
import numpy as np

import sys
print(sys._is_gil_enabled())
import torch
print(sys._is_gil_enabled())
#import finufft
#print(sys._is_gil_enabled())
#import cufinufft
#print(sys._is_gil_enabled())

class SenseOp:

    @torch.compile
    def run_inner(self, input, coilmap, kernel):
        spatial_shp = input.shape #shp[1:]
        expanded_shp = [2*s for s in spatial_shp]
        transform_dims = [i+1 for i in range(len(spatial_shp))]

        ncoil = coilmap.shape[0]
        nrun = ncoil // {0}
        
        out = torch.zeros_like(input)
        for run in range(nrun):
            bst = run*{0}
            cmap = coilmap[bst:(bst+{0})]
            c = cmap * input
            c = torch.fft_fftn(c, expanded_shp, transform_dims)
            c *= kernel
            c = torch.fft_ifftn(c, None, transform_dims)

            for dim in range(len(spatial_shp)):
                c = torch.slice(c, dim+1, spatial_shp[dim]-1, -1)

            c *= cmap.conj()
            out += torch.sum(c, 0)

        out *= (1 / torch.prod(torch.tensor(spatial_shp)))
        
        return out

    @torch.compile
    def run()

