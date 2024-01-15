import math
import numba as nb
import numpy as np


@nb.jit(nopython=True, cache=True, parallel=True)  # pragma: no cover
def svt_numba3(output, input, lamda, blk_shape, blk_strides, block_iter, num_encodes):

    bz = input.shape[1] // blk_strides[0]
    by = input.shape[2] // blk_strides[1]
    bx = input.shape[3] // blk_strides[2]

    Sz = int(input.shape[1])
    Sy = int(input.shape[2])
    Sx = int(input.shape[3])

    scale = float(1.0 / block_iter)

    num_frames = int(input.shape[0]/num_encodes)
    # bmat_shape = (num_frames, num_encodes*blk_shape[0] * blk_shape[1] * blk_shape[2])
    bmat_shape = (num_encodes*blk_shape[0] * blk_shape[1] * blk_shape[2], num_frames)

    shifts = np.zeros((3, block_iter), np.int32)
    for d in range(3):
        for biter in range(block_iter):
            shifts[d,biter] = np.random.randint(blk_shape[d])

    for iter in range(block_iter):
        #print('block iter = ',iter)
        shiftz = shifts[0, iter]
        shifty = shifts[1, iter]
        shiftx = shifts[2, iter]

        for nz in nb.prange(bz):
            sz = nz * blk_strides[0] + shiftz
            ez = sz + blk_shape[0]

            for ny in range(by):
                sy = ny * blk_strides[1] + shifty
                ey = sy + blk_shape[1]

                for nx in range(bx):

                    sx = nx * blk_strides[2] + shiftx
                    ex = sx + blk_shape[2]

                    block = np.zeros(bmat_shape, input.dtype)

                    # Grab a block
                    for tframe in range(num_frames):
                        count = 0
                        for encode in range(num_encodes):
                            store_pos = int(tframe * num_encodes + encode)
                            for k in range(sz, ez):
                                for j in range(sy, ey):
                                    for i in range(sx, ex):
                                        # block[tframe, count] = input[store_pos, k % Sz, j % Sy, i % Sx]
                                        block[count, tframe] = input[store_pos, k % Sz, j % Sy, i % Sx]
                                        count += 1

                    # Svd
                    u, s, vh = np.linalg.svd(block, full_matrices=False)

                    for k in range(u.shape[1]):

                        # s[k] = max(s[k] - lamda, 0)
                        abs_input = abs(s[k])
                        if abs_input == 0:
                            sign = 0
                        else:
                            sign = s[k] / abs_input

                        s[k] = abs_input - lamda
                        s[k] = (abs(s[k]) + s[k]) / 2
                        s[k] = s[k] * sign

                        for i in range(u.shape[0]):
                            u[i, k] *= s[k]

                    block = np.dot(u, vh)

                    # Put block back
                    for tframe in range(num_frames):
                        count = 0
                        for encode in range(num_encodes):
                            store_pos = int(tframe * num_encodes + encode)
                            for k in range(sz, ez):
                                for j in range(sy, ey):
                                    for i in range(sx, ex):
                                        # output[store_pos, k % Sz, j % Sy, i % Sx] += scale*block[tframe, count]
                                        output[store_pos, k % Sz, j % Sy, i % Sx] += scale*block[count, tframe]
                                        count += 1

    return output


@nb.jit(nopython=True, cache=True, parallel=True)  # pragma: no cover
def svt_numba2(output, input, lamda, blk_shape, blk_strides, block_iter, num_encodes):

    by = input.shape[1] // blk_strides[0]
    bx = input.shape[2] // blk_strides[1]

    Sy = int(input.shape[1])
    Sx = int(input.shape[2])

    scale = float(1.0 / block_iter)

    num_frames = int(input.shape[0]/num_encodes)
    bmat_shape = (num_encodes*blk_shape[0] * blk_shape[1], num_frames)

    shifts = np.zeros((2, block_iter), np.int32)
    for d in range(2):
        for biter in range(block_iter):
            shifts[d,biter] = np.random.randint(blk_shape[d])

    for iter in range(block_iter):
        #print('block iter = ',iter)
        shiftx = shifts[0, iter]
        shifty = shifts[1, iter]

        for ny in nb.prange(by):
            sy = ny * blk_strides[0] + shifty
            ey = sy + blk_shape[0]

            for nx in range(bx):

                sx = nx * blk_strides[1] + shiftx
                ex = sx + blk_shape[1]

                block = np.zeros(bmat_shape, input.dtype)

                # Grab a block
                for tframe in range(num_frames):
                    count = 0
                    for encode in range(num_encodes):
                        store_pos = int(tframe * num_encodes + encode)
                        for j in range(sy, ey):
                            for i in range(sx, ex):
                                block[count, tframe] = input[store_pos, j % Sy, i % Sx]
                                count += 1

                # Svd
                u, s, vh = np.linalg.svd(block, full_matrices=False)

                for k in range(u.shape[1]):

                    # s[k] = max(s[k] - lamda, 0)
                    abs_input = abs(s[k])
                    if abs_input == 0:
                        sign = 0
                    else:
                        sign = s[k] / abs_input

                    s[k] = abs_input - lamda
                    s[k] = (abs(s[k]) + s[k]) / 2
                    s[k] = s[k] * sign

                    for i in range(u.shape[0]):
                        u[i, k] *= s[k]

                block = np.dot(u, vh)

                # Put block back
                for tframe in range(num_frames):
                    count = 0
                    for encode in range(num_encodes):
                        store_pos = int(tframe * num_encodes + encode)
                        for j in range(sy, ey):
                            for i in range(sx, ex):
                                output[store_pos, j % Sy, i % Sx] += scale*block[count, tframe]
                                count += 1

    return output