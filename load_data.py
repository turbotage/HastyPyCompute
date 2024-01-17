import h5py
import numpy as np
import cupy as cp

import asyncio
import concurrent

import time


def _load_one_encode(i, settings):
    with h5py.File(settings['file'], 'r') as f:
        kdataset = f['Kdata']
        keys = kdataset.keys()

        ret = ()

        if settings['load_coords']:
            coord = np.squeeze(np.stack([
                        kdataset['KX_E'+str(i)][()],
                        kdataset['KY_E'+str(i)][()],
                        kdataset['KZ_E'+str(i)][()]
                    ], axis=0))
            ret += (('coords', coord),)
            
        if settings['load_weights']:
            weights = np.squeeze(kdataset['KW_E'+str(i)][()])
            ret += (('weights', weights),)

        if settings['load_kdata']:
            kdata = []
            for j in range(len(keys)):
                coilname = 'KData_E'+str(i)+'_C'+str(j)
                if coilname in kdataset:
                    kdata.append(kdataset[coilname]['real'] + kdataset[coilname]['imag'])
            kdata = np.squeeze(np.stack(kdata, axis=0))
            ret += (('kdata', kdata),)

        return ret

async def load_flow_data(file, num_encodes=5, load_coords=True, load_kdata=True, 
        load_weights=True, load_gating=True, gating_names=[]):

    settings = {
                'file': file,
                'load_coords': load_coords, 
                'load_kdata': load_kdata, 
                'load_weights': load_weights
                }

    loop = asyncio.get_event_loop()
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=num_encodes)

    ret = {}

    futures = []
    for encode in range(num_encodes):
        futures.append(loop.run_in_executor(executor, _load_one_encode, encode, settings))

    if load_gating:
        with h5py.File(file, 'r') as f:
            gatingset = f['Gating']

            gating = {}
            for gatename in gating_names:
                gating[gatename] = np.squeeze(gatingset[gatename][()])

            ret['gating'] = gating

    def get_val(key, resvec):
        for res in resvec:
            if res[0] == key:
                ret = res[1]
                del res
                return ret

    futures = [await fut for fut in futures]

    if load_coords:
        ret['coords'] = [get_val('coords', resvec) for resvec in futures]

    if load_kdata:
        ret['kdata'] = [get_val('kdata', resvec) for resvec in futures]

    if load_weights:
        ret['weights'] = [get_val('weights', resvec) for resvec in futures]

    return ret


async def gate_time(dataset, num_encodes=5):
    time_gating = dataset['gating']['TIME_E0']
    
    loop = asyncio.get_event_loop()
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=num_encodes)

    def do_gating(dataset, timeidx, encode):
        c = dataset['coords'][encode]
        c[:] = c[:,timeidx,:]

        k = dataset['kdata'][encode]
        k[:] = k[:,timeidx,:]

        if 'weights' in dataset:
            w = dataset['weights'][encode]
            w[:] = w[timeidx,:]

    timeidx = np.argsort(time_gating)
    futures = []
    for encode in range(num_encodes):
        futures.append(loop.run_in_executor(executor, do_gating, dataset, timeidx, encode))

    for fut in futures:
        await fut

    return dataset


async def crop_kspace(coords, kdatas, weights, im_size, crop_factors=(1.0,1.0,1.0), prefovkmuls=(1.0,1.0,1.0), postfovkmuls=(1.0,1.0,1.0)):

    kim_size = tuple(0.5*im_size[i]*crop_factors[i] for i in range(3))
    
    cp.fuse(kernel_name='crop_func')
    def crop_func(c, k, w):
        c[0,:] *= prefovkmuls[0]
        c[1,:] *= prefovkmuls[1]
        c[2,:] *= prefovkmuls[2]

        idxx = cp.abs(c[0,:]) < kim_size[0]
        idxy = cp.abs(c[1,:]) < kim_size[1]
        idxz = cp.abs(c[2,:]) < kim_size[2]

        idx = cp.logical_and(idxx, cp.logical_and(idxy, idxz))

        c = coord[:,idx]
        c[0,:] *= postfovkmuls[0] * cp.pi / kim_size[0]
        c[1,:] *= postfovkmuls[1] * cp.pi / kim_size[1]
        c[2,:] *= postfovkmuls[2] * cp.pi / kim_size[2]

        c = cp.maximum(cp.minimum(upp_bound, c), -upp_bound)

        k = k[:,idx]
        
        if w is not None:
            w = w[idx]


    mem_stream = cp.cuda.Stream(non_blocking=True)

    upp_bound = 0.99999*cp.pi
    for i in range(len(coords)):
        coord = cp.empty_like(coords[i])
        coord.set(coords[i], stream=mem_stream)

        kdata = cp.empty_like(kdatas[i])
        kdata.set(kdatas[i], stream=mem_stream)

        if weights is not None:
            weight = cp.empty_like(weights[i])
            weight.set(weights[i], stream=mem_stream)
        else:
            weight = None

        mem_stream.synchronize()
        crop_func(coord, kdata, weight)
        mem_stream.synchronize()

        coords[i] = coord.get(mem_stream)
        kdatas[i] = kdata.get(mem_stream)
        weight[i] = weight.get(mem_stream)
        
    
    return (coords, kdatas, weights)


async def translate(coord_vec, kdata_vec, translation):

    cp.fuse(kernel_func='translace_func')
    def translate_func(k, m, c):
        k *= cp.exp(1j * cp.sum(m, c, axis=1))

    mem_stream = cp.cuda.Stream(non_blocking=True)

    mult = cp.array(list(translation))[...,None]
    for i in range(len(coord_vec)):

        coord = cp.empty_like(coord_vec[i])
        coord.set(coord_vec[i], stream=mem_stream)

        kdata = cp.empty_like(kdata_vec[i])        
        kdata.set(kdata_vec[i], stream=mem_stream)
        
        mem_stream.synchronize()

        translate_func(kdata, mult, coord)

        kdata_vec[i] = kdata.get(mem_stream)


    return kdata_vec



#start = time.time()
#datamat = load_five_point('/home/turbotage/Documents/4DRecon/MRI_Raw.h5', gating_names=['TIME_E0', 'ECG_E0'])
#end = time.time()
#print(f"Time = {end - start}")
#print('Hello')


