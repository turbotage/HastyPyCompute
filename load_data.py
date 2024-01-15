import h5py
import numpy as np
import multiprocessing as mp
import time


def load_one_encode(i, settings):
    with h5py.File(settings['file'], 'r') as f:
        kdataset = f['Kdata']
        keys = kdataset.keys()

        ret = ()

        if settings['load_coords']:
            coord = np.stack([
                        kdataset['KX_E'+str(i)][()],
                        kdataset['KY_E'+str(i)][()],
                        kdataset['KZ_E'+str(i)][()]
                    ], axis=0)
            ret += (coord,)
            
        if settings['load_weights']:
            weights = kdataset['KW_E'+str(i)][()]
            ret += (weights,)

        if settings['load_kdata']:
            kdata_vec = []
            for j in range(len(keys)):
                coilname = 'KData_E'+str(i)+'_C'+str(j)
                if coilname in kdataset:
                    kdata_vec.append(kdataset[coilname]['real'] + kdataset[coilname]['imag'])
            ret += (kdata_vec,)

        return ret

def load_five_point(file, load_coords=True, load_kdata=True, 
        load_weights=True, load_gating=True, gating_names=[]):

    settings = {
                'file': file,
                'load_coords': load_coords, 
                'load_kdata': load_kdata, 
                'load_weights': load_weights
                }

    with mp.Pool(processes=5) as pool:

        ret = ()

        results = [pool.apply_async(load_one_encode, (i,settings)) for i in range(5)]

        if load_gating:
            with h5py.File(file, 'r') as f:
                gatingset = f['Gating']

                gating = {}
                for gatename in gating_names:
                    gating[gatename] = gatingset[gatename][()]

                ret += (gating,)

        return ret + ([res.get(timeout=500) for res in results],)



start = time.time()

datamat = load_five_point('/home/turbotage/Documents/4DRecon/MRI_Raw.h5', gating_names=['TIME_E0', 'ECG_E0'])

end = time.time()

print(f"Time = {end - start}")



print('Hello')


