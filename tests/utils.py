import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../build'))
import parallel_ivf

def get_all_indexes(d, nlist):
    print(parallel_ivf.__dict__)
    return [
        parallel_ivf.IVFBase(d=d, nlist=nlist), 
        parallel_ivf.IVFSIMD(d=d, nlist=nlist),
        parallel_ivf.IVFSIMDQueryParallel(d=d, nlist=nlist)]
