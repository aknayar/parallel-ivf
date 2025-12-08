import sys
import os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../build'))
import parallel_ivf

def get_all_indexes(d, nlist):
    
    return [
        parallel_ivf.IVFBase(d=d, nlist=nlist),
        parallel_ivf.IVFSIMD(d=d, nlist=nlist),
        parallel_ivf.IVFCache(d=d, nlist=nlist),
        parallel_ivf.IVFCacheSIMD(d=d, nlist=nlist),
        parallel_ivf.IVFCacheV2(d=d, nlist=nlist),
        parallel_ivf.IVFCacheV2SIMD(d=d, nlist=nlist),
        parallel_ivf.IVFSIMDQueryParallel(d=d, nlist=nlist),
        parallel_ivf.IVFSIMDCandidateParallel(d=d, nlist=nlist),
        parallel_ivf.IVFCacheQueryParallel(d=d, nlist=nlist),
        parallel_ivf.IVFCacheSIMDQueryParallel(d=d, nlist=nlist),
        parallel_ivf.IVFCacheV2QueryParallel(d=d, nlist=nlist),
        parallel_ivf.IVFCacheV2SIMDQueryParallel(d=d, nlist=nlist),
        parallel_ivf.IVFCacheCandidateParallel(d=d, nlist=nlist),
        parallel_ivf.IVFCacheSIMDCandidateParallel(d=d, nlist=nlist),
        parallel_ivf.IVFScalarQueryParallel(d=d, nlist=nlist),
        parallel_ivf.IVFScalarCandidateParallel(d=d, nlist=nlist)
    ]

def getIndex(name, d, nlist):
    indexes = []
    
    if "IVFBase" == name:
        indexes.append(parallel_ivf.IVFBase(d=d, nlist=nlist))
    if "IVFSIMD" == name:
        indexes.append(parallel_ivf.IVFSIMD(d=d, nlist=nlist))
    if "IVFCache" == name:
        indexes.append(parallel_ivf.IVFCache(d=d, nlist=nlist))
    if "IVFCacheSIMD" == name:
        indexes.append(parallel_ivf.IVFCacheSIMD(d=d, nlist=nlist))
    if "IVFCacheV2" == name:
        indexes.append(parallel_ivf.IVFCacheV2(d=d, nlist=nlist))
    if "IVFCacheV2SIMD" == name:
        indexes.append(parallel_ivf.IVFCacheV2SIMD(d=d, nlist=nlist))
    if "IVFOMPSIMD" == name:
        indexes.append(parallel_ivf.IVFOMPSIMD(d=d, nlist=nlist))
    if "IVFSIMDQueryParallel" == name:
        indexes.append(parallel_ivf.IVFSIMDQueryParallel(d=d, nlist=nlist))
    if "IVFSIMDCandidateParallel" == name:
        indexes.append(parallel_ivf.IVFSIMDCandidateParallel(d=d, nlist=nlist))
    if "IVFSIMDQueryCandidateParallel" == name:
        indexes.append(parallel_ivf.IVFSIMDQueryCandidateParallel(d=d, nlist=nlist))
    if "IVFCacheQueryParallel" == name:
        indexes.append(parallel_ivf.IVFCacheQueryParallel(d=d, nlist=nlist))
    if "IVFCacheQueryCandidateParallel" == name:
        indexes.append(parallel_ivf.IVFCacheQueryCandidateParallel(d=d, nlist=nlist))
    if "IVFCacheSIMDQueryParallel" == name:
        indexes.append(parallel_ivf.IVFCacheSIMDQueryParallel(d=d, nlist=nlist))
    if "IVFCacheV2QueryParallel" == name:
        indexes.append(parallel_ivf.IVFCacheV2QueryParallel(d=d, nlist=nlist))
    if "IVFCacheV2SIMDQueryParallel" == name:
        indexes.append(parallel_ivf.IVFCacheV2SIMDQueryParallel(d=d, nlist=nlist))
    if "IVFCacheQueryCandidateParallel" == name:
        indexes.append(parallel_ivf.IVFCacheQueryCandidateParallel(d=d, nlist=nlist))
    if "IVFCacheSIMDQueryParallel" == name:
        indexes.append(parallel_ivf.IVFCacheSIMDQueryParallel(d=d, nlist=nlist))
    if "IVFCacheCandidateParallel" == name:
        indexes.append(parallel_ivf.IVFCacheCandidateParallel(d=d, nlist=nlist))
    if "IVFCacheSIMDQueryCandidateParallel" == name:
        indexes.append(parallel_ivf.IVFCacheSIMDQueryCandidateParallel(d=d, nlist=nlist))
    if "IVFCacheSIMDCandidateParallel" == name:
        indexes.append(parallel_ivf.IVFCacheSIMDCandidateParallel(d=d, nlist=nlist))
    if "IVFScalarQueryParallel" == name:
        indexes.append(parallel_ivf.IVFScalarQueryParallel(d=d, nlist=nlist))
    if "IVFScalarCandidateParallel" == name:
        indexes.append(parallel_ivf.IVFScalarCandidateParallel(d=d, nlist=nlist))
    if "IVFScalarQueryCandidateParallel" == name:
        indexes.append(parallel_ivf.IVFScalarQueryCandidateParallel(d=d, nlist=nlist))
    if "IVFOMPSIMDQueryParallel" == name:
        indexes.append(parallel_ivf.IVFOMPSIMDQueryParallel(d=d, nlist=nlist))
    if "IVFOMPSIMDCandidateParallel" == name:
        indexes.append(parallel_ivf.IVFOMPSIMDCandidateParallel(d=d, nlist=nlist))
    if "IVFOMPSIMDQueryCandidateParallel" == name:
        indexes.append(parallel_ivf.IVFOMPSIMDQueryCandidateParallel(d=d, nlist=nlist))
    return indexes

def read_fvecs(filename, n_max=None):
    if not os.path.exists(filename):
        raise FileNotFoundError(f"File not found: {filename}")

    with open(filename, 'rb') as f:
        dim_header = np.fromfile(f, dtype=np.int32, count=1)
        
        if dim_header.size == 0:
            return np.array([], dtype=np.float32)
            
        d = dim_header[0]
        f.seek(0)
        row_width = d + 1
        
        count = -1
        if n_max is not None:
            count = n_max * row_width
            
        data = np.fromfile(f, dtype=np.float32, count=count)

    data = data.reshape(-1, row_width)
    return data[:, 1:].copy() 

def load_fvecs_data(base_path, learn_path, query_path, nb=None, nt=None, nq=None):
    xb = read_fvecs(base_path, n_max=nb)
    xt = read_fvecs(learn_path, n_max=nt)    
    xq = read_fvecs(query_path, n_max=nq)

    return xb, xt, xq
