import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../build'))
import parallel_ivf

def get_all_indexes(d, nlist):
    
    return [
        parallel_ivf.IVFBase(d=d, nlist=nlist), 
        parallel_ivf.IVFSIMD(d=d, nlist=nlist),
        parallel_ivf.IVFSIMDQueryParallel(d=d, nlist=nlist)]


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

