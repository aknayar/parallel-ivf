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
    
    if "IVFBase" in name:
        indexes.append(parallel_ivf.IVFBase(d=d, nlist=nlist))
    if "IVFSIMD" in name:
        indexes.append(parallel_ivf.IVFSIMD(d=d, nlist=nlist))
    if "IVFSIMDQueryParallel" in name:
        indexes.append(parallel_ivf.IVFSIMDQueryParallel(d=d, nlist=nlist))
    if "IVFSIMDCandidateParallel" in name:
        indexes.append(parallel_ivf.IVFSIMDCandidateParallel(d=d, nlist=nlist))
    if "IVFScalarQueryParallel" in name:
        indexes.append(parallel_ivf.IVFScalarQueryParallel(d=d, nlist=nlist))
    if "IVFScalarCandidateParallel" in name:
        indexes.append(parallel_ivf.IVFScalarCandidateParallel(d=d, nlist=nlist))
    return indexes

