import numpy as np
import pytest
import sys
import os
import time
from faiss.contrib.datasets import SyntheticDataset
from utils import *
from sklearn.cluster import KMeans as SklearnKMeans
import time
from test_kmeans import generate_gaussian_clusters
import argparse

def _gen_data_and_queries(clusters, samples_per, queries, dim):
    data, _ = generate_gaussian_clusters(
        n_clusters=clusters, 
        n_samples_per_cluster=samples_per, 
        dim=dim
    )
    queries, _ = generate_gaussian_clusters(
        n_clusters=1, 
        n_samples_per_cluster=queries, 
        dim=dim
    )
    return data, queries


def _test_loop(base, test, data, queries, k, n_probe, extreme=False):
    serialTime, serialTrainTime, serialComputeTime = 0.0, 0.0, 0.0
    optimTime, optimTrainTime, optimComputeTime = 0.0, 0.0, 0.0

    NUM_ITERS = 3
    if extreme:
        NUM_ITERS=1

    for _ in range(NUM_ITERS):
        t0 = time.time()
        base.train(data)
        base.build(data)
        t1 = time.time()
        
        base.search(queries, k=k, nprobe=n_probe)
        t2 = time.time()
        serialTrainTime += t1-t0
        serialComputeTime += t2-t1
        serialTime += t2-t0
    
    serialTime /= NUM_ITERS
    serialComputeTime /= NUM_ITERS
    serialTrainTime /= NUM_ITERS
    # if extreme:
    #     print(serialTime, serialComputeTime, serialTrainTime)

    for _ in range(NUM_ITERS):
        t0 = time.time()
        test.train(data)
        test.build(data)
        t1 = time.time()
        
        test.search(queries, k=k, nprobe=n_probe)
        t2 = time.time()
        optimTrainTime += t1-t0
        optimComputeTime += t2-t1
        optimTime += t2-t0
    
    optimTime /= NUM_ITERS
    optimComputeTime /= NUM_ITERS
    optimTrainTime /= NUM_ITERS

    return serialTime, serialTrainTime, serialComputeTime, optimTime, optimTrainTime, optimComputeTime



def easy_test(name):

    data, queries = _gen_data_and_queries(5, 40, 30, 2)

    d = 2
    nlist = 5

    base = getIndex("IVFBase",d, nlist)[0]
    test = getIndex(name, d, nlist)[0]

    return _test_loop(base,test,data,queries,5,3)

def medium_test(name):
    d = 40
    nlist = 10

    data, queries = _gen_data_and_queries(nlist, 200, 300, d)

    base = getIndex("IVFBase",d, nlist)[0]
    test = getIndex(name, d, nlist)[0]

    return _test_loop(base,test,data,queries,10,5)

def hard_test(name):
    d = 128
    nlist = 30

    data, queries = _gen_data_and_queries(nlist, 150, 3000, d)

    base = getIndex("IVFBase",d, nlist)[0]
    test = getIndex(name, d, nlist)[0]

    return _test_loop(base,test,data,queries,20,10)

def extreme_test(name):
    d = 256
    nlist = 70

    data, queries = _gen_data_and_queries(nlist, 350, 2000, d)

    base = getIndex("IVFBase",d, nlist)[0]
    test = getIndex(name, d, nlist)[0]

    return _test_loop(base,test,data,queries,20,15, True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                    prog='Correctness')
    parser.add_argument('-i','--index',required=True)
    args = vars(parser.parse_args())
    name = args["index"]
    
    tests = [easy_test, medium_test, hard_test, extreme_test]
    test_names = ["Easy", "Medium", "Hard", "Extreme"]

    for i, test in enumerate(tests):
        serialTime, serialTrainTime, serialComputeTime, optimTime, optimTrainTime, optimComputeTime = test(name)
        speedup = serialTime/optimTime
        trainSpeedup = serialTrainTime/optimTrainTime
        compSpeedup = serialComputeTime/optimComputeTime
        print(f"Template: {name} TEST: {test_names[i]} \n SPEEDUP: {speedup :.4f}"
        f" TRAIN SPEEDUP: {trainSpeedup :.4f} SEARCH/ADD SPEEDUP: {compSpeedup :.4f}")
        print("-"*150)
    

    

    
    




    


