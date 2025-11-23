import numpy as np
import pytest
import sys
import os
import time
from faiss.contrib.datasets import SyntheticDataset
from utils import *
import parallel_ivf
from test_kmeans import *
from test_ivf import *
import argparse

def correctness(name):
    indexes = []
    indexes = getIndex(name, 2,3)
    try:
        test_kmeans_vs_sklearn_2d(indexes)
    except:
        print("Basic Test K Means Failed Correctness")
    
    try:
        indexes = getIndex(name, 128,15)
        test_kmeans_vs_sklearn_128d(indexes)
    except:
        print("High Dimension Test K Means Failed Correctness")

    try:
        indexes = getIndex(name, 3,5)
        test_kmeans_unique_centroids(indexes)
    except:
        print("Test K Means Unique Centroids Failed")

    print("K Means Correctness Tests Passed")

    try:
        indexes = getIndex(name, 2,2)
        test_ivf_two_cluster_search_and_add(indexes)
    except:
        print("Basic IVF Search and Add Failed")
    
    try:
        indexes = getIndex(name, 30,6)
        test_ivf_matches_on_median_dataset(indexes)
    except:
        print("Larger IVF Search and Add Failed")
    
    print("IVF Tests Passed")

    print(f"All Correctness Tests Passed for {name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                    prog='Correctness')
    parser.add_argument('-i','--index',required=True)
    args = vars(parser.parse_args())
    name = args["index"]
    correctness(name)

