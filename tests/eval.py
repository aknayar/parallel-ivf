from threadpoolctl import threadpool_limits
import os
import time
from faiss.contrib.datasets import SyntheticDataset
from utils import *
import time
import argparse

from faiss.contrib.datasets import SyntheticDataset


def test(index, nq, xb, xt, xq, k, n_probe, n_threads):
    train_time, build_time, query_time = 0.0, 0.0, 0.0

    NUM_ITERS = 1
    os.environ["OMP_NUM_THREADS"] = str(n_threads)

    for _ in range(NUM_ITERS):
        t0 = time.time()
        index.train(xt)
        t1 = time.time()
        index.build(xb)
        t2 = time.time()
        index.search(xq, k=k, nprobe=n_probe)
        t3 = time.time()
        train_time += t1-t0
        build_time += t2-t1
        query_time += t3-t2
    
    train_time /= NUM_ITERS
    query_time /= NUM_ITERS
    build_time /= NUM_ITERS
    qps = nq / query_time

    return train_time, build_time, query_time, qps


TEST_PARAMS = {
    "easy": {
        "nq": 100,
        "nb": 10000,
        "nt": 1000,
        "d": 128,
        "nlist": 10,
        "k": 10,
        "n_probe": 5
    },
    "medium": {
        "nq": 100,
        "nb": 10000,
        "nt": 1000,
        "d": 512,
        "nlist": 10,
        "k": 10,
        "n_probe": 5
    },
    "hard": {
        "nq": 100,
        "nb": 100000,
        "nt": 1000,
        "d": 1024,
        "nlist": 50,
        "k": 10,
        "n_probe": 25
    },
    "extreme": {
        "nq": 100,
        "nb": 1000000,
        "nt": 10000,
        "d": 1024,
        "nlist": 50,
        "k": 10,
        "n_probe": 25
    },
    "gist": {
        "nq": 100,
        "nb": 1000000,
        "nt": 10000,
        "d": 960,
        "nlist": 50,
        "k": 10,
        "n_probe": 5
    }
}


if __name__ == "__main__":
    # args: --dataset (easy/medium/hard/extreme/gist)
    parser = argparse.ArgumentParser(
                    prog='Correctness')
    parser.add_argument('-d','--dataset',required=True)
    args = vars(parser.parse_args())
    dataset = args["dataset"]

    args = vars(parser.parse_args())

    indexes = [
        # "IVFBase",
        # "IVFSIMD",
        # "IVFCache",
        # "IVFCacheSIMD",
        # "IVFCacheV2",
        # "IVFCacheV2SIMD",
        "IVFSIMDQueryParallel",
        "IVFSIMDCandidateParallel",
        "IVFCacheQueryParallel",
        "IVFCacheSIMDQueryParallel",
        "IVFCacheV2QueryParallel",
        "IVFCacheV2SIMDQueryParallel",
        "IVFCacheCandidateParallel",
        "IVFCacheSIMDCandidateParallel",
        "IVFScalarQueryParallel",
        "IVFScalarCandidateParallel"
    ]

    test_params = TEST_PARAMS[dataset]
    if dataset == "gist":
        xb, xt, xq  = load_fvecs_data(f"data/gist1M/gist_base.fvecs", f"data/gist1M/gist_learn.fvecs", f"data/gist1M/gist_query.fvecs", test_params["nb"], test_params["nt"], test_params["nq"])
    else:
        ds = SyntheticDataset(test_params["d"], test_params["nt"], test_params["nb"], test_params["nq"], seed=1337)
        xt, xb, xq = ds.get_train(), ds.get_database(), ds.get_queries()

    for index_name in indexes:
        print(f"----------------------")
        index = getIndex(index_name, test_params["d"], test_params["nlist"])[0]

        os.makedirs(f"results/{dataset}", exist_ok=True)

        # iterate through threads
        max_threads = os.cpu_count()
        threads = [1]
        while threads[-1] * 2 <= max_threads:
            threads.append(2 * threads[-1])
        with open(f"results/{dataset}/{dataset}_{index_name}.csv", "w") as f:
            f.write(f"n_threads,train_time,build_time,query_time,qps\n")
        for n_threads in threads:
            print(f"Testing {index_name} with {n_threads} threads...")
            train_time, build_time, query_time, qps = 0, 0, 0, 0
            with threadpool_limits(limits=n_threads):
                train_time, build_time, query_time, qps = test(index, test_params["nq"], xb, xt, xq, test_params["k"], test_params["n_probe"], n_threads)
                with open(f"results/{dataset}/{dataset}_{index_name}.csv", "a") as f:
                    f.write(f"{n_threads},{train_time},{build_time},{query_time},{qps}\n")
            
        