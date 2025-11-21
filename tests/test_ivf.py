import numpy as np
import pytest
import sys
import os
import time
from faiss.contrib.datasets import SyntheticDataset
from utils import *

def _make_two_cluster_data():
    """Helper to make two clusters of data for test"""
    rng = np.random.default_rng(123)

    cluster_a = rng.normal(loc=(0.0, 0.0), scale=0.1, size=(5, 2))
    cluster_b = rng.normal(loc=(10.0, 0.0), scale=0.1, size=(5, 2))

    data = np.vstack([cluster_a, cluster_b]).astype(np.float32)
    return data, cluster_a, cluster_b


def assert_search_results_equal(
        I_real,
        I_test,
        rtol=1e-5,
        atol=1e-7,
        otol=1e-3,
    ):
        # Allow small tolerance in overlap rate
        overlap_rate = np.mean(np.array(I_real) == np.array(I_test))

        assert overlap_rate > 1 - otol, f"Overlap rate {overlap_rate:.6f} is not > {1-otol:.3f}. "


def test_ivf_two_cluster_search_and_add():
    """Small IVF testing"""
    data, cluster_a, cluster_b = _make_two_cluster_data()
    n, d = data.shape
    nlist = 2

    # Build IVF index
    for ivf in get_all_indexes(d, nlist):
        ivf.train(data)
        ivf.build(data)  # labels 0..9

        # Query near cluster A
        q_a = np.array([[0.5, 0.0]], dtype=np.float32)
        results_a = ivf.search(q_a, k=5, nprobe=1)  
        labels_a = results_a[0]
        
        assert all(0 <= lab < 5 for lab in labels_a), f"Cluster A query returned wrong labels: {labels_a}"

        q_a = np.array([[0.5, 0.0]], dtype=np.float32)
        results_a = ivf.search(q_a, k=5, nprobe=2)  
        labels_a = results_a[0]
        
        assert all(0 <= lab < 5 for lab in labels_a), f"Cluster A query returned wrong labels when n_probe=2: {labels_a}"

        #Query near cluster B
        q_b = np.array([[9.5, 0.5]], dtype=np.float32)
        results_b = ivf.search(q_b, k=5, nprobe=2)
        labels_b = results_b[0]
        
        assert all(5 <= lab < 10 for lab in labels_b), f"Cluster B query returned wrong labels: {labels_b}"

        # --- Add a new point very close to cluster A ---
        new_point = np.array([[0.01, -0.02]], dtype=np.float32)
        ivf.add(new_point)  # should get label = 10

        # Query again near cluster A
        results_a2 = ivf.search(q_a, k=6, nprobe=1)
        labels_a2 = results_a2[0]

        
        assert 10 in labels_a2, f"new point added to wrong centroid"
        assert all(0<= lab < 5 for lab in labels_a or lab == 10), f"Add issue"



def generate_six_gaussian_clusters_30d():
    """ Generate 6 distinct gaussian clusters"""
    rng = np.random.default_rng(123)

    means = [-20.0, -10.0, 0.0, 10.0, 20.0, 30.0]
    clusters = []

    for mu in means:
        # mean is the same in all 30 dimensions
        cluster = rng.normal(loc=mu, scale=0.5, size=(50, 30))
        clusters.append(cluster)

    data = np.vstack(clusters).astype(np.float32)
    return data, np.array(means, dtype=np.float32)

def _bruteforce_nearest_neighbors(data, queries, k):
    """
    Find best k L2 norm over all data
    """
    results = []
    for q in queries:
        dists = np.linalg.norm(data - q[None, :], axis=1)
        topk = np.argsort(dists)[:k]
        results.append(topk.tolist())
    return results


def test_ivf_matches_on_median_dataset():
    
    data, means = generate_six_gaussian_clusters_30d()
    n, d = data.shape

    nlist = 6
    k = 5

    means = [-20.0, -10.0, 0.0, 10.0, 20.0, 30.0]

    queries = []
    for mu in means:
        q = np.full((1, d), fill_value=mu, dtype=np.float32)
        queries.append(q)
    queries = np.vstack(queries).astype(np.float32)

    # Build IVF index. Shuffle data beforehand to make centroids less predictable
    np.random.shuffle(data)


    for ivf in get_all_indexes(d, nlist):
        ivf.train(data)
        ivf.build(data)

        queries = []
        for mu in means:
            q = np.full((1, d), fill_value=mu, dtype=np.float32)
            queries.append(q)
        queries = np.vstack(queries).astype(np.float32)

        ivf_results = ivf.search(queries, k=k, nprobe=nlist)

        bf_results = _bruteforce_nearest_neighbors(data, queries, k=k)

        # Compare IVF vs bruteforce for each query
        for qi, (ivf_labels, bf_labels) in enumerate(zip(ivf_results, bf_results)):
            assert ivf_labels == bf_labels, (
                f"Mismatch for query {qi}: IVF {ivf_labels} vs brute-force {bf_labels}"
            )

def test_ivf_synthetic_dataset():
    
    ds = SyntheticDataset(d=128, nb=10000, nq=10000, nt=1000)

    xq = ds.get_queries()
    xb = ds.get_database()
    xt = ds.get_train()

    nb, d = xb.shape
    nt, d = xt.shape
    nq, d = xq.shape

    nlist = 15

    bf_results = _bruteforce_nearest_neighbors(xb, xq, k=10)

    for ivf in get_all_indexes(d, nlist):
        ivf.train(xt)
        ivf.build(xb)

        ivf_results = ivf.search(xq, k=10, nprobe=10)

        # Compare IVF vs bruteforce for each query
        assert_search_results_equal(ivf_results, bf_results, otol=0.01)



if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
