import numpy as np
import pytest
import sys
import os

# Make sure we can import the built extension
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../build"))
import parallel_ivf

def _make_two_cluster_data():
    """Helper to make two clusters of data for test"""
    rng = np.random.default_rng(123)

    cluster_a = rng.normal(loc=(0.0, 0.0), scale=0.1, size=(5, 2))
    cluster_b = rng.normal(loc=(10.0, 0.0), scale=0.1, size=(5, 2))

    data = np.vstack([cluster_a, cluster_b]).astype(np.float32)
    return data, cluster_a, cluster_b


def test_ivf_two_cluster_search_and_add():
    """Small IVF testing"""
    data, cluster_a, cluster_b = _make_two_cluster_data()
    n, d = data.shape
    nlist = 2

    # Build IVF index
    ivf = parallel_ivf.IVFBase(d=d, nlist=nlist)
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

    ivf = parallel_ivf.IVFBase(d=d, nlist=nlist)
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
    
    ivf_results = ivf.search(queries, k=k, nprobe=2)

    # Compare IVF vs bruteforce for each query
    for qi, (ivf_labels, bf_labels) in enumerate(zip(ivf_results, bf_results)):
        assert ivf_labels == bf_labels, (
            f"Mismatch for query {qi}: IVF {ivf_labels} vs brute-force {bf_labels}"
        )



if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
