"""Simple pytest comparing our k-means to sklearn"""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../build'))
import parallel_ivf

from sklearn.cluster import KMeans as SklearnKMeans

EPSILON = 1e-4

def generate_gaussian_clusters(n_clusters=3, n_samples_per_cluster=100, dim=2, seed=42):
    """Generate simple Gaussian clusters"""
    np.random.seed(seed)
    
    # Generate cluster centers
    centers = np.random.randn(n_clusters, dim) * 5
    
    # Generate points around each center
    data = []
    for center in centers:
        cluster_data = center + np.random.randn(n_samples_per_cluster, dim)
        data.append(cluster_data)
    
    data = np.vstack(data).astype(np.float32)
    return data, centers


def test_kmeans_vs_sklearn_2d():
    """Compare our k-means to sklearn on 2D Gaussian clusters"""
    # Generate simple 2D data
    n_clusters = 3
    data, true_centers = generate_gaussian_clusters(
        n_clusters=n_clusters, 
        n_samples_per_cluster=100, 
        dim=2
    )
    n_samples, d = data.shape
    
    # Our k-means
    ivf = parallel_ivf.IVFBase(d=d, nlist=n_clusters)
    ivf.train(data)
    our_centroids = np.array(ivf.centroids).reshape(n_clusters, d)
    
    # Sklearn k-means
    sklearn_kmeans = SklearnKMeans(n_clusters=n_clusters, random_state=42, n_init=1)
    sklearn_kmeans.fit(data)
    sklearn_centroids = sklearn_kmeans.cluster_centers_
    
    # Compare closest centroids
    max_centroid_distance = 0
    for our_c in our_centroids:
        distances = [np.linalg.norm(our_c - sk_c) for sk_c in sklearn_centroids]
        min_dist = min(distances)
        max_centroid_distance = max(max_centroid_distance, min_dist)
    
    assert max_centroid_distance < EPSILON, f"Centroids too different: max dist {max_centroid_distance:.2f}"


def test_kmeans_vs_sklearn_128d():
    """Compare our k-means to sklearn on 128D Gaussian clusters"""
    # Generate 128D data
    n_clusters = 15
    data, true_centers = generate_gaussian_clusters(
        n_clusters=n_clusters, 
        n_samples_per_cluster=100, 
        dim=128
    )
    n_samples, d = data.shape
    
    # Our k-means
    ivf = parallel_ivf.IVFBase(d=d, nlist=n_clusters)
    ivf.train(data)
    our_centroids = np.array(ivf.centroids).reshape(n_clusters, d)
    
    # Sklearn k-means
    sklearn_kmeans = SklearnKMeans(n_clusters=n_clusters, random_state=42, n_init=1)
    sklearn_kmeans.fit(data)
    sklearn_centroids = sklearn_kmeans.cluster_centers_
    
    # Compare closest centroids
    centroid_distances = []
    for our_c in our_centroids:
        distances = [np.linalg.norm(our_c - sk_c) for sk_c in sklearn_centroids]
        centroid_distances.append(min(distances))
    
    max_centroid_distance = np.max(centroid_distances)

    assert max_centroid_distance < EPSILON, f"Max centroid distance {max_centroid_distance:.2f} too large"


def test_kmeans_unique_centroids():
    """Ensure all centroids are unique"""
    data, _ = generate_gaussian_clusters(n_clusters=5, n_samples_per_cluster=50, dim=3)
    n_clusters = 5
    d = 3
    
    ivf = parallel_ivf.IVFBase(d=d, nlist=n_clusters)
    ivf.train(data)
    centroids = np.array(ivf.centroids).reshape(n_clusters, d)
    
    # Check all centroids are different
    for i in range(n_clusters):
        for j in range(i + 1, n_clusters):
            dist = np.linalg.norm(centroids[i] - centroids[j])
            assert dist > EPSILON, f"Centroids {i} and {j} are too close: {dist}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
