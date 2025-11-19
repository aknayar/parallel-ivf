#!/usr/bin/env python3
"""
Example usage of parallel_ivf Python bindings
"""

import numpy as np
import sys
import os

# Add build directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'build'))

import parallel_ivf

def main():
    print("=" * 60)
    print("Parallel IVF Example")
    print("=" * 60)
    
    # Parameters
    d = 128          # dimension
    nlist = 100      # number of clusters
    n_train = 10000  # number of training vectors
    n_add = 50000    # number of vectors to add
    n_query = 10     # number of queries
    k = 10           # number of nearest neighbors
    nprobe = 5       # number of clusters to probe
    
    print(f"\nParameters:")
    print(f"  Dimension: {d}")
    print(f"  Number of clusters: {nlist}")
    print(f"  Training vectors: {n_train}")
    print(f"  Vectors to index: {n_add}")
    print(f"  Query vectors: {n_query}")
    print(f"  Neighbors to find: {k}")
    print(f"  Clusters to probe: {nprobe}")
    
    # Create index
    print(f"\n1. Creating IVF index...")
    ivf = parallel_ivf.IVFBase(d=d, nlist=nlist)
    print(f"   Created: {ivf}")
    
    # Generate random training data
    print(f"\n2. Generating training data...")
    np.random.seed(42)
    train_data = np.random.randn(n_train, d).astype(np.float32)
    print(f"   Shape: {train_data.shape}, dtype: {train_data.dtype}")
    
    # Train the index
    print(f"\n3. Training index...")
    ivf.train(train_data)
    print(f"   Training complete!")
    print(f"   Centroids stored: {len(ivf.centroids)}")
    
    # Generate vectors to add
    print(f"\n4. Generating vectors to add...")
    add_data = np.random.randn(n_add, d).astype(np.float32)
    print(f"   Shape: {add_data.shape}")
    
    # Add vectors to index
    print(f"\n5. Adding vectors to index...")
    ivf.add(add_data)
    print(f"   Added {n_add} vectors!")
    print(f"   Number of inverted lists: {len(ivf.inv_lists)}")
    
    # Generate query vectors
    print(f"\n6. Generating query vectors...")
    queries = np.random.randn(n_query, d).astype(np.float32)
    print(f"   Shape: {queries.shape}")
    
    # Search
    print(f"\n7. Searching for nearest neighbors...")
    ivf.search(queries, k=k, nprobe=nprobe)
    print(f"   Search complete!")
    
    # Access attributes
    print(f"\n8. Index attributes:")
    print(f"   d: {ivf.d}")
    print(f"   nlist: {ivf.nlist}")
    print(f"   nprobe: {ivf.nprobe}")
    print(f"   Number of centroids: {len(ivf.centroids)}")
    print(f"   Number of inverted lists: {len(ivf.inv_lists)}")
    
    # Show inverted list sizes
    if ivf.inv_lists:
        list_sizes = [len(lst) for lst in ivf.inv_lists]
        print(f"   Inverted list sizes (min/mean/max): "
              f"{min(list_sizes)}/{np.mean(list_sizes):.1f}/{max(list_sizes)}")
    
    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)

if __name__ == "__main__":
    main()

