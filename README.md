# Making RAG Faster
*Parallelizing inverted file indexes for efficient retrieval and addition to vector databases* 

## Collaborators  
- Drew Byrapatna  
- Akash Nayar   

## Course
15-618 Fall 2025

---

##  Information
This repository contains various implementations of a parallelized inverted file index, a database index used to efficiently retrieve documents from a vector database similar to a given query. Our IVF implementation has the ability to train and build an IVF for an arbitrary set of vector data, retrieve vector embeddings similar to a provided query, and update the database with new documents.

## Getting Started
### 1. Clone the Repository  
```bash
git clone https://github.com/aknayar/parallel-ivf.git
cd parallel-ivf
```

### 2. Build the Executable
```bash
./build.sh
```

### 3. Configure Python environment
```bash
python -m venv ivfvenv
source ivfvenv/bin/activate
pip install -r requirements.txt
```

### 4. Run the testing driver
```bash
./driver.sh -i $INDEX_NAME
```

A full set of currently implemented index types is listed below. You can learn about how to use the driver by calling 
```bash
./driver.sh -h
```

### Currently Implemented Indexes
- IVFBase (Serial Reference Implementation)
- IVFSIMD
- IVFCache
- IVFCacheSIMD
- IVFScalarQueryParallel
- IVFSIMDQueryParallel
- IVFCacheQueryParallel
- IVFCacheSIMDQueryParallel

Please refer to our report for information on each of these index types.
