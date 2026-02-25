import os
import faiss
import numpy as np
from examples_utils import TicToc, read_hdf5_data
from pdxearch import IndexPDXIVF

"""
PDXearch with exhaustive search (nprobe=0 visits all clusters).
This lets the pruning strategy shine against brute-force FAISS.
Download the .hdf5 data here: https://drive.google.com/drive/folders/1f76UCrU52N2wToGMFg9ir1MY8ZocrN34?usp=sharing
"""
if __name__ == "__main__":
    dataset_name = 'openai-1536-angular.hdf5'
    num_dimensions = 1536
    knn = 100
    print(f'Running example: PDXearch Exhaustive Search\n- D={num_dimensions}\n- k={knn}\n- nprobe=ALL\n- dataset={dataset_name}')
    train, queries = read_hdf5_data(os.path.join('./benchmarks/datasets/downloaded', dataset_name))

    index = IndexPDXIVF(num_dimensions=num_dimensions, distance_metric="cosine")
    print('Building index...')
    index.build(train)
    print(f'Index built: {index.num_clusters} clusters')

    queries = queries[:100]
    print(f'{len(queries)} queries with PDX (exhaustive)')
    times = []
    clock = TicToc()
    for i in range(len(queries)):
        q = np.ascontiguousarray(queries[i])
        clock.tic()
        index.search(q, knn, nprobe=0)  # nprobe=0 searches all clusters
        times.append(clock.toc())
    print('PDX med. time:', np.median(np.array(times)))

    # Show results of first query
    ids, dists = index.search(np.ascontiguousarray(queries[0]), knn, nprobe=0)
    print('First query results (ids):', ids[:10])
    print('First query results (dists):', dists[:10])

    print(f'{len(queries)} queries with FAISS (brute force)')
    times = []
    clock = TicToc()
    faiss_index = faiss.IndexFlatL2(num_dimensions)
    faiss_index.add(train)
    for i in range(len(queries)):
        q = np.ascontiguousarray(np.array([queries[i]]))
        clock.tic()
        faiss_index.search(q, k=knn)
        times.append(clock.toc())
    print('FAISS med. time:', np.median(np.array(times)))
