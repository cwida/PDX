import math
import os
import faiss
import numpy as np
from examples_utils import TicToc, read_hdf5_data
from pdxearch.index_factory import IndexPDXBONDIVFFlat

"""
PDXearch (pruned search) + BOND with an IVF index (built with FAISS)
We can do exact-search by exploring all the buckets. This lets the pruning strategy shine.
Download the .hdf5 data here: https://drive.google.com/drive/folders/1f76UCrU52N2wToGMFg9ir1MY8ZocrN34?usp=sharing
"""
if __name__ == "__main__":
    dataset_name = 'msong-420.hdf5'
    num_dimensions = 420
    knn = 10
    print(f'Running example: PDXearch + BOND (Exhaustive with IVFFlat)\n- D={num_dimensions}\n- k={knn}\n- nprobe=ALL\n- dataset={dataset_name}')
    train, queries = read_hdf5_data(os.path.join('./benchmarks/datasets/downloaded', dataset_name))
    nbuckets = 2 * math.ceil(math.sqrt(len(train)))

    index = IndexPDXBONDIVFFlat(ndim=num_dimensions, nbuckets=nbuckets)
    print('Training')
    training_points = nbuckets * 50
    rng = np.random.default_rng()
    training_sample_idxs = rng.choice(len(train), size=training_points, replace=False)
    training_sample_idxs.sort()
    index.train(train[training_sample_idxs])
    print('PDXifying')
    index.add_load(train)
    print(f'{len(queries)} queries with PDX')
    times = []
    clock = TicToc()
    results = []
    for i in range(len(queries)):
        q = np.ascontiguousarray(queries[i])
        clock.tic()
        index.search(q, knn, nprobe=0)
        times.append(clock.toc())
    print('PDX avg. time:', sum(times) / float(len(times)))
    # To check results...
    # results = index.search(queries[0], knn)
    # for result in results:
    #     print(result.index, result.distance)

    print(f'{len(queries)} queries with FAISS')
    times = []
    clock = TicToc()
    results = []
    faiss_index = faiss.IndexFlatL2(num_dimensions)
    faiss_index.add(train)
    for i in range(len(queries)):
        q = np.ascontiguousarray(np.array([queries[i]]))
        clock.tic()
        faiss_index.search(q, k=knn)
        times.append(clock.toc())
    print('FAISS avg. time:', sum(times) / float(len(times)))
    # To check results...
    # print(faiss_index.search(np.array([queries[0]]), k=knn))


