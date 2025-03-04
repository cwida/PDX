import math
import numpy as np
import sklearn.datasets
from sklearn.model_selection import train_test_split
from examples_utils import TicToc
from pdxearch.index_factory import IndexPDXADSamplingIVFFlat, IndexPDXBONDIVFFlat

"""
PDXearch (pruned search) + ADSampling with an IVF index (built with FAISS)
This example uses a random collection of vectors
"""
if __name__ == "__main__":
    num_dimensions = 768
    num_embeddings = 100_000
    num_query_embeddings = 1000
    knn = 10
    nprobe = 64
    print(f'Running example: PDXearch + ADSampling (IVFFlat)\n- D={num_dimensions}\n- k={knn}\n- nprobe={nprobe}\n- dataset=RANDOM')
    X, _ = sklearn.datasets.make_blobs(n_samples=num_embeddings, n_features=num_dimensions, centers=1000, random_state=1)
    X = X.astype(np.float32)
    data, queries = train_test_split(X, test_size=num_query_embeddings)

    nbuckets = 1 * math.ceil(math.sqrt(num_embeddings))
    index = IndexPDXADSamplingIVFFlat(ndim=num_dimensions, nbuckets=nbuckets)

    print('Preprocessing')
    index.preprocess(data)  # Preprocess vectors with ADSampling
    print('Training IVF')
    index.train(data)  # Train IVF with FAISS
    print('PDXifying')
    index.add_load(data)  # Add vectors and load PDX index in memory
    print(f'{len(queries)} queries with PDX')
    times = []
    clock = TicToc()
    results = []
    for i in range(num_query_embeddings):
        q = np.ascontiguousarray(queries[i])
        clock.tic()
        index.search(q, knn, nprobe=nprobe)
        times.append(clock.toc())
    print('PDX avg. time:', sum(times) / float(len(times)))
    # To check results...
    results = index.search(queries[0], knn)
    for result in results:
        print(result.index, result.distance)

    times = []
    clock = TicToc()
    results = []
    queries = index.preprocess(queries, inplace=False)
    index.core_index.index.nprobe = nprobe
    print(f'{len(queries)} queries with FAISS')
    for i in range(num_query_embeddings):
        q = np.ascontiguousarray(np.array([queries[i]]))
        clock.tic()
        index.core_index.index.search(q, k=knn)
        times.append(clock.toc())
    print('FAISS avg. time:', sum(times) / float(len(times)))
    # To check results...
    print(index.core_index.index.search(np.array([queries[0]]), k=knn))


