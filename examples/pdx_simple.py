import numpy as np
import sklearn.datasets
from sklearn.model_selection import train_test_split
from examples_utils import TicToc
from pdxearch import IndexPDXIVF

"""
PDXearch with a single-level IVF index on random data.
This example uses a random collection of vectors generated with sklearn.
"""
if __name__ == "__main__":
    num_dimensions = 768
    num_embeddings = 100_000
    num_query_embeddings = 100
    knn = 100
    nprobe = 64
    print(f'Running example: PDXearch IVF (F32) on random data\n- D={num_dimensions}\n- k={knn}\n- nprobe={nprobe}\n- dataset=RANDOM')
    X, _ = sklearn.datasets.make_blobs(n_samples=num_embeddings, n_features=num_dimensions, centers=1000, random_state=1)
    X = X.astype(np.float32)
    data, queries = train_test_split(X, test_size=num_query_embeddings)

    index = IndexPDXIVF(num_dimensions=num_dimensions, normalize=True)
    print('Building index...')
    index.build(data)
    print(f'Index built: {index.num_clusters} clusters')
    print(f'Index in-memory size: {index.in_memory_size_bytes / (1024 * 1024):.2f} MB')

    print(f'{len(queries)} queries with PDX')
    times = []
    clock = TicToc()
    for i in range(num_query_embeddings):
        q = np.ascontiguousarray(queries[i])
        clock.tic()
        index.search(q, knn, nprobe=nprobe)
        times.append(clock.toc())
    print('PDX med. time:', np.median(np.array(times)))

    # Show results of first query
    ids, dists = index.search(np.ascontiguousarray(queries[0]), knn, nprobe=nprobe)
    print('First query results (ids):', ids[:10])
    print('First query results (dists):', dists[:10])
