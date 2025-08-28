import math
import os
import numpy as np
from examples_utils import TicToc, read_hdf5_data
from pdxearch.index_factory import IndexPDXIVF2SQ8
np.random.seed(42)

"""
Filtered Search
Download the .hdf5 data here: https://drive.google.com/drive/folders/1f76UCrU52N2wToGMFg9ir1MY8ZocrN34?usp=sharing
"""
if __name__ == "__main__":
    dataset_name = 'agnews-mxbai-1024-euclidean.hdf5'
    num_dimensions = 1024
    nprobe = 64
    knn = 100
    selectivity = 0.1 # From 0 to 1
    print(f'Running example: Filtered Search\n- D={num_dimensions}\n- k={knn}\n- nprobe={nprobe}\n- dataset={dataset_name}\n- selectivity={selectivity}')
    train, queries = read_hdf5_data(os.path.join('./benchmarks/datasets/downloaded', dataset_name))
    nbuckets = 4 * math.ceil(math.sqrt(len(train)))

    index = IndexPDXIVF2SQ8(ndim=num_dimensions, nbuckets=nbuckets, normalize=True)
    print('Preprocessing')
    index.preprocess(train)
    print('Training')
    training_points = nbuckets * 50
    rng = np.random.default_rng()
    training_sample_idxs = rng.choice(len(train), size=training_points, replace=False)
    training_sample_idxs.sort()
    index.train(train[training_sample_idxs])
    print('PDXifying')
    index.add(train)
    print(f'{len(queries)} queries with PDX')
    times = []
    clock = TicToc()
    results = []
    for i in range(len(queries)):
        q = np.ascontiguousarray(queries[i])
        # We choose random tuples at a certain level of selectivity
        passing_tuples = np.random.choice(np.arange(0, len(train)), size=(int(len(train) * selectivity)), replace=False)
        # We mock a predicate evaluation
        index.evaluate_predicate(passing_tuples)
        clock.tic()
        index.filtered_search(q, knn, nprobe=nprobe)
        times.append(clock.toc())
    print(f'PDX med. time at {selectivity} selectivity: {np.median(np.array(times))} ')
    # We check the filtering correctness by choosing 100 random tuples
    passing_tuples = np.random.choice(np.arange(0, len(train)), size=100, replace=False)
    index.evaluate_predicate(passing_tuples)
    results = index.filtered_search(np.ascontiguousarray(queries[0]), knn, nprobe=nbuckets)
    # The same 100 chosen tuples should be returned by PDX
    print('Got correct results?', len(set(passing_tuples).intersection(set(results[0]))) == 100)

