import os
import numpy as np
from examples_utils import TicToc, read_hdf5_data
from pdxearch import IndexPDXIVFTreeSQ8, load_index

"""
Example to save a PDX index to a file and reload it later.
Download the .hdf5 data here: https://drive.google.com/drive/folders/1f76UCrU52N2wToGMFg9ir1MY8ZocrN34?usp=sharing
"""
if __name__ == "__main__":
    dataset_name = 'agnews-mxbai-1024-euclidean.hdf5'
    num_dimensions = 1024
    nprobe = 25
    knn = 20
    print(f'Running example: Persist and Load PDX Index\n- D={num_dimensions}\n- k={knn}\n- nprobe={nprobe}\n- dataset={dataset_name}')

    train, queries = read_hdf5_data(os.path.join('./benchmarks/datasets/downloaded', dataset_name))

    index = IndexPDXIVFTreeSQ8(num_dimensions=num_dimensions, normalize=True)
    print('Building index...')
    index.build(train)
    print(f'Index built: {index.num_clusters} clusters')

    index_path = './examples/my_idx.pdx'
    print(f'Saving index to {index_path}')
    index.save(index_path)

    print('Loading index from file...')
    del index
    restored_index = load_index(index_path)

    print(f'{len(queries)} queries with restored PDX index')
    times = []
    clock = TicToc()
    restored_index.set_nprobe(nprobe)
    for i in range(len(queries)):
        q = np.ascontiguousarray(queries[i])
        clock.tic()
        restored_index.search(q, knn)
        times.append(clock.toc())
    print('PDX med. time:', np.median(np.array(times)))

    ids, dists = restored_index.search(np.ascontiguousarray(queries[0]), knn)
    print('First query results (ids):', ids[:10])
    print('First query results (dists):', dists[:10])

    # Cleanup
    os.remove(index_path)
