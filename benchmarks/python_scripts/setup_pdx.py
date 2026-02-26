import json
import sys
from setup_utils import *
from benchmark_utils import TicToc
from pdxearch import IndexPDXIVF, IndexPDXIVFSQ8, IndexPDXIVFTree, IndexPDXIVFTreeSQ8
from WrapperBruteForce import BruteForceFAISS
from sklearn import preprocessing

INDEX_CLASSES = {
    "pdx_f32": IndexPDXIVF,
    "pdx_u8": IndexPDXIVFSQ8,
    "pdx_tree_f32": IndexPDXIVFTree,
    "pdx_tree_u8": IndexPDXIVFTreeSQ8,
}


def generate_ground_truth(dataset_abbrev, KNNS=(100,), normalize=True):
    """Generate ground truth with FAISS brute-force search."""
    hdf5_name, dims = DATASET_INFO[dataset_abbrev]
    print(f'Generating ground truth: {dataset_abbrev} -> {hdf5_name}')

    train, test = read_hdf5_data(hdf5_name)
    N_QUERIES = len(test)
    print('N. Queries', N_QUERIES)

    if normalize:
        train = preprocessing.normalize(train, axis=1, norm='l2')
        test = preprocessing.normalize(test, axis=1, norm='l2')

    algo = BruteForceFAISS("euclidean")
    algo.fit(train)
    for knn in KNNS:
        gt_filename = get_ground_truth_filename(hdf5_name, knn, normalize)
        gt_name = os.path.join(SEMANTIC_GROUND_TRUTH_PATH, gt_filename)
        gt = {}
        index_data = []
        distance_data = []
        print('Querying for GT...')
        dist, index = algo.query_batch(test, n=knn)
        for i in range(N_QUERIES):
            index_data.append(index[i])
            distance_data.append(dist[i])
            gt[i] = index[i].tolist()
        with open(os.path.join(GROUND_TRUTH_DATA, gt_filename.replace('.json', '')), "wb") as file:
            file.write(np.array(index_data, dtype=np.uint32).tobytes("C"))
            file.write(np.array(distance_data, dtype=np.float32).tobytes("C"))
        with open(gt_name, 'w') as f:
            json.dump(gt, f)


def generate_test_data(dataset_abbrev):
    """Save queries from HDF5 to binary format for C++ benchmarks."""
    hdf5_name, dims = DATASET_INFO[dataset_abbrev]
    print(f'Saving test data: {dataset_abbrev} -> {hdf5_name}')

    test = read_hdf5_test_data(hdf5_name)
    N_QUERIES = len(test)
    with open(os.path.join(QUERIES_DATA, hdf5_name), "wb") as file:
        file.write(N_QUERIES.to_bytes(4, sys.byteorder, signed=False))
        file.write(test.tobytes("C"))


def generate_index(dataset_abbrev: str, index_type: str, normalize=True, seed=42):
    """Build and save a PDX index."""
    hdf5_name, dims = DATASET_INFO[dataset_abbrev]
    print(f'{dataset_abbrev} -> {hdf5_name} ({index_type})')
    data = read_hdf5_train_data(hdf5_name)

    cls = INDEX_CLASSES[index_type]
    index = cls(num_dimensions=dims, normalize=normalize, seed=seed)
    clock = TicToc()
    print('Building index...')
    clock.tic()
    index.build(data)
    print(f'Index built: {index.num_clusters} clusters ({clock.toc():.2f}ms)')

    save_path = os.path.join(PDX_DATA, dataset_abbrev + '-' + index_type)
    print(f'Saving to {save_path}')
    index.save(save_path)
    print('Done.')


if __name__ == "__main__":
    # generate_ground_truth('mxbai', normalize=True)
    # generate_test_data('mxbai')
    # generate_index('mxbai', 'pdx_f32', normalize=True)
    pass
