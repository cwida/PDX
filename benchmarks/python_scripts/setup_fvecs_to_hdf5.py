from setup_utils import *

def generate_hdf5_file(path_data: str, path_query, dataset_name):
    data = read_fvecs(path_data)
    queries = read_fvecs(path_query)
    print('Queries', len(queries))
    print('Vectors', len(data))
    print('D=', len(data[0]))
    with h5py.File(dataset_name, 'w') as f:
        f.create_dataset("train", data=data)
        f.create_dataset("test", data=queries)


if __name__ == "__main__":
    generate_hdf5_file(
        './benchmarks/datasets/downloaded/word2vec_base.fvecs',
        './benchmarks/datasets/downloaded/word2vec_query.fvecs',
        './benchmarks/datasets/downloaded/word2vec-300.hdf5'
    )