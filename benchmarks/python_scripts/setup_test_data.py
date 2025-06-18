import sys
from setup_utils import *
from setup_settings import *


# Transforms the queries from HDF5 format to .fvecs
def generate_test_data(dataset):
    if not os.path.exists(GROUND_TRUTH_DATA):
        os.makedirs(GROUND_TRUTH_DATA)
    test = read_hdf5_test_data(dataset)
    N_QUERIES = len(test)
    with open(os.path.join(QUERIES_DATA, dataset), "wb") as file:
        file.write(N_QUERIES.to_bytes(4, sys.byteorder, signed=False))
        file.write(test.tobytes("C"))


if __name__ == "__main__":
    # generate_test_data('fashion-mnist-784-euclidean')
    # generate_test_data('word2vec-300')
    generate_test_data('msong-420')
    generate_test_data('instructorxl-arxiv-768')
    generate_test_data('gist-960-euclidean')
    generate_test_data('contriever-768')
