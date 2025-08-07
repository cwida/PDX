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
    # generate_test_data('word2vec-300')
    # generate_test_data('msong-420')
    # generate_test_data('instructorxl-arxiv-768')
    # generate_test_data('gist-960-euclidean')
    # generate_test_data('contriever-768')
    # generate_test_data('gooaq-distilroberta-768-normalized')
    # generate_test_data('agnews-mxbai-1024-euclidean')
    # generate_test_data('coco-nomic-768-normalized')
    # generate_test_data('simplewiki-openai-3072-normalized')

    generate_test_data('imagenet-align-640-normalized')

    generate_test_data('yandex-200-cosine')
    generate_test_data('imagenet-clip-512-normalized')
    generate_test_data('laion-clip-512-normalized')
    generate_test_data('codesearchnet-jina-768-cosine')

    generate_test_data('yi-128-ip')
    generate_test_data('landmark-dino-768-cosine')
    generate_test_data('landmark-nomic-768-normalized')
    generate_test_data('arxiv-nomic-768-normalized')

    generate_test_data('ccnews-nomic-768-normalized')
    generate_test_data('celeba-resnet-2048-cosine')
    generate_test_data('llama-128-ip')
    generate_test_data('yahoo-minilm-384-normalized')
