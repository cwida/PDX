import os
SOURCE_DIR = os.getcwd()

"""
Creates the directories needed to run benchmarks
"""

# `benchmarks` directory should already exist
DATA_DIRECTORY = os.path.join("benchmarks", "datasets")
if not os.path.exists(os.path.join(SOURCE_DIR, DATA_DIRECTORY)):
    os.makedirs(os.path.join(SOURCE_DIR, DATA_DIRECTORY))

RAW_DATA = os.path.join(SOURCE_DIR, DATA_DIRECTORY, "downloaded")
GROUND_TRUTH_DATA = os.path.join(SOURCE_DIR, DATA_DIRECTORY, "ground_truth")
SEMANTIC_GROUND_TRUTH_PATH = os.path.join(SOURCE_DIR, "benchmarks", "gt")

CORE_INDEXES = os.path.join(SOURCE_DIR, "benchmarks", "core_indexes")
CORE_INDEXES_FAISS = os.path.join(CORE_INDEXES, "faiss")
CORE_INDEXES_FAISS_U8 = os.path.join(CORE_INDEXES, "faiss_sq8")
CORE_INDEXES_FAISS_L0 = os.path.join(CORE_INDEXES, "faiss_l0")
CORE_INDEXES_LORANN = os.path.join(CORE_INDEXES, "lorann")

PURESCAN_DATA = os.path.join(SOURCE_DIR, DATA_DIRECTORY, "purescan")

QUERIES_DATA = os.path.join(SOURCE_DIR, DATA_DIRECTORY, "queries")

NARY_DATA = os.path.join(SOURCE_DIR, DATA_DIRECTORY, "nary")
NARY_ADSAMPLING_DATA = os.path.join(SOURCE_DIR, DATA_DIRECTORY, "adsampling_nary")

PDX_DATA = os.path.join(SOURCE_DIR, DATA_DIRECTORY, "pdx")
PDX_ADSAMPLING_DATA = os.path.join(SOURCE_DIR, DATA_DIRECTORY, "adsampling_pdx")

directories = [
    RAW_DATA, GROUND_TRUTH_DATA, PURESCAN_DATA,
    NARY_DATA, NARY_ADSAMPLING_DATA,
    PDX_DATA, PDX_ADSAMPLING_DATA,
    CORE_INDEXES, CORE_INDEXES_FAISS, CORE_INDEXES_FAISS_U8,
    CORE_INDEXES_FAISS_L0, CORE_INDEXES_LORANN, QUERIES_DATA,
    SEMANTIC_GROUND_TRUTH_PATH
]

for needed_directory in directories:
    if not os.path.exists(needed_directory):
        os.makedirs(needed_directory)

# Datasets to set up and use
DATASETS = [
    'nytimes-16-angular',
    'fashion-mnist-784-euclidean',
    'glove-50-angular',
    'glove-200-angular',
    'sift-128-euclidean',
    'msong-420',
    'contriever-768',
    'gist-960-euclidean',
    'deep-image-96-angular',
    'instructorxl-arxiv-768',
    'openai-1536-angular',
    'word2vec-300',
    'gooaq-distilroberta-768-normalized',
    'agnews-mxbai-1024-euclidean',
    'coco-nomic-768-normalized',
    'simplewiki-openai-3072-normalized',
    'imagenet-align-640-normalized',
    'yandex-200-cosine',
    'imagenet-clip-512-normalized',
    'laion-clip-512-normalized',
    'codesearchnet-jina-768-cosine',
    'yi-128-ip',
    'landmark-dino-768-cosine',
    'landmark-nomic-768-normalized',
    'arxiv-nomic-768-normalized',
    'ccnews-nomic-768-normalized',
    'celeba-resnet-2048-cosine',
    'llama-128-ip',
    'yahoo-minilm-384-normalized'
]

DIMENSIONALITIES = {
    'random-xs-20-angular': 20,
    'random-s-100-euclidean': 100,
    'har-561': 561,
    'nytimes-16-angular': 16,
    'nytimes-256-angular': 256,
    'mnist-784-euclidean': 784,
    'fashion-mnist-784-euclidean': 784,
    'glove-25-angular': 25,
    'glove-50-angular': 50,
    'glove-100-angular': 100,
    'glove-200-angular': 200,
    'sift-128-euclidean': 128,
    'trevi-4096': 4096,
    'msong-420': 420,
    'contriever-768': 768,
    'stl-9216': 9216,
    'gist-960-euclidean': 960,
    'deep-image-96-angular': 96,
    'instructorxl-arxiv-768': 768,
    'openai-1536-angular': 1536,
    'word2vec-300': 300,
    'gooaq-distilroberta-768-normalized': 768,
    'agnews-mxbai-1024-euclidean': 1024,
    'coco-nomic-768-normalized': 768,
    'simplewiki-openai-3072-normalized': 3072,
    'imagenet-align-640-normalized': 640,
    'yandex-200-cosine': 200,
    'imagenet-clip-512-normalized': 512,
    'laion-clip-512-normalized': 512,
    'codesearchnet-jina-768-cosine': 768,
    'yi-128-ip': 128,
    'landmark-dino-768-cosine': 768,
    'landmark-nomic-768-normalized': 768,
    'arxiv-nomic-768-normalized': 768,
    'ccnews-nomic-768-normalized': 768,
    'celeba-resnet-2048-cosine': 2048,
    'llama-128-ip': 128,
    'yahoo-minilm-384-normalized': 384
}

DATA_EXPONENTS = {
    'msong-420': 3,
    'sift-128-euclidean': 0,
    'instructorxl-arxiv-768': 3,
    'openai-1536-angular': 3,
    'gist-960-euclidean': 2
}
