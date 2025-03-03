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

PURESCAN_DATA = os.path.join(SOURCE_DIR, DATA_DIRECTORY, "purescan")

QUERIES_DATA = os.path.join(SOURCE_DIR, DATA_DIRECTORY, "queries")

NARY_DATA = os.path.join(SOURCE_DIR, DATA_DIRECTORY, "nary")
NARY_ADSAMPLING_DATA = os.path.join(SOURCE_DIR, DATA_DIRECTORY, "adsampling_nary")
NARY_BSA_DATA = os.path.join(SOURCE_DIR, DATA_DIRECTORY, "bsa_nary")

DSM_ADSAMPLING_DATA = os.path.join(SOURCE_DIR, DATA_DIRECTORY, "adsampling_dsm")
DSM_BSA_DATA = os.path.join(SOURCE_DIR, DATA_DIRECTORY, "bsa_dsm")
DSM_BOND_DATA = os.path.join(SOURCE_DIR, DATA_DIRECTORY, "bond_dsm")

PDX_DATA = os.path.join(SOURCE_DIR, DATA_DIRECTORY, "pdx")
PDX_ADSAMPLING_DATA = os.path.join(SOURCE_DIR, DATA_DIRECTORY, "adsampling_pdx")
PDX_BSA_DATA = os.path.join(SOURCE_DIR, DATA_DIRECTORY, "bsa_pdx")

directories = [
    RAW_DATA, GROUND_TRUTH_DATA, PURESCAN_DATA,
    NARY_DATA, NARY_ADSAMPLING_DATA, NARY_BSA_DATA,
    DSM_ADSAMPLING_DATA, DSM_BSA_DATA, DSM_BOND_DATA,
    PDX_DATA, PDX_ADSAMPLING_DATA, PDX_BSA_DATA,
    CORE_INDEXES, CORE_INDEXES_FAISS, QUERIES_DATA,
    SEMANTIC_GROUND_TRUTH_PATH
]

for needed_directory in directories:
    if not os.path.exists(needed_directory):
        os.makedirs(needed_directory)

# Datasets to set up
DATASETS = [
    # 'random-xs-20-angular',
    # 'random-s-100-euclidean',
    # 'nytimes-256-angular',
    # 'mnist-784-euclidean',
    # 'glove-25-angular',
    # 'glove-100-angular',
    # 'trevi-4096',
    # 'stl-9216',
    # 'har-561',

    # 'nytimes-16-angular',
    'fashion-mnist-784-euclidean',
    # 'glove-50-angular',
    # 'glove-200-angular',
    # 'sift-128-euclidean',
    # 'msong-420',
    # 'contriever-768',
    # 'gist-960-euclidean',
    # 'deep-image-96-angular',
    # 'instructorxl-arxiv-768',
    # 'openai-1536-angular'
]

