import os
import zipfile
from setup_settings import RAW_DATA, DATA_DIRECTORY, DATASETS
from setup_ground_truth import generate_ground_truth
from setup_adsampling import generate_adsampling_ivf, generate_adsampling_ivf_global8,  generate_adsampling_imi_global8, generate_adsampling_imi
from setup_bond import generate_bond_ivf, generate_bond_flat
from setup_core_index import generate_core_ivf, generate_core_imi
from setup_test_data import generate_test_data

DOWNLOAD = False  # Download raw HDF5 data
GENERATE_GT = False  # Creates ground truth with sklearn
GENERATE_IVF = True # Creates IVF indexes with FAISS
KNN = [100]
ALGORITHMS = [  # Choose the pruning algorithms for which indexes are going to be created
    'adsampling',
    # 'bond'
]
DATASETS_TO_USE = [
    'openai-1536-angular',
    'agnews-mxbai-1024-euclidean',
    'instructorxl-arxiv-768',
    'simplewiki-openai-3072-normalized',
    'msong-420',
    'llama-128-ip',
]
if __name__ == "__main__":
    if DOWNLOAD:
        import gdown
        # All datasets: ~60GB compressed, ~80GB uncompressed
        gdown.download(
            'https://drive.google.com/file/d/1ei6DV0goMyInp_wFcrbJG3KV40mAPfAa/view?usp=sharing',
            os.path.join(DATA_DIRECTORY, 'datasets_hdf5.zip'),
            fuzzy=True
        )
        with zipfile.ZipFile(os.path.join(DATA_DIRECTORY, 'datasets_hdf5.zip'), 'r') as zip_ref:
            zip_ref.extractall(RAW_DATA)

    # If you don't define some datasets we will try to use all of them
    if not len(DATASETS_TO_USE): DATASETS_TO_USE = DATASETS
    for dataset in DATASETS_TO_USE:
        print('\n================ PROCESSING:', dataset, '================')
        if GENERATE_GT:
            print('==== Generating ground truth...')
            generate_ground_truth(dataset, KNN)

        print('==== Saving queries in a binary format...')
        generate_test_data(dataset)

        if GENERATE_IVF:
            print('==== Creating Core IMI index with FAISS (this might take a while)...')
            generate_core_imi(dataset)

        if 'adsampling' in ALGORITHMS:
            print('==== Generating ADSampling...')
            generate_adsampling_ivf(dataset)
            generate_adsampling_imi(dataset)

            print('==== Generating ADSampling SQ8...')
            generate_adsampling_ivf_global8(dataset)
            generate_adsampling_imi_global8(dataset)

        # Generate BOND Data
        if 'bond' in ALGORITHMS:
            print('==== Generating BOND IVF [PDX]')
            generate_bond_ivf(dataset)
            print('==== Generating BOND Flat (for exact-search)')
            generate_bond_flat(dataset)

