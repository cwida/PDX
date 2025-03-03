import os
import zipfile
from setup_settings import RAW_DATA, DATA_DIRECTORY, DATASETS
from setup_ground_truth import generate_ground_truth
from setup_adsampling import generate_adsampling_ivf
from setup_bsa import generate_bsa_ivf
from setup_bond import generate_bond_ivf, generate_bond_flat
from setup_core_index import generate_core_ivf
from setup_test_data import generate_test_data
from setup_purescan import generate_synthetic_data

DOWNLOAD = True  # Download raw HDF5 data
GENERATE_GT = False  # Creates ground truth with sklearn
GENERATE_IVF = True  # Creates IVF indexes with FAISS
GENERATE_SYNTHETIC = False  # Generates synthetic collections of vectors for the kernels experiment
KNN = [10]
ALGORITHMS = [  # Choose the pruning algorithms for which indexes are going to be created
    'adsampling',
    'bsa',
    'bond'
]
if __name__ == "__main__":
    if DOWNLOAD:
        import gdown
        gdown.download(
            'https://drive.google.com/file/d/1I8pbwGDCSe3KqfIegAllwoP5q6F4ohj2/view?usp=sharing',
            os.path.join(DATA_DIRECTORY, 'datasets_hdf5.zip'),
            fuzzy=True
        )
        with zipfile.ZipFile(os.path.join(DATA_DIRECTORY, 'datasets_hdf5.zip'), 'r') as zip_ref:
            zip_ref.extractall(RAW_DATA)

    if GENERATE_SYNTHETIC:
        generate_synthetic_data()
        generate_synthetic_data(32, (768,), (1024,))
        generate_synthetic_data(128, (768,), (1024,))
        generate_synthetic_data(256, (768,), (1024,))
        generate_synthetic_data(512, (768,), (1024,))

    for dataset in DATASETS:
        print('\n================ PROCESSING:', dataset, '================')
        if GENERATE_GT:
            print('==== Generating ground truth...')
            generate_ground_truth(dataset, KNN)

        print('==== Saving queries in a binary format...')
        generate_test_data(dataset)

        if GENERATE_IVF:
            print('==== Creating Core IVF index with FAISS (this might take a while)...')
            generate_core_ivf(dataset)

        if 'adsampling' in ALGORITHMS:
            print('==== Generating ADSampling  [PDX & N-ary]...')
            generate_adsampling_ivf(dataset)

        if 'bsa' in ALGORITHMS:
            print('==== Generating BSA [PDX & N-ary] (this might take a while)...')
            generate_bsa_ivf(dataset)

        # Generate BOND Data
        if 'bond' in ALGORITHMS:
            print('==== Generating BOND IVF [PDX]')
            generate_bond_ivf(dataset)
            print('==== Generating BOND Flat [PDX] (for exact-search)')
            generate_bond_flat(dataset)

