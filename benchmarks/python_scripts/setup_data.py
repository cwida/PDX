import os
import zipfile
from setup_utils import RAW_DATA, DATA_DIRECTORY, DATASET_INFO
from setup_pdx import generate_ground_truth, generate_index, generate_test_data
from setup_faiss import generate_faiss_index, generate_faiss_sq8_index

DOWNLOAD = False  # Download raw HDF5 data
GENERATE_GT = False  # Creates ground truth with sklearn
KNN = [100]
SEED = 42
DATASETS_TO_USE = [
    'mxbai',
    'arxiv',
    'openai',
    # 'cohere'
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
    if not len(DATASETS_TO_USE):
        DATASETS_TO_USE = list(DATASET_INFO.keys())

    for dataset in DATASETS_TO_USE:
        print(f'\n================ PROCESSING: {dataset} ================')
        if GENERATE_GT:
            print('==== Generating ground truth...')
            generate_ground_truth(dataset, KNN)

        print('==== Saving queries in a binary format...')
        generate_test_data(dataset)

        print('==== Generating PDX indexes...')
        generate_index(dataset, 'pdx_f32', normalize=True, seed=SEED)
        generate_index(dataset, 'pdx_u8', normalize=True, seed=SEED)
        generate_index(dataset, 'pdx_tree_f32', normalize=True, seed=SEED)
        generate_index(dataset, 'pdx_tree_u8', normalize=True, seed=SEED)

        print('==== Generating FAISS indexes...')
        generate_faiss_index(dataset, normalize=True)
        generate_faiss_sq8_index(dataset, normalize=True)