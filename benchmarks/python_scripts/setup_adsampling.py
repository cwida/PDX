import faiss
from setup_utils import *
from setup_settings import *
from pdxearch.index_base import BaseIndexPDXIVF
from pdxearch.preprocessors import ADSampling
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler


def generate_adsampling_ivf(dataset_name: str, _types=('pdx', 'dual'), normalize=True):
    base_idx = BaseIndexPDXIVF(DIMENSIONALITIES[dataset_name], 'l2sq')
    # Core index IVF must exist
    index_path = os.path.join(CORE_INDEXES_FAISS, get_core_index_filename(dataset_name, norm=normalize))
    # Reads the core index created by faiss to generate the PDX index
    base_idx.core_index.index = faiss.read_index(index_path)
    data = read_hdf5_train_data(dataset_name)
    print('Normalizing')
    if normalize:
        data = preprocessing.normalize(data, axis=1, norm='l2')
    preprocessor = ADSampling(DIMENSIONALITIES[dataset_name])
    preprocessor.preprocess(data, inplace=True)
    print('Saving...')
    # PDX
    base_idx._to_pdx(data, _type='pdx', centroids_preprocessor=preprocessor, use_original_centroids=True)
    base_idx._persist(os.path.join(PDX_ADSAMPLING_DATA, dataset_name + '-ivf'))

    # DUAL-BLOCK
    #base_idx._to_pdx(data, _type='dual', delta_d=get_delta_d(len(data[0])), centroids_preprocessor=preprocessor, use_original_centroids=True)
    #base_idx._persist(os.path.join(NARY_ADSAMPLING_DATA, dataset_name + '-ivf-dual-block'))

    # METADATA
    # Store metadata needed by ADSampling
    preprocessor.store_metadata(os.path.join(NARY_ADSAMPLING_DATA, dataset_name + '-matrix'))


if __name__ == "__main__":
    # generate_adsampling_ivf('sift-128-euclidean')
    # generate_adsampling_ivf('openai-1536-angular')
    # generate_adsampling_ivf('instructorxl-arxiv-768')
    # generate_adsampling_ivf('msong-420')
    # generate_adsampling_ivf('contriever-768')
    # generate_adsampling_ivf('instructorxl-arxiv-768')
    generate_adsampling_ivf('gooaq-distilroberta-768-normalized')
    generate_adsampling_ivf('agnews-mxbai-1024-euclidean')
    generate_adsampling_ivf('coco-nomic-768-normalized')
    generate_adsampling_ivf('simplewiki-openai-3072-normalized')
