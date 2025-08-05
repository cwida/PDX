import faiss
from setup_utils import *
from setup_settings import *
from pdxearch.index_base import BaseIndexPDXIVF, BaseIndexPDXIMI
from pdxearch.preprocessors import ADSampling
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler


def generate_adsampling_ivf(dataset_name: str, _type='pdx', normalize=True):
    print(dataset_name)
    base_idx = BaseIndexPDXIVF(DIMENSIONALITIES[dataset_name], 'l2sq')
    # Core index IVF must exist
    index_path = os.path.join(CORE_INDEXES_FAISS, get_core_index_filename(dataset_name, norm=normalize))
    # Reads the core index created by faiss to generate the PDX index
    base_idx.core_index.read_index(index_path)
    data = read_hdf5_train_data(dataset_name)
    if normalize:
        print('Normalizing')
        data = preprocessing.normalize(data, axis=1, norm='l2')
    preprocessor = ADSampling(DIMENSIONALITIES[dataset_name])
    preprocessor.preprocess(data, inplace=True)
    print('Saving...')
    # PDX
    base_idx._to_pdx(data, _type='pdx', centroids_preprocessor=preprocessor, use_original_centroids=True)
    base_idx._persist(os.path.join(PDX_ADSAMPLING_DATA, dataset_name + '-ivf'))

    # Store metadata needed by ADSampling
    preprocessor.store_metadata(os.path.join(NARY_ADSAMPLING_DATA, dataset_name + '-matrix'))

def generate_adsampling_ivf_global8(dataset_name: str, normalize=True):
    base_idx = BaseIndexPDXIVF(DIMENSIONALITIES[dataset_name], 'l2sq')
    # Core index IVF must exist
    index_path = os.path.join(CORE_INDEXES_FAISS, get_core_index_filename(dataset_name, norm=normalize))
    # Reads the core index created by faiss to generate the PDX index
    base_idx.core_index.read_index(index_path)
    print('Reading train data')
    data = read_hdf5_train_data(dataset_name)
    preprocessor = ADSampling(DIMENSIONALITIES[dataset_name])
    preprocessor.preprocess(data, inplace=True, normalize=True)
    print('Saving')
    # PDX FLAT BLOCKIFIED
    base_idx._to_pdx(data, _type='pdx-v4-h', quantize=True, centroids_preprocessor=preprocessor, use_original_centroids=True)
    base_idx._persist(os.path.join(PDX_ADSAMPLING_DATA, dataset_name + '-ivf-u8'))

    preprocessor.store_metadata(os.path.join(NARY_ADSAMPLING_DATA, dataset_name + '-ivf-u8-matrix'))

def generate_adsampling_imi(dataset_name: str, normalize=True):
    base_idx = BaseIndexPDXIMI(DIMENSIONALITIES[dataset_name], 'l2sq')
    # Core index IVF must exist
    index_path = os.path.join(CORE_INDEXES_FAISS, get_core_index_filename(dataset_name, norm=normalize))
    index_path_l0 = os.path.join(CORE_INDEXES_FAISS_L0, get_core_index_filename(dataset_name, norm=normalize))
    # Reads the core index created by faiss to generate the PDX index
    base_idx.core_index.read_index(index_path, index_path_l0)
    data = read_hdf5_train_data(dataset_name)
    preprocessor = ADSampling(DIMENSIONALITIES[dataset_name])
    preprocessor.preprocess(data, inplace=True)
    print('Saving...')
    # PDX
    base_idx._to_pdx(data, _type='pdx', centroids_preprocessor=preprocessor, use_original_centroids=True)
    base_idx._persist(os.path.join(PDX_ADSAMPLING_DATA, dataset_name + '-imi'))

    # Store metadata needed by ADSampling
    preprocessor.store_metadata(os.path.join(NARY_ADSAMPLING_DATA, dataset_name + '-imi-matrix'))

def generate_adsampling_imi_global8(dataset_name: str, normalize=True):
    base_idx = BaseIndexPDXIMI(DIMENSIONALITIES[dataset_name], 'l2sq')
    # Core index IVF must exist
    index_path = os.path.join(CORE_INDEXES_FAISS, get_core_index_filename(dataset_name, norm=normalize))
    index_path_l0 = os.path.join(CORE_INDEXES_FAISS_L0, get_core_index_filename(dataset_name, norm=normalize))
    # Reads the core index created by faiss to generate the PDX index
    print('Reading index')
    base_idx.core_index.read_index(index_path, index_path_l0)
    # base_idx.core_index.index = faiss.read_index(index_path)
    # base_idx.core_index.index_l0 = faiss.read_index(index_path_l0)
    print('Reading train data')
    data = read_hdf5_train_data(dataset_name)
    preprocessor = ADSampling(DIMENSIONALITIES[dataset_name])
    preprocessor.preprocess(data, inplace=True, normalize=True)
    print('Saving')
    # PDX FLAT BLOCKIFIED
    base_idx._to_pdx(data, _type='pdx-v4-h', quantize=True, centroids_preprocessor=preprocessor, use_original_centroids=True)
    base_idx._persist(os.path.join(PDX_ADSAMPLING_DATA, dataset_name + '-imi-u8'))

    preprocessor.store_metadata(os.path.join(NARY_ADSAMPLING_DATA, dataset_name + '-imi-u8-matrix'))


if __name__ == "__main__":
    generate_adsampling_ivf('coco-nomic-768-normalized', normalize=True)
    generate_adsampling_ivf('simplewiki-openai-3072-normalized', normalize=True)
    generate_adsampling_ivf('imagenet-align-640-normalized', normalize=True)
    generate_adsampling_ivf('imagenet-clip-512-normalized', normalize=True)
    generate_adsampling_ivf('laion-clip-512-normalized', normalize=True)
    generate_adsampling_ivf('codesearchnet-jina-768-cosine', normalize=True)
    generate_adsampling_ivf('yi-128-ip', normalize=True)
    generate_adsampling_ivf('landmark-dino-768-cosine', normalize=True)
    generate_adsampling_ivf('landmark-nomic-768-normalized', normalize=True)
    generate_adsampling_ivf('arxiv-nomic-768-normalized', normalize=True)
    generate_adsampling_ivf('ccnews-nomic-768-normalized', normalize=True)
    generate_adsampling_ivf('celeba-resnet-2048-cosine', normalize=True)
    generate_adsampling_ivf('llama-128-ip', normalize=True)
    generate_adsampling_ivf('yandex-200-cosine', normalize=True)
    generate_adsampling_ivf('word2vec-300', normalize=True)
    generate_adsampling_ivf('sift-128-euclidean', normalize=True)
    generate_adsampling_ivf('openai-1536-angular', normalize=True)
    generate_adsampling_ivf('msong-420', normalize=True)
    generate_adsampling_ivf('instructorxl-arxiv-768', normalize=True)
    generate_adsampling_ivf('contriever-768', normalize=True)
    generate_adsampling_ivf('gist-960-euclidean', normalize=True)
    generate_adsampling_ivf('yahoo-minilm-384-normalized', normalize=True)
    generate_adsampling_ivf('gooaq-distilroberta-768-normalized', normalize=True)
    generate_adsampling_ivf('agnews-mxbai-1024-euclidean', normalize=True)
    generate_adsampling_ivf('glove-200-angular', normalize=True)

    generate_adsampling_ivf_global8('coco-nomic-768-normalized', normalize=True)
    generate_adsampling_ivf_global8('simplewiki-openai-3072-normalized', normalize=True)
    generate_adsampling_ivf_global8('imagenet-align-640-normalized', normalize=True)
    generate_adsampling_ivf_global8('imagenet-clip-512-normalized', normalize=True)
    generate_adsampling_ivf_global8('laion-clip-512-normalized', normalize=True)
    generate_adsampling_ivf_global8('codesearchnet-jina-768-cosine', normalize=True)
    generate_adsampling_ivf_global8('yi-128-ip', normalize=True)
    generate_adsampling_ivf_global8('landmark-dino-768-cosine', normalize=True)
    generate_adsampling_ivf_global8('landmark-nomic-768-normalized', normalize=True)
    generate_adsampling_ivf_global8('arxiv-nomic-768-normalized', normalize=True)
    generate_adsampling_ivf_global8('ccnews-nomic-768-normalized', normalize=True)
    generate_adsampling_ivf_global8('celeba-resnet-2048-cosine', normalize=True)
    generate_adsampling_ivf_global8('llama-128-ip', normalize=True)
    generate_adsampling_ivf_global8('yandex-200-cosine', normalize=True)
    generate_adsampling_ivf_global8('word2vec-300', normalize=True)
    generate_adsampling_ivf_global8('sift-128-euclidean', normalize=True)
    generate_adsampling_ivf_global8('openai-1536-angular', normalize=True)
    generate_adsampling_ivf_global8('msong-420', normalize=True)
    generate_adsampling_ivf_global8('instructorxl-arxiv-768', normalize=True)
    generate_adsampling_ivf_global8('contriever-768', normalize=True)
    generate_adsampling_ivf_global8('gist-960-euclidean', normalize=True)
    generate_adsampling_ivf_global8('yahoo-minilm-384-normalized', normalize=True)
    generate_adsampling_ivf_global8('gooaq-distilroberta-768-normalized', normalize=True)
    generate_adsampling_ivf_global8('agnews-mxbai-1024-euclidean', normalize=True)
    generate_adsampling_ivf_global8('glove-200-angular', normalize=True)

    generate_adsampling_imi('coco-nomic-768-normalized', normalize=True)
    generate_adsampling_imi('simplewiki-openai-3072-normalized', normalize=True)
    generate_adsampling_imi('imagenet-align-640-normalized', normalize=True)
    generate_adsampling_imi('imagenet-clip-512-normalized', normalize=True)
    generate_adsampling_imi('laion-clip-512-normalized', normalize=True)
    generate_adsampling_imi('codesearchnet-jina-768-cosine', normalize=True)
    generate_adsampling_imi('yi-128-ip', normalize=True)
    generate_adsampling_imi('landmark-dino-768-cosine', normalize=True)
    generate_adsampling_imi('landmark-nomic-768-normalized', normalize=True)
    generate_adsampling_imi('arxiv-nomic-768-normalized', normalize=True)
    generate_adsampling_imi('ccnews-nomic-768-normalized', normalize=True)
    generate_adsampling_imi('celeba-resnet-2048-cosine', normalize=True)
    generate_adsampling_imi('llama-128-ip', normalize=True)
    generate_adsampling_imi('yandex-200-cosine', normalize=True)
    generate_adsampling_imi('word2vec-300', normalize=True)
    generate_adsampling_imi('sift-128-euclidean', normalize=True)
    generate_adsampling_imi('openai-1536-angular', normalize=True)
    generate_adsampling_imi('msong-420', normalize=True)
    generate_adsampling_imi('instructorxl-arxiv-768', normalize=True)
    generate_adsampling_imi('contriever-768', normalize=True)
    generate_adsampling_imi('gist-960-euclidean', normalize=True)
    generate_adsampling_imi('yahoo-minilm-384-normalized', normalize=True)
    generate_adsampling_imi('gooaq-distilroberta-768-normalized', normalize=True)
    generate_adsampling_imi('agnews-mxbai-1024-euclidean', normalize=True)
    generate_adsampling_imi('glove-200-angular', normalize=True)

    generate_adsampling_imi_global8('coco-nomic-768-normalized', normalize=True)
    generate_adsampling_imi_global8('simplewiki-openai-3072-normalized', normalize=True)
    generate_adsampling_imi_global8('imagenet-align-640-normalized', normalize=True)
    generate_adsampling_imi_global8('imagenet-clip-512-normalized', normalize=True)
    generate_adsampling_imi_global8('laion-clip-512-normalized', normalize=True)
    generate_adsampling_imi_global8('codesearchnet-jina-768-cosine', normalize=True)
    generate_adsampling_imi_global8('yi-128-ip', normalize=True)
    generate_adsampling_imi_global8('landmark-dino-768-cosine', normalize=True)
    generate_adsampling_imi_global8('landmark-nomic-768-normalized', normalize=True)
    generate_adsampling_imi_global8('arxiv-nomic-768-normalized', normalize=True)
    generate_adsampling_imi_global8('ccnews-nomic-768-normalized', normalize=True)
    generate_adsampling_imi_global8('celeba-resnet-2048-cosine', normalize=True)
    generate_adsampling_imi_global8('llama-128-ip', normalize=True)
    generate_adsampling_imi_global8('yandex-200-cosine', normalize=True)
    generate_adsampling_imi_global8('word2vec-300', normalize=True)
    generate_adsampling_imi_global8('sift-128-euclidean', normalize=True)
    generate_adsampling_imi_global8('openai-1536-angular', normalize=True)
    generate_adsampling_imi_global8('msong-420', normalize=True)
    generate_adsampling_imi_global8('instructorxl-arxiv-768', normalize=True)
    generate_adsampling_imi_global8('contriever-768', normalize=True)
    generate_adsampling_imi_global8('gist-960-euclidean', normalize=True)
    generate_adsampling_imi_global8('yahoo-minilm-384-normalized', normalize=True)
    generate_adsampling_imi_global8('gooaq-distilroberta-768-normalized', normalize=True)
    generate_adsampling_imi_global8('agnews-mxbai-1024-euclidean', normalize=True)
    generate_adsampling_imi_global8('glove-200-angular', normalize=True)