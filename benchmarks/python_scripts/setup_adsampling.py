from setup_utils import *
from setup_settings import *
from pdxearch import IndexPDXIVF, IndexPDXIVFSQ8, IndexPDXIVF2, IndexPDXIVF2SQ8

INDEX_CLASSES = {
    "pdx_f32": IndexPDXIVF,
    "pdx_u8": IndexPDXIVFSQ8,
    "pdx_tree_f32": IndexPDXIVF2,
    "pdx_tree_u8": IndexPDXIVF2SQ8,
}


def generate_index(dataset_abbrev: str, index_type: str, normalize=True):
    hdf5_name, dims = DATASET_INFO[dataset_abbrev]
    print(f'{dataset_abbrev} -> {hdf5_name} ({index_type})')
    data = read_hdf5_train_data(hdf5_name)

    cls = INDEX_CLASSES[index_type]
    index = cls(num_dimensions=dims, normalize=normalize)
    print('Building index...')
    index.build(data)
    print(f'Index built: {index.num_clusters} clusters')

    save_path = os.path.join(PDX_ADSAMPLING_DATA, dataset_abbrev + '-' + index_type)
    print(f'Saving to {save_path}')
    index.save(save_path)
    print('Done.')


if __name__ == "__main__":
    # generate_index('sift', 'pdx_f32', normalize=True)
    # generate_index('sift', 'pdx_u8', normalize=True)
    # generate_index('sift', 'pdx_tree_f32', normalize=True)
    # generate_index('sift', 'pdx_tree_u8', normalize=True)

    # generate_index('mxbai', 'pdx_f32', normalize=True)
    # generate_index('mxbai', 'pdx_u8', normalize=True)
    # generate_index('mxbai', 'pdx_tree_f32', normalize=True)
    # generate_index('mxbai', 'pdx_tree_u8', normalize=True)

    # generate_index('gist', 'pdx_f32', normalize=True)
    # generate_index('openai', 'pdx_f32', normalize=True)
    # generate_index('arxiv', 'pdx_f32', normalize=True)
    # generate_index('wiki', 'pdx_f32', normalize=True)
    # generate_index('contriever', 'pdx_f32', normalize=True)
    # generate_index('clip', 'pdx_f32', normalize=True)
    # generate_index('yahoo', 'pdx_f32', normalize=True)
    # generate_index('yandex', 'pdx_f32', normalize=True)
    # generate_index('glove200', 'pdx_f32', normalize=True)
    # generate_index('yi', 'pdx_f32', normalize=True)
    # generate_index('llama', 'pdx_f32', normalize=True)
    pass
