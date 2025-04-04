import faiss
import math
from setup_utils import *
from setup_settings import *
from pdxearch.index_base import BaseIndexPDXIVF
from pdxearch.preprocessors import ADSampling
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


class Partition:
    def __init__(self):
        self.num_embeddings = 0
        self.indices = np.array([])
        self.blocks = []


def generate_adsampling_ivf(dataset_name: str, _types=('pdx', 'dual')):
    base_idx = BaseIndexPDXIVF(DIMENSIONALITIES[dataset_name], 'l2sq')
    # Core index IVF must exist
    index_path = os.path.join(CORE_INDEXES_FAISS, get_core_index_filename(dataset_name))
    # Reads the core index created by faiss to generate the PDX index
    base_idx.core_index.index = faiss.read_index(index_path)
    before_norm_data = read_hdf5_train_data(dataset_name)

    data = preprocessing.normalize(before_norm_data, axis=1, norm='l2')
    preprocessor = ADSampling(DIMENSIONALITIES[dataset_name])
    adsampling_data = preprocessor.preprocess(data, inplace=False)

    print('Saving...')
    print('Partitions', base_idx.core_index.index.nlist)
    #for list_id in range(base_idx.core_index.index.nlist):
    for list_id in range(101):
        num_list_embeddings, list_ids = base_idx.core_index.get_inverted_list_metadata(list_id)
        partition = Partition()
        partition.num_embeddings = num_list_embeddings
        partition.indices = np.zeros((partition.num_embeddings,), dtype=np.uint32)
        for embedding_index in range(partition.num_embeddings):
            partition.indices[embedding_index] = list_ids[embedding_index]
        # print(data[partition.indices, :])
        # pre_data = data[partition.indices, :]
        # means = np.mean(pre_data, axis=0)
        # stdevs = np.std(pre_data, axis=0)
        # max_vals = np.max(pre_data, axis=0)
        # min_vals = np.min(pre_data, axis=0)
        # ranges = max_vals - min_vals
        # p90 = np.percentile(pre_data, 90, axis=0)
        # # unique_counts = np.array([len(np.unique(data[:, i])) for i in range(data.shape[1])])
        # extended_data = np.vstack([pre_data, means, stdevs, max_vals, min_vals, ranges, p90])

        ads_pre_data = adsampling_data[partition.indices, :]
        pre_data = data[partition.indices, :]
        before_norm_part = before_norm_data[partition.indices, :]
        # ads_pre_data = ads_pre_data * math.pow(10, DATA_EXPONENTS[dataset_name])
        # ads_pre_data = ads_pre_data * math.pow(10, 2.5)
        ads_pre_data = ads_pre_data * math.pow(10, 3)
        ads_pre_data = ads_pre_data.round(decimals=0)
        ads_min_vals = np.min(ads_pre_data, axis=0)
        for_data = ads_pre_data - ads_min_vals
        for_data_max = np.max(for_data, axis=0)

        # ads_means = np.mean(ads_pre_data, axis=0)
        # ads_stdevs = np.std(ads_pre_data, axis=0)
        ads_max_vals = np.max(ads_pre_data, axis=0)

        ads_ranges = ads_max_vals - ads_min_vals

        pre_data_min = np.min(pre_data, axis=0)
        pre_data_max = np.max(pre_data, axis=0)
        pre_data_ranges = pre_data_max - pre_data_min

        bnd_min = np.min(before_norm_part, axis=0)
        bnd_max = np.max(before_norm_part, axis=0)
        bnd_range = bnd_max - bnd_min

        # ads_p90 = np.percentile(ads_pre_data, 90, axis=0)
        # if np.any(for_data > 255):
        #     raise ValueError(f'Overflow detected when converting to uint8 in partition {list_id}')
        # ads_unique_counts = np.array([len(np.unique(data[:, i])) for i in range(data.shape[1])])
        ads_extended_data = np.vstack([ads_pre_data, ads_min_vals, ads_ranges])
        normal_extended_data = np.vstack([pre_data, pre_data_min, pre_data_ranges])
        raw_extended_data = np.vstack([before_norm_part, bnd_min, bnd_range])
        for_data_extended = np.vstack([for_data, for_data_max])
        if list_id == 10 or list_id == 64:
            # print(for_data)
            # np.savetxt(f'./benchmarks/python_scripts/partition_info/ads_for_{dataset_name}_{list_id}.dat', for_data_extended, '%6.3f')
            # np.savetxt(f'./benchmarks/python_scripts/partition_info/ads_raw_{dataset_name}_{list_id}.dat', ads_extended_data, '%6.3f')
            # np.savetxt(f'./benchmarks/python_scripts/partition_info/norm_{dataset_name}_{list_id}.dat', normal_extended_data, '%6.3f')
            # np.savetxt(f'./benchmarks/python_scripts/partition_info/raw_{dataset_name}_{list_id}.dat', raw_extended_data, '%6.3f')
            for dimm in [2, 33, 99]:
                df = pd.DataFrame(ads_pre_data)
                # print(df)
                d_toplot = df.loc[:, dimm] # column
                d_toplot = df.loc[dimm, :]  # row
                # print(d_toplot)
                # pd.set_option('display.max_columns', None)
                # pd.set_option('display.max_rows', None)
                # print("Variance:", d_toplot.var())
                # print("Unique", d_toplot.nunique())
                plt.figure()
                g = sns.histplot(d_toplot, bins=25, kde=True, cumulative=True)
                # g = sns.kdeplot(
                #     d_toplot,
                #     fill=True,
                #     palette="mako",
                #     linewidth=0.04,
                #     alpha=0.02,
                #     log_scale=True
                # )
                # g.set_title("")
                # g.set_xlabel("")
                # g.set_ylabel("")
                # g.set_xticklabels([])
                # g.set_yticklabels([])
                g.legend().set_visible(False)
                sns.despine(bottom=True, left=True, right=True, top=True)
                plt.savefig(f'./benchmarks/python_scripts/partition_info/{dataset_name}_{list_id}_{dimm}_bf_row.png', dpi=300, bbox_inches='tight', pad_inches=0.0)
                plt.close()
    # PDX
    # base_idx._to_pdx(data, _type='pdx', centroids_preprocessor=preprocessor, use_original_centroids=True)
    # base_idx._persist(os.path.join(PDX_ADSAMPLING_DATA, dataset_name + '-ivf'))

    # METADATA
    # Store metadata needed by ADSampling
    # preprocessor.store_metadata(os.path.join(NARY_ADSAMPLING_DATA, dataset_name + '-matrix'))


if __name__ == "__main__":
    # generate_adsampling_ivf('fashion-mnist-784-euclidean')
    generate_adsampling_ivf('openai-1536-angular')
    # generate_adsampling_ivf('sift-128-euclidean')
    # generate_adsampling_ivf('msong-420')
    # generate_adsampling_ivf('instructorxl-arxiv-768')
    # generate_adsampling_ivf('contriever-768')
    # generate_adsampling_ivf('gist-960-euclidean')
