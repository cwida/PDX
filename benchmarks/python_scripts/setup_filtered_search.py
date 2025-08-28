import numpy as np
np.random.seed(42)

import json
import random
from setup_utils import *
from setup_settings import *
from pdxearch.index_base import BaseIndexPDXIVF, BaseIndexPDXIVF2
from sklearn import preprocessing
from WrapperBruteForce import BruteForceSKLearn

from decimal import Decimal

SELECTIVITIES = [
    0.0001,
    0.000135,
    0.00015, 0.001, 0.01,
    0.1, 0.2, 0.3, 0.4, 0.5, 0.75,
    0.9, 0.95, 0.99,
]

SPECIAL_SELECTIVITIES = [
    "PART 30",
    "PART 1",
    "PART+ 1",
]

def generate_ground_truth(dataset, train, test, filtered_ids, selectivity_str, KNNS=(100,), normalize=True):
    print('Generating ground truth for', len(train), 'points', 'at', selectivity_str, 'selectivity')
    N_QUERIES = len(test)
    algo = BruteForceSKLearn("euclidean", njobs=-1)
    algo.fit(train)
    for knn in KNNS:
        gt_filename = f"{dataset}_{knn}_norm_{selectivity_str}.json"
        gt_name = os.path.join(SEMANTIC_FILTERED_GROUND_TRUTH_PATH, gt_filename)
        gt = {}
        index_data = []
        distance_data = []
        print('Querying for GT...')
        dist, index = algo.query_batch(test, n=knn)
        for i in range(N_QUERIES):
            index_data.append(filtered_ids[index[i]])
            distance_data.append(dist[i] ** 2)
            gt[i] = filtered_ids[index[i]].tolist()
        with open(os.path.join(FILTERED_GROUND_TRUTH_DATA, gt_filename.replace('.json', '')), "wb") as file:
            file.write(np.array(index_data, dtype=np.uint32).tobytes("C"))
            file.write(np.array(distance_data, dtype=np.float32).tobytes("C"))
        with open(gt_name, 'w') as f:
            json.dump(gt, f)

def generate_selection_vector(dataset_name: str, _type='pdx', normalize=True):
    print(dataset_name)

    train, test = read_hdf5_data(dataset_name)
    if normalize:
        train = preprocessing.normalize(train, axis=1, norm='l2')
        test = preprocessing.normalize(test, axis=1, norm='l2')

    base_idx = BaseIndexPDXIVF(DIMENSIONALITIES[dataset_name], 'l2sq')
    # Core index IVF must exist
    index_path = os.path.join(CORE_INDEXES_FAISS, get_core_index_filename(dataset_name, norm=normalize))
    # Reads the core index created by faiss to generate the PDX index
    base_idx.core_index.read_index(index_path)

    n_clusters = base_idx.core_index.nbuckets
    labels = base_idx.core_index.labels
    all_ordered_labels = np.concatenate([np.array(sub) for sub in labels])
    assert(n_clusters == len(labels))

    # The GT is not correctly calculated

    total_points = sum(len(lst) for lst in labels)
    print('Total points', total_points)

    for selectivity in SELECTIVITIES:
        selection_per_cluster = []
        selection_vector = (np.random.rand(total_points) < selectivity).astype(np.uint8)
        selected = np.sum(selection_vector)
        print(f'For selectivity {selectivity}, {selected} points were chosen ({float(selected) / total_points})')
        real_selection_vectors = np.array([]).astype(np.uint8)
        for l in labels:
            l_np = np.array(l)
            real_selection_vector = selection_vector[l_np]
            selected_on_this_cluster = np.sum(real_selection_vector)
            selection_per_cluster.append(selected_on_this_cluster)
            real_selection_vectors = np.concatenate((real_selection_vectors, real_selection_vector))
        selection_per_cluster = np.array(selection_per_cluster).astype(np.uint32)
        assert(len(selection_per_cluster) == len(labels))
        assert(np.sum(selection_per_cluster) == selected)
        assert(total_points == len(real_selection_vectors))
        assert(np.sum(selection_per_cluster) == np.sum(real_selection_vectors))

        selectivity_str = format(Decimal(str(selectivity)), 'f').replace('.', '_')
        selectivity_filename = f'{dataset_name}_{selectivity_str}.bin'
        selectivity_filename_path = os.path.join(FILTER_SELECTION_VECTORS, selectivity_filename)
        data = bytearray()
        data.extend(selection_per_cluster.tobytes("C"))
        data.extend(real_selection_vectors.tobytes("C"))
        with open(selectivity_filename_path, "wb") as file:
            file.write(bytes(data))
        filtered_ids = all_ordered_labels[real_selection_vectors.astype(bool)]
        filtered_train = train[all_ordered_labels[real_selection_vectors.astype(bool)]]
        generate_ground_truth(dataset_name, filtered_train, test, filtered_ids, selectivity_str, KNNS=(100,), normalize=normalize)


    for mode in SPECIAL_SELECTIVITIES:
        _type, param = mode.split(' ')
        n = int(param)
        selection_per_cluster = []
        selected = 0
        real_selection_vectors = np.array([]).astype(np.uint8)
        for l in labels:
            n_points = len(l)
            if n <= 0:
                real_selection_vector = np.full((n_points), 0, dtype=np.uint8)
                if '+' in _type:
                    real_selection_vector[random.randint(0, n_points - 1)] = 1
            else:
                real_selection_vector = np.full((n_points), 1, dtype=np.uint8)
                n -= 1
            selected_on_this_cluster = np.sum(real_selection_vector)
            selection_per_cluster.append(selected_on_this_cluster)
            real_selection_vectors = np.concatenate((real_selection_vectors, real_selection_vector))
            selected += selected_on_this_cluster
        selection_per_cluster = np.array(selection_per_cluster).astype(np.uint32)
        print(f'For selectivity {_type} {param}, {selected} points were chosen ({float(selected) / total_points * 100})')
        assert(len(selection_per_cluster) == len(labels))
        assert(sum(selection_per_cluster) == selected)
        assert(total_points == len(real_selection_vectors))

        selectivity_str = f'{_type}_{param}'
        selectivity_filename = f'{dataset_name}_{selectivity_str}.bin'
        selectivity_filename_path = os.path.join(FILTER_SELECTION_VECTORS, selectivity_filename)
        data = bytearray()
        data.extend(selection_per_cluster.tobytes("C"))
        data.extend(real_selection_vectors.tobytes("C"))
        with open(selectivity_filename_path, "wb") as file:
            file.write(bytes(data))
        filtered_ids = all_ordered_labels[real_selection_vectors.astype(bool)]
        filtered_train = train[all_ordered_labels[real_selection_vectors.astype(bool)]]
        generate_ground_truth(dataset_name, filtered_train, test, filtered_ids, selectivity_str, KNNS=(100,), normalize=normalize)


if __name__ == "__main__":
    # generate_selection_vector('agnews-mxbai-1024-euclidean', normalize=True)
    generate_selection_vector('instructorxl-arxiv-768', normalize=True)
