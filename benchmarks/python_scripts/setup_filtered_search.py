import numpy as np
import json
import os
from decimal import Decimal
from setup_utils import *
from sklearn import preprocessing
from WrapperBruteForce import BruteForceFAISS

np.random.seed(42)

SELECTIVITIES = [
    0.0001, 0.000135, 0.00015, 0.001, 0.01,
    0.1, 0.2, 0.3, 0.4, 0.5, 0.75,
    0.9, 0.95, 0.99,
]

KNN = 100

def generate_filtered_ground_truth(hdf5_name, train, test, passing_row_ids, selectivity_str):
    """Brute-force kNN on filtered subset; write binary GT + JSON GT."""
    filtered_train = train[passing_row_ids]
    actual_knn = min(KNN, len(filtered_train))
    if actual_knn == 0:
        print(f'  WARNING: 0 passing points for {selectivity_str}, skipping GT')
        return

    print(f'  GT: {len(filtered_train)} filtered points, {len(test)} queries')
    algo = BruteForceFAISS("euclidean")
    algo.fit(filtered_train)

    dist, index = algo.query_batch(test, n=actual_knn)

    index_data = []
    distance_data = []
    gt = {}
    for i in range(len(test)):
        original_ids = passing_row_ids[index[i]]
        # Pad to KNN for consistent binary format
        padded_ids = np.full(KNN, 0xFFFFFFFF, dtype=np.uint32)
        padded_ids[:actual_knn] = original_ids
        padded_dists = np.full(KNN, np.finfo(np.float32).max, dtype=np.float32)
        padded_dists[:actual_knn] = dist[i]
        index_data.append(padded_ids)
        distance_data.append(padded_dists)
        gt[i] = original_ids.tolist()

    gt_filename = f"{hdf5_name}_{KNN}_norm_{selectivity_str}"
    with open(os.path.join(FILTERED_GROUND_TRUTH_DATA, gt_filename), "wb") as f:
        f.write(np.array(index_data, dtype=np.uint32).tobytes("C"))
        f.write(np.array(distance_data, dtype=np.float32).tobytes("C"))
    with open(os.path.join(SEMANTIC_FILTERED_GROUND_TRUTH_PATH, gt_filename + ".json"), 'w') as f:
        json.dump(gt, f)


def generate_filtered_data(dataset_abbrev, normalize=True):
    """For each selectivity, generate passing row IDs + filtered ground truth."""
    hdf5_name, dims = DATASET_INFO[dataset_abbrev]
    print(f'\n{dataset_abbrev} -> {hdf5_name}')

    train, test = read_hdf5_data(hdf5_name)
    if normalize:
        train = preprocessing.normalize(train, axis=1, norm='l2')
        test = preprocessing.normalize(test, axis=1, norm='l2')

    total_points = len(train)
    print(f'Total points: {total_points}')

    for selectivity in SELECTIVITIES:
        mask = np.random.rand(total_points) < selectivity
        passing_row_ids = np.where(mask)[0].astype(np.uint32)
        num_passing = len(passing_row_ids)
        print(f'  Selectivity {selectivity}: {num_passing} points ({num_passing / total_points:.6f})')

        selectivity_str = format(Decimal(str(selectivity)), 'f').replace('.', '_')

        # Write passing row IDs as binary: [uint32 count][uint32[] ids]
        filename = f'{hdf5_name}_{selectivity_str}.bin'
        filepath = os.path.join(FILTER_SELECTION_VECTORS, filename)
        with open(filepath, "wb") as f:
            f.write(np.uint32(num_passing).tobytes())
            f.write(passing_row_ids.tobytes("C"))

        generate_filtered_ground_truth(hdf5_name, train, test, passing_row_ids, selectivity_str)


if __name__ == "__main__":
    generate_filtered_data('mxbai', normalize=True)
