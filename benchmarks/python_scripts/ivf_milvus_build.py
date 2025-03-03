import sys
import numpy as np
from numpy.random import default_rng
from setup_utils import *
from setup_settings import *
from benchmark_utils import *
from WrapperMilvus import *

np.random.seed(42)

if __name__ == '__main__':
    arg_dataset = ""
    if len(sys.argv) > 1:
        arg_dataset = sys.argv[1]
    for dataset in DATASETS:
        if len(arg_dataset) and dataset != arg_dataset:
            continue
        print('Milvus:', dataset, '====================================')
        runtimes = []
        recalls = []
        clock = TicToc()
        dimensionality = DIMENSIONALITIES[dataset]
        gt_name = os.path.join(SEMANTIC_GROUND_TRUTH_PATH, get_ground_truth_filename(dataset, KNN))
        searcher = IVFMilvus("euclidean", dimensionality, dataset, force_create=True)

        print('Reading data of', dataset)
        data, _ = read_hdf5_data(dataset)
        data = np.ascontiguousarray(data)
        num_embeddings = len(data)
        print('Setting up IVF with all vectors')
        # Generate IVF partitions
        # Milvus recommendation is 4 * SQRT of number of embeddings
        if num_embeddings < 500_000:  # However if collection is too small we better use only 1 * SQRT(n)
            min_num_partitions = math.ceil(1 * math.sqrt(num_embeddings))
        elif num_embeddings < 2_500_000:
            min_num_partitions = math.ceil(2 * math.sqrt(num_embeddings))
        else:  # For Deep which is bigger than normal
            min_num_partitions = math.ceil(4 * math.sqrt(num_embeddings))
        training_points = min_num_partitions * 50
        if training_points < num_embeddings:
            rng = default_rng()
            training_sample_idxs = rng.choice(num_embeddings, size=training_points, replace=False)
            training_sample_idxs.sort()
            print('Training with', training_points)
            searcher.fit(data[training_sample_idxs])
            rest_data = np.delete(data, training_sample_idxs, axis=0)
            searcher.add(rest_data)
        else:
            print('Training with all points')
            searcher.fit(data)

        print('Training and Building MILVUS IVF_FLAT with:', min_num_partitions, 'buckets')
        searcher.fit(data)
        searcher.create_index(min_num_partitions)
        print('Index created')
        searcher.release()
