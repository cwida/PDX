import json
from WrapperBruteForce import BruteForceSKLearn
from setup_utils import *
from setup_settings import *

# Generates ground truth with SKLearn
def generate_ground_truth(dataset, KNNS=(10, 100)):
    if not os.path.exists(GROUND_TRUTH_DATA):
        os.makedirs(GROUND_TRUTH_DATA)
    train, test = read_hdf5_data(dataset)
    N_QUERIES = len(test)
    print('N. Queries', N_QUERIES)
    algo = BruteForceSKLearn("euclidean", njobs=-1)
    algo.fit(train)
    for knn in KNNS:
        gt_filename = get_ground_truth_filename(dataset, knn)
        gt_name = os.path.join(SEMANTIC_GROUND_TRUTH_PATH, gt_filename)
        gt = {}
        index_data = []
        distance_data = []
        print('Querying for GT...')
        dist, index = algo.query_batch(test, n=knn)
        for i in range(N_QUERIES):
            index_data.append(index[i])
            distance_data.append(dist[i] ** 2)
            gt[i] = index[i].tolist()
        with open(os.path.join(GROUND_TRUTH_DATA, gt_filename.replace('.json', '')), "wb") as file:
            file.write(np.array(index_data, dtype=np.uint32).tobytes("C"))
            file.write(np.array(distance_data, dtype=np.float32).tobytes("C"))
        with open(gt_name, 'w') as f:
            json.dump(gt, f)


if __name__ == "__main__":
    ks = [10]
    generate_ground_truth('fashion-mnist-784-euclidean', ks)
