import json
from WrapperBruteForce import BruteForceSKLearn
from setup_utils import *
from setup_settings import *
from sklearn import preprocessing

# Generates ground truth with SKLearn
def generate_ground_truth(dataset, KNNS=(10, 100), normalize=False):
    if not os.path.exists(GROUND_TRUTH_DATA):
        os.makedirs(GROUND_TRUTH_DATA)
    train, test = read_hdf5_data(dataset)
    N_QUERIES = len(test)
    # test = test[:N_QUERIES]
    print('N. Queries', N_QUERIES)

    if normalize:
        train = preprocessing.normalize(train, axis=1, norm='l2')
        test = preprocessing.normalize(test, axis=1, norm='l2')

    algo = BruteForceSKLearn("euclidean", njobs=-1)
    algo.fit(train)
    for knn in KNNS:
        gt_filename = get_ground_truth_filename(dataset, knn, normalize)
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
    ks = [100]
    # generate_ground_truth('word2vec-300', ks, normalize=True)

    # generate_ground_truth('gooaq-distilroberta-768-normalized', ks, normalize=True)
    # generate_ground_truth('agnews-mxbai-1024-euclidean', ks, normalize=True)
    # generate_ground_truth('coco-nomic-768-normalized', ks, normalize=True)
    # generate_ground_truth('simplewiki-openai-3072-normalized', ks, normalize=True)

    # generate_ground_truth('imagenet-align-640-normalized', ks, normalize=True)
    # generate_ground_truth('yandex-200-cosine', ks, normalize=True)
    # generate_ground_truth('imagenet-clip-512-normalized', ks, normalize=True)
    # generate_ground_truth('laion-clip-512-normalized', ks, normalize=True)
    # generate_ground_truth('codesearchnet-jina-768-cosine', ks, normalize=True)
    # generate_ground_truth('yi-128-ip', ks, normalize=True)
    # generate_ground_truth('landmark-dino-768-cosine', ks, normalize=True)
    # generate_ground_truth('landmark-nomic-768-normalized', ks, normalize=True)
    # generate_ground_truth('arxiv-nomic-768-normalized', ks, normalize=True)
    # generate_ground_truth('ccnews-nomic-768-normalized', ks, normalize=True)
    # generate_ground_truth('celeba-resnet-2048-cosine', ks, normalize=True)
    # generate_ground_truth('llama-128-ip', ks, normalize=True)
    generate_ground_truth('yahoo-minilm-384-normalized', ks, normalize=True)

    # generate_ground_truth('openai-1536-angular', ks)
    # generate_ground_truth('msong-420', ks)
    # generate_ground_truth('instructorxl-arxiv-768', ks)
    # generate_ground_truth('sift-128-euclidean', ks)
    # generate_ground_truth('gist-960-euclidean', ks)
    #
    # generate_ground_truth('glove-200-angular', ks)
    #
    # generate_ground_truth('contriever-768', ks)

