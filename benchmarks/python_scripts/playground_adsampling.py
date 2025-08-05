import json
from WrapperBruteForce import BruteForceSKLearn
from setup_utils import *
from setup_settings import *
from sklearn import preprocessing
import math
import time
import torch
from hadamard_transform import randomized_hadamard_transform, pad_to_power_of_2, is_a_power_of_2
import numpy.random as npr

from scipy.fft import dct

import math
import heapq
import numpy
from typing import Optional, List

import json


"""
This is a playground for random rotation algorithms and ADSampling pruning
IT IS VERY SLOW. It is only intended to quickly test correctness.

"""

class PruningProfile:

    def __init__(self):
        self.abs_profile = {}
        self.perc_profile = {}

    def fill_profile(self, D, steps):
        for i in range(0, D, steps):
            self.abs_profile[i] = 0
            self.perc_profile[i] = 0
        if D-1 not in self.abs_profile:
            self.abs_profile[D-1] = 0
            self.perc_profile[D-1] = 0

    def fill_perc_from_abs(self, total):
        for d, count in self.abs_profile.items():
            self.perc_profile[d] = float(count) / total * 100

    def __str__(self):
        return f'Absolute pruning:({json.dumps(self.abs_profile, indent=2)})\nPercentage:({json.dumps(self.perc_profile, indent=2)})'

class SearchResult:
    def __init__(self, idx: int, distance: float):
        self.idx = idx
        self.distance = -distance # Todo: this is dumb

    def __lt__(self, other):
        return self.distance < other.distance

    def __eq__(self, other):
        return self.idx == other.idx  # Todo: Needs to be fixed with real indexes

    def __str__(self):
        return f"({self.idx}, {self.distance:.4f})"

    def __repr__(self):
        return str(self)

    @staticmethod
    def build_results(points: numpy.array, distances: numpy.array):
        return [SearchResult(point, distance) for point, distance in zip(points, distances)]

    @staticmethod
    def build_result(point: numpy.array, distance: numpy.array):
        return [SearchResult(point, distance)]



class SearchState:
    def __init__(self, knn: int, avoided_values: Optional[int] = 0, pruned_profile: Optional[PruningProfile] = PruningProfile(),
                 avoided_values_percentage: Optional[float] = 0.0):
        self.knn_heap: List[SearchResult] = []
        self.heap_size = knn
        self.avoided_values: float = avoided_values
        self.pruned_profile: PruningProfile = pruned_profile
        self.avoided_values_percentage: float = avoided_values_percentage
        self.vectors_not_pruned : int = 0
        self.per_partition_pruning : dict = {}
        self.per_technique_pruning: dict = {}
        self.stopped_at: list[int] = []
        self.hot_dimensions: list[int] = []
        self.pruning_through_time: list[int] = []
        heapq.heapify(self.knn_heap)

    def get_knn(self) -> List[SearchResult]:
        return heapq.nsmallest(self.heap_size, self.knn_heap)

    def get_worst(self) -> SearchResult: # Should be get_best (negative sign shittery)
        return min(self.knn_heap) # More efficient than n_largest when n == 1

    def get_worst_distance(self) -> float: # Should be get_best_distance (negative sign shittery)
        return -min(self.knn_heap).distance # More efficient than n_largest when n == 1

    def get_best(self) -> SearchResult: # Should be get_worst (negative shittery)
        return max(self.knn_heap) # More efficient than n_smallest when n == 1

    def merge_search(self, search_result_list: List[SearchResult]) -> None:
        for search_result in search_result_list:
            if len(self.knn_heap) < self.heap_size:
                heapq.heappush(self.knn_heap, search_result)
            else:
                heapq.heappushpop(self.knn_heap, search_result)

    def __str__(self):
        return f"SearchState(Avoided Values {self.avoided_values_percentage:.4f}%)"

class ADSampling:
    def __init__(self, query: numpy.array, dimensions: int, knn : int = 10, prune_stats: bool=False):
        self.q = query
        self.original_query = query
        self.knn = knn
        self.dimensions = dimensions
        self.prune_stats = prune_stats
        self.grouping = 32

        self.epsilon0 = 1.5


    def ratio(self, i):
        D = self.dimensions
        if i == D: return 1
        return (1.0 * i) / D * (1.0 + self.epsilon0 / math.sqrt(i)) *(1.0 + self.epsilon0 / math.sqrt(i))

    def search(self, embeddings) -> SearchState:
        values_avoided = 0
        local_search_state = SearchState(self.knn)
        pruned_profile = PruningProfile()

        threshold = math.inf
        n_points = embeddings.shape[0]
        accum_last_checked_dim = 0.0
        pruning_through_time = numpy.zeros(n_points)

        # if __debug__: print(f'ADSampling=(len(v)={n_points}, D={self.dimensions}, e={self.epsilon0}, d_delta={self.grouping}')

        # for embedding in numpy.nditer(embeddings, flags=['external_loop'], order='C'):
        # print(n_points, self.dimensions, self.epsilon0, self.grouping)
        for v in range(n_points):
            embedding = embeddings[v]
            cur_distance = 0.0
            i = 0
            distance = None
            counted_pruning = False
            while i < self.dimensions:
                check = min(self.grouping, self.dimensions - i)
                tmp = embedding[i:i+check] - self.q[i:i+check]
                cur_distance += np.sum(tmp * tmp)
                # for j in range(i, i + check):
                #     to_multiply = embedding[j] - self.q[j]
                #     cur_distance += to_multiply * to_multiply # L2
                # if v == 0:
                #     print(cur_distance)
                i += check
                if cur_distance >= threshold * self.ratio(i):
                    distance = -cur_distance * self.dimensions / i
                    values_avoided += self.dimensions - i
                    pruned_profile.abs_profile[i-1] = pruned_profile.abs_profile.get(i-1, 0) + 1
                    pruning_through_time[v] = i
                    counted_pruning = True
                    break
                distance = cur_distance
            if not counted_pruning: # Went till last value
                accum_last_checked_dim += i - 1
                pruning_through_time[v] = i
                pruned_profile.abs_profile[i-1] = pruned_profile.abs_profile.get(i-1, 0) + 1
            if distance >= 0:
                local_search_state.merge_search(SearchResult.build_result(v, distance))
            if (len(local_search_state.knn_heap) == self.knn and
                    local_search_state.get_worst_distance() < threshold):
                threshold = local_search_state.get_worst_distance()

        pruned_profile.fill_perc_from_abs(n_points)
        local_search_state.avoided_values = values_avoided
        total_values = n_points * self.dimensions
        local_search_state.avoided_values_percentage = (local_search_state.avoided_values / total_values) * 100
        local_search_state.pruned_profile = pruned_profile
        local_search_state.stopped_at.append(float(accum_last_checked_dim) / n_points / self.dimensions * 100)
        local_search_state.pruning_through_time = pruning_through_time

        return local_search_state

def hadamard(Y):
    n_vectors, n_Y = Y.shape
    matching = (n_vectors, 2, int(n_Y / 2))
    H = np.array([[1, 1], [1, -1]])  # Julia uses 2 and not sqrt(2)?
    steps = int(np.log(n_Y) / np.log(2))
    assert(2**steps == n_Y)  # required
    for _ in range(steps):
        Y = np.transpose(Y.reshape(matching), (0, 2, 1)).dot(H)
    Y = Y.reshape((n_vectors, n_Y))
    return Y

def prev_power_of_2(x):
    return 2 ** (x.bit_length() - 1)

def next_power_of_2(x):
    return 1 if x == 0 else 2**(x - 1).bit_length()

def kak_walk(X, d):
    assert (d == X.shape[1])
    mid = d // 2
    first = X[:, :mid].copy()
    second = X[:, mid:].copy()
    X[:, :mid] = first + second
    X[:, mid:] = first - second

def ffht_kak_walk(X, rounds=4):
    n, d = X.shape
    d_pad = next_power_of_2(d)
    trunc_dim = prev_power_of_2(d)  # Full padded length
    correcting_fac = 1.0 / math.sqrt(float(trunc_dim))

    # Pad input if necessary
    if d_pad > d:
        X_padded = np.zeros((n, d_pad))
        X_padded[:, :d] = X.copy()
    else:
        X_padded = X.copy()
    X_rot = X_padded

    print('Starting')
    # Pre-sample random sign flip vectors D_1 to D_4 (shape: (rounds, d))
    sign_flips = np.random.choice([-1.0, 1.0], size=(rounds, d_pad))
    if trunc_dim == d:
        X_rot *= sign_flips[0]
        X_rot = hadamard(X_rot)
        X_rot *= correcting_fac

        X_rot *= sign_flips[1]
        X_rot = hadamard(X_rot)
        X_rot *= correcting_fac

        X_rot *= sign_flips[2]
        X_rot = hadamard(X_rot)
        X_rot *= correcting_fac

        X_rot *= sign_flips[3]
        X_rot = hadamard(X_rot)
        X_rot *= correcting_fac
        return X_rot

    start = d_pad - trunc_dim
    print('Start', start)
    print('D pad', d_pad)
    print('Trunc dim', trunc_dim)
    X_rot *= sign_flips[0]
    X_rot[:, :start] = hadamard(X_rot[:, :start])
    X_rot[:, :trunc_dim] *= correcting_fac
    kak_walk(X_rot, d_pad)

    X_rot *= sign_flips[1]
    X_rot[:, -start:] = hadamard(X_rot[:, -start:])
    X_rot[:, -trunc_dim:] *= correcting_fac
    kak_walk(X_rot, d_pad)

    X_rot *= sign_flips[2]
    X_rot[:, :start] = hadamard(X_rot[:, :start])
    X_rot[:, :trunc_dim] *= correcting_fac
    kak_walk(X_rot, d_pad)

    X_rot *= sign_flips[3]
    X_rot[:, -start:] = hadamard(X_rot[:, -start:])
    X_rot[:, -trunc_dim:] *= correcting_fac
    kak_walk(X_rot, d_pad)

    X_rot *= 0.25
    return X_rot[:, :]


def rotate_baseline(A, transformation_matrix):
    return np.dot(
        A, transformation_matrix)


def fast_random_orthogonal_rotation(X, ones):
    n = X.shape[1]
    # Step 1: Random sign flipping
    X = X * ones

    # Step 2: Fast orthogonal transform — DCT instead of Hadamard
    X = dct(X, norm='ortho', axis=1)
    return X


def one_transform_bench(vector):
    d = len(vector)
    D = np.random.choice([1, -1], size=1)
    transformation_matrix, _ = np.linalg.qr(np.random.randn(d, d).astype(np.float32))
    start = time.time()
    rep = 1
    for i in range(rep):
        np.dot(vector, transformation_matrix)
    print ('JLT', (time.time() - start) * 1000 / rep)
    start = time.time()
    for i in range(rep):
        dct(vector * D, type=2, norm='ortho')
    print ('DCT', (time.time() - start) * 1000 / rep)


if __name__ == "__main__":
    # dataset = 'sift-128-euclidean'
    # dataset = 'contriever-768'
    # dataset = 'openai-1536-angular'
    # dataset = 'agnews-mxbai-1024-euclidean'
    # dataset = 'arxiv-nomic-768-normalized'
    # dataset = 'fashion-mnist-784-euclidean'
    # dataset = 'simplewiki-openai-3072-normalized'
    # dataset = 'ccnews-nomic-768-normalized'
    # dataset = 'celeba-resnet-2048-cosine'
    # dataset = 'codesearchnet-jina-768-cosine'
    # dataset = 'gist-960-euclidean'
    # dataset = 'gooaq-distilroberta-768-normalized'
    # dataset = 'imagenet-align-640-normalized'
    # dataset = 'imagenet-clip-512-normalized'
    # dataset = 'laion-clip-512-normalized'
    # dataset = 'llama-128-ip'
    # dataset = 'yandex-200-cosine'
    dataset = 'yahoo-minilm-384-normalized'
    if not os.path.exists(GROUND_TRUTH_DATA):
        os.makedirs(GROUND_TRUTH_DATA)
    train, test = read_hdf5_data(dataset)
    # one_transform_bench(train[0])
    # exit()
    # train = train[:100000]
    N_QUERIES = 10
    KNN = 100
    # test = test[:N_QUERIES]

    d = len(test[0])
    D = np.random.choice([1, -1], size=d)
    transformation_matrix, _ = np.linalg.qr(np.random.randn(d, d).astype(np.float32))

    train = preprocessing.normalize(train, axis=1, norm='l2')

    train_ = rotate_baseline(train, transformation_matrix)
    test_ = rotate_baseline(test, transformation_matrix)
    algo = BruteForceSKLearn("euclidean", njobs=-1)
    algo.fit(train)

    dist, index = algo.query_batch(test[:N_QUERIES], n=KNN)
    # print('GT', '\n', index[0], '\n', dist[0])

    print('Rotating')
    # r = random_rotation_hadamard(train)

    r = fast_random_orthogonal_rotation(train, D)
    r_t = fast_random_orthogonal_rotation(test, D)

    # r = ffht_kak_walk(train)

    # r = np.zeros_like(train)
    # r[:, :512] = ffht_kak_walk(train[:, :512])
    # r[:, 512:] = ffht_kak_walk(train[:, 512:])

    # r = fjlt(train)

    # print(r)
    # print(r.shape)
    algo = BruteForceSKLearn("euclidean", njobs=-1)
    algo.fit(r)
    dist_1, index_1 = algo.query_batch(r_t[:N_QUERIES], n=KNN)
    # print('GT', '\n', index_1[0], '\n', dist_1[0])

    norms_before = np.linalg.norm(train, axis=1)
    norms_after = np.linalg.norm(r, axis=1)
    print("Norms close?", np.allclose(norms_before, norms_after, atol=1e-5))

    print('Maintains rankings?', np.sum(index_1[0] == index[0]) == KNN, np.sum(index_1[0] == index[0]))

    dct_recalls = []
    dct_avoided = []
    jlt_recalls = []
    jlt_avoided = []
    for i in range(N_QUERIES):
        print(f'Q_{i}')
        searcher1 = ADSampling(r_t[i], r.shape[1], KNN, True)
        res1 = searcher1.search(r)
        dct_res = np.array([x.idx for x in res1.knn_heap])
        dct_recall = np.intersect1d(index[i], dct_res)
        dct_avoided.append(res1.avoided_values_percentage)
        dct_recalls.append(dct_recall.size)
        # print('dct:', res1)

        searcher2 = ADSampling(test_[i], train_.shape[1], KNN,True)
        res2 = searcher2.search(train_)
        jlt_res = np.array([x.idx for x in res2.knn_heap])
        jlt_recall = np.intersect1d(index[i], jlt_res)
        jlt_avoided.append(res2.avoided_values_percentage)
        jlt_recalls.append(jlt_recall.size)
        # print('jlt:', res2)

    print(jlt_recalls)
    print('JLT recall vs GT: ', np.mean(np.array(jlt_recalls)))
    print('JLT avoidd vs GT: ', np.mean(np.array(jlt_avoided)))
    print()
    print(dct_recalls)
    print('DCT recall vs GT: ', np.mean(np.array(dct_recalls)))
    print('DCT avoidd vs GT: ', np.mean(np.array(dct_avoided)))