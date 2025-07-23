import os
import faiss
import numpy
numpy.random.seed(42)
import sklearn.neighbors
from usearch.index import search as usearch_search
from usearch.index import MetricKind


class BruteForceUsearch:
    def __init__(self, metric):
        if metric not in ("angular", "euclidean", "hamming"):
            raise NotImplementedError("BruteForce doesn't support metric %s" % metric)
        self._metric = metric
        self.name = "BruteForceSIMD()"
        self._nbrs = None

    def query(self, data, v, n):
        matches = usearch_search(data, v, n, MetricKind.L2sq, exact=True, threads=1)
        return matches


class BruteForceFAISS:
    def __init__(self, metric, dimension):
        if metric not in ("angular", "euclidean", "hamming", "ip"):
            raise NotImplementedError("BruteForce doesn't support metric %s" % metric)
        self.metric = metric
        if metric == "euclidean":
            self._metric = faiss.METRIC_L2
        elif metric == "ip":
            self._metric = faiss.METRIC_INNER_PRODUCT
        else:
            self._metric = faiss.METRIC_L2

        self._metric = metric
        self.name = "BruteForceSIMD()"
        self._nbrs = None
        self.dimension = dimension
        faiss.omp_set_num_threads(1)

    def fit(self, X):
        if self.metric == "euclidean":
            self.index = faiss.IndexFlatL2(self.dimension)
        elif self.metric == "ip":
            self.index = faiss.IndexFlatIP(self.dimension)
        else:
            self.index = faiss.IndexFlatL2(self.dimension)
        self.index.add(X)

    def query(self, v, n, X=None, force_fit=True):
        if force_fit and X is not None: self.fit(X)
        points, distances = self.index.search(numpy.array([v]), k=n)
        return points, distances

class BruteForceSKLearn:
    def __init__(self, metric, njobs=1):
        if metric not in ("angular", "euclidean", "hamming"):
            raise NotImplementedError("BruteForce doesn't support metric %s" % metric)
        self._metric = metric
        self.metric = {"angular": "cosine", "euclidean": "l2", "hamming": "hamming"}[self._metric]
        self.name = "BruteForce()"
        self._nbrs = sklearn.neighbors.NearestNeighbors(algorithm="brute", metric=self.metric, n_jobs=njobs)

    def fit(self, X):
        self._nbrs.fit(X)

    def query(self, v, n, X=None, force_fit=True):
        if force_fit and X is not None: self._nbrs.fit(X)
        return list(self._nbrs.kneighbors([v], return_distance=True, n_neighbors=n))

    def query_batch(self, v, n, X=None, force_fit=True):
        if force_fit and X is not None: self._nbrs.fit(X)
        return list(self._nbrs.kneighbors(v, return_distance=True, n_neighbors=n))