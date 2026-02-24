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
    def __init__(self, metric, njobs=-1):
        if metric not in ("angular", "euclidean", "ip"):
            raise NotImplementedError("BruteForce doesn't support metric %s" % metric)
        self.metric = metric
        self.name = "BruteForceFAISS()"

    def fit(self, X):
        X = numpy.ascontiguousarray(X, dtype=numpy.float32)
        dimension = X.shape[1]
        if self.metric == "ip":
            self.index = faiss.IndexFlatIP(dimension)
        else:
            self.index = faiss.IndexFlatL2(dimension)
        self.index.add(X)

    def query(self, v, n):
        v = numpy.ascontiguousarray(numpy.array([v]), dtype=numpy.float32)
        distances, indices = self.index.search(v, k=n)
        return distances[0], indices[0]

    def query_batch(self, v, n):
        """Batch query. Returns (distances, indices).
        Note: for L2, distances are squared L2 (unlike sklearn which returns L2)."""
        v = numpy.ascontiguousarray(v, dtype=numpy.float32)
        distances, indices = self.index.search(v, k=n)
        return distances, indices

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