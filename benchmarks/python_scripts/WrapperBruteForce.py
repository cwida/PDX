import os
import faiss
import numpy
numpy.random.seed(42)
import sklearn.neighbors
from usearch.index import search as usearch_search
from usearch.index import MetricKind
from pymilvus import MilvusClient


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


class BruteForceMilvus:
    def __init__(self, metric, dimension, dataset):
        if metric not in ("euclidean"):
            raise NotImplementedError("BruteForce doesn't support metric %s" % metric)
        self.metric = metric

        self._metric = metric
        self.name = "BruteForceMILVUS()"
        self._nbrs = None
        self.dimension = dimension

        self.client = MilvusClient("milvus_pdxearch.db")
        self.dataset = dataset.replace("-", "_")
        self.just_created = False

        # If the collection doesn't exist, we must create it
        if not self.client.has_collection(collection_name=self.dataset):
            self.client.drop_collection(collection_name=self.dataset)
            self.client.create_collection(
                collection_name=self.dataset,
                metric_type="L2",  # TODO: Support other metrics
                dimension=self.dimension,
            )
            self.just_created = True

    def fit(self, X):
        if self.just_created:  # Batch insert if it's a new collection
            print('Generating')
            data = [
                {"id": i, "vector": X[i]}
                for i in range(len(X))
            ]
            print('Inserting')
            i = 0
            for _ in range(0, len(data), 1000):
                print(_)
                if i + 1000 > len(data):
                    break
                self.client.insert(collection_name=self.dataset, data=data[i: i+1000])
                i += 1000
            if i != len(data):
                self.client.insert(collection_name=self.dataset, data=data[i: len(data)])
        self.client.load_collection(self.dataset)

    def query(self, v, n):
        results = self.client.search(
            collection_name=self.dataset,
            data=v,
            search_params={"metric_type": "L2", "params": {}},
            limit=n
        )
        return results

    def release(self):
        self.client.release_collection(
            collection_name=self.dataset
        )


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