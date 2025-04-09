import numpy as np
np.random.seed(42)
from abc import ABC, abstractmethod
from sklearn.decomposition import PCA

#
# Preprocessors (ADSampling and BSA)
# These classes includes the methods for data transformation and storing the respective metadata
#


class Preprocessor:
    def preprocess(self, data: np.array, inplace=False, normalize=True):
        if inplace:
            data[:] = self.normalize(data) if normalize else data
        else:
            return self.normalize(data) if normalize else data

    def store_metadata(self, *args):
        pass

    def normalize(self, data):
        norms = np.linalg.norm(data, axis=1, keepdims=True)
        return data/norms


class ADSampling(Preprocessor):
    def __init__(
            self,
            ndim
    ):
        self.transformation_matrix, _ = np.linalg.qr(np.random.randn(ndim, ndim).astype(np.float32))

    def preprocess(self, data: np.array, inplace=False, normalize=True):
        if inplace:
            data[:] = np.dot(
                self.normalize(data) if normalize else data,
                self.transformation_matrix)
        else:
            return np.dot(
                self.normalize(data) if normalize else data,
                self.transformation_matrix)

    def store_metadata(self, path: str):
        with open(path, "wb") as file:
            file.write(self.transformation_matrix.tobytes("C"))


class BSA(Preprocessor):
    def __init__(
            self,
            ndim
    ):
        self.pca_dim = ndim
        self.pca = PCA(n_components=self.pca_dim)
        self.projection_matrix = np.array([])
        self.base_square = np.array([])
        self.variances = np.array([])
        self.mean = np.array([])

    def preprocess(self, data: np.array, inplace=False, normalize=True):
        N = len(data)
        if normalize:
            data = self.normalize(data)
        self.mean = np.mean(data, axis=0)
        data -= self.mean
        print('Fitting PCA')
        if N < 10000000:
            self.pca.fit(data)
        else:
            self.pca.fit(data[:10000000])
        print('Transforming')
        self.projection_matrix = self.pca.components_.T
        data = np.dot(data, self.projection_matrix)
        self.base_square = np.sum((np.subtract(data, self.mean)) ** 2, axis=1)
        self.variances = data.var(axis=0)
        return data

    def store_metadata(
            self,
            matrix_path: str,
            variances_path: str,
            means_path: str,
            base_square_path: str
    ):
        with open(matrix_path, "wb") as file:
            file.write(self.projection_matrix.tobytes("C"))

        with open(variances_path, "wb") as file:
            file.write(self.variances.tobytes("C"))

        with open(means_path, "wb") as file:
            file.write(self.mean.tobytes("C"))

        with open(base_square_path, "wb") as file:
            file.write(self.base_square.tobytes("C"))
