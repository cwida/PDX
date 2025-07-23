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


