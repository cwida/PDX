import numpy as np
from scipy.fft import dct
np.random.seed(42)

from pdxearch.constants import PDXConstants
from abc import ABC, abstractmethod

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
        norms[norms == 0] = 1
        return data/norms


class ADSampling(Preprocessor):
    def __init__(
            self,
            ndim
    ):
        self.d = ndim
        if PDXConstants.HAS_FFTW and self.d >= PDXConstants.D_THRESHOLD_FOR_DCT_ROTATION:
            self.transformation_matrix = np.random.choice([-1.0, 1.0], size=(ndim)).astype(np.float32)
        else:
            self.transformation_matrix, _ = np.linalg.qr(np.random.randn(ndim, ndim).astype(np.float32))

    def fjlt(self, X):
        n, _ = X.shape
        X = X * self.transformation_matrix
        X = dct(X, norm='ortho', axis=1)
        return X

    def preprocess(self, data: np.array, inplace=False, normalize=True):
        if PDXConstants.HAS_FFTW and self.d >= PDXConstants.D_THRESHOLD_FOR_DCT_ROTATION:
            if inplace:
                data[:] = self.fjlt(self.normalize(data) if normalize else data)
            else:
                return self.fjlt(self.normalize(data) if normalize else data)
        else:
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
