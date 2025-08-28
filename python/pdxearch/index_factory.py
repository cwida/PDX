import numpy as np

from pdxearch.index_base import (
    BaseIndexPDXIVF, BaseIndexPDXIVF2, BaseIndexPDXFlat
)
from pdxearch.preprocessors import (
    ADSampling, Preprocessor
)
from pdxearch.compiled import (
    IndexADSamplingIVFFlat as _IndexPDXADSamplingIVFFlat,
    IndexBONDIVFFlat as _IndexPDXBONDIVFFlat,
    IndexBONDFlat as _IndexBONDFlat,
    IndexADSamplingFlat as _IndexADSamplingFlat,
    IndexPDXFlat as _IndexPDXFlat,
    IndexADSamplingIVF2SQ8 as _IndexADSamplingIVF2SQ8,
    IndexADSamplingIVF2Flat as _IndexADSamplingIVF2Flat
)
from pdxearch.constants import PDXConstants

from pdxearch.predicate_evaluator import PredicateEvaluator

#
# Python wrappers of the C++ lib
#

class IndexPDXIVF2(BaseIndexPDXIVF2):
    def __init__(
            self,
            *,
            ndim: int = 128,
            metric: str = "l2sq",
            nbuckets: int = 256,
            nbuckets_l0: int = 64,
            normalize: bool = True
    ) -> None:
        super().__init__(ndim, metric, nbuckets, nbuckets_l0, normalize)
        self.preprocessor = ADSampling(ndim)
        self.pdx_index = _IndexADSamplingIVF2Flat()
        self.pe = PredicateEvaluator()

    def preprocess(self, data, inplace: bool = True):
        return self.preprocessor.preprocess(data, inplace=inplace, normalize=self.normalize)

    # Used in Python API (TODO)
    def add_persist(self, data, path: str, matrix_path: str):
        self._add(data)
        self._train_add_l0()
        self._to_pdx(data, _type='pdx', use_original_centroids=True)
        self._persist(path)
        self.preprocessor.store_metadata(matrix_path)

    # Used in Python API (TODO)
    def persist(self, path: str, matrix_path: str):  # TODO: Rename
        self._persist(path)
        self.preprocessor.store_metadata(matrix_path)

    def add(self, data):
        self._add(data)
        self._train_add_l0()
        self._to_pdx(data, _type='pdx', use_original_centroids=True)
        self.pdx_index.load(self.materialized_index, self.preprocessor.transformation_matrix)

    # Used in Python API (TODO)
    def restore(self, path: str, matrix_path: str):
        self.pdx_index.restore(path, matrix_path)

    def search(self, q: np.ndarray, knn: int, nprobe: int = 16):
        return self.pdx_index.search(q, knn, nprobe)

    def evaluate_predicate(self, passing_tuples_ids):
        self.pe.evaluate_predicate(passing_tuples_ids, self.core_index.labels)

    def filtered_search(self, q: np.ndarray, knn: int, nprobe: int = 16):
        if len(self.pe.n_passing_tuples) == 0 or len(self.pe.selection_vector) == 0:
            raise ValueError("Call .evaluate_predicate([<passing_tuples>]) first to generate the selection_vector of your predicate")
        return self.pdx_index.filtered_search(q, knn, nprobe, self.pe.n_passing_tuples, self.pe.selection_vector)

    def set_pruning_confidence(self, confidence: float):
        self.pdx_index.set_pruning_confidence(confidence)


class IndexPDXIVF2SQ8(BaseIndexPDXIVF2):
    def __init__(
            self,
            *,
            ndim: int = 128,
            metric: str = "l2sq",
            nbuckets: int = 256,
            nbuckets_l0: int = 64,
            normalize: bool = True
    ) -> None:
        super().__init__(ndim, metric, nbuckets, nbuckets_l0, normalize)
        self.preprocessor = ADSampling(ndim)
        self.pdx_index = _IndexADSamplingIVF2SQ8()
        self.pe = PredicateEvaluator()

    def preprocess(self, data, inplace: bool = True):
        return self.preprocessor.preprocess(data, inplace=inplace, normalize=self.normalize)

    # Used in Python API (TODO)
    def add_persist(self, data, path: str, matrix_path: str):
        self._add(data)
        self._train_add_l0()
        self._to_pdx(data, _type='pdx-v4-h', quantize=True, use_original_centroids=True)
        self._persist(path)
        self.preprocessor.store_metadata(matrix_path)

    # Used in Python API (TODO)
    def persist(self, path: str, matrix_path: str):  # TODO: Rename
        self._persist(path)
        self.preprocessor.store_metadata(matrix_path)

    def add(self, data):
        self._add(data)
        self._train_add_l0()
        self._to_pdx(data, _type='pdx-v4-h', quantize=True, use_original_centroids=True)
        self.pdx_index.load(self.materialized_index, self.preprocessor.transformation_matrix)

    # Used in Python API (TODO)
    def restore(self, path: str, matrix_path: str):
        self.pdx_index.restore(path, matrix_path)

    def search(self, q: np.ndarray, knn: int, nprobe: int = 16):
        return self.pdx_index.search(q, knn, nprobe)

    def evaluate_predicate(self, passing_tuples_ids):
        self.pe.evaluate_predicate(passing_tuples_ids, self.core_index.labels)

    def filtered_search(self, q: np.ndarray, knn: int, nprobe: int = 16):
        if len(self.pe.n_passing_tuples) == 0 or len(self.pe.selection_vector) == 0:
            raise ValueError("Call .evaluate_predicate([<passing_tuples>]) first to generate the selection_vector of your predicate")
        return self.pdx_index.filtered_search(q, knn, nprobe, self.pe.n_passing_tuples, self.pe.selection_vector)

    def set_pruning_confidence(self, confidence: float):
        self.pdx_index.set_pruning_confidence(confidence)


class IndexPDXADSamplingIVFFlat(BaseIndexPDXIVF):
    def __init__(
            self,
            *, ndim: int = 128,
            metric: str = "l2sq",
            nbuckets: int = 256,
            normalize: bool = True
    ) -> None:
        super().__init__(ndim, metric, nbuckets, normalize)
        self.preprocessor = ADSampling(ndim)
        self.pdx_index = _IndexPDXADSamplingIVFFlat()
        self.pe = PredicateEvaluator()

    def preprocess(self, data, inplace: bool = True):
        return self.preprocessor.preprocess(data, inplace=inplace, normalize=self.normalize)

    def add_persist(self, data, path: str, matrix_path: str):
        self._add(data)
        # I don't need to pass the centroid preprocessor here, as the centroids are already rotated
        # Because the training was done on the transformed vectors
        self._to_pdx(data, _type='pdx', use_original_centroids=True)
        self._persist(path)
        self.preprocessor.store_metadata(matrix_path)

    def persist(self, path: str, matrix_path: str):  # TODO: Rename
        self._persist(path)
        self.preprocessor.store_metadata(matrix_path)

    def add(self, data):
        self._add(data)
        # I don't need to pass the centroid preprocessor here, as the centroids are already rotated
        # Because the training was done on the transformed vectors
        self._to_pdx(data, _type='pdx', use_original_centroids=True)
        self.pdx_index.load(self.materialized_index, self.preprocessor.transformation_matrix)

    def restore(self, path: str, matrix_path: str):
        self.pdx_index.restore(path, matrix_path)

    def search(self, q: np.ndarray, knn: int, nprobe: int = 16):
        return self.pdx_index.search(q, knn, nprobe)

    def evaluate_predicate(self, passing_tuples_ids):
        self.pe.evaluate_predicate(passing_tuples_ids, self.core_index.labels)

    def filtered_search(self, q: np.ndarray, knn: int, nprobe: int = 16):
        if len(self.pe.n_passing_tuples) == 0 or len(self.pe.selection_vector) == 0:
            raise ValueError("Call .evaluate_predicate([<passing_tuples>]) first to generate the selection_vector of your predicate")
        return self.pdx_index.filtered_search(q, knn, nprobe, self.pe.n_passing_tuples, self.pe.selection_vector)

    def set_pruning_confidence(self, confidence: float):
        self.pdx_index.set_pruning_confidence(confidence)

class IndexPDXBONDIVFFlat(BaseIndexPDXIVF):
    def __init__(
            self,
            *, ndim: int = 128,
            metric: str = "l2sq",
            nbuckets: int = 256,
            normalize: bool = True
    ) -> None:
        super().__init__(ndim, metric, nbuckets, normalize)
        self.preprocessor = Preprocessor()
        self.pdx_index = _IndexPDXBONDIVFFlat()
        self.pe = PredicateEvaluator()

    def preprocess(self, data, inplace: bool = True):
        return self.preprocessor.preprocess(data, inplace=inplace, normalize=self.normalize)

    def add_persist(self, data, path: str):
        self._add(data)
        self._to_pdx(data, _type='pdx', use_original_centroids=True, bond=True)
        self._persist(path)

    def persist(self, path: str):  # TODO: Rename
        self._persist(path)

    def add(self, data):
        self._add(data)
        self._to_pdx(data, _type='pdx', use_original_centroids=True, bond=True)
        self.pdx_index.load(self.materialized_index)

    def restore(self, path: str, matrix_path: str):
        self.pdx_index.restore(path, matrix_path)

    def search(self, q: np.ndarray, knn: int, nprobe: int = 16):
        return self.pdx_index.search(q, knn, nprobe)

    def evaluate_predicate(self, passing_tuples_ids):
        self.pe.evaluate_predicate(passing_tuples_ids, self.core_index.labels)

    def filtered_search(self, q: np.ndarray, knn: int, nprobe: int = 16):
        if len(self.pe.n_passing_tuples) == 0 or len(self.pe.selection_vector) == 0:
            raise ValueError("Call .evaluate_predicate([<passing_tuples>]) first to generate the selection_vector of your predicate")
        return self.pdx_index.filtered_search(q, knn, nprobe, self.pe.n_passing_tuples, self.pe.selection_vector)


class IndexPDXBONDFlat(BaseIndexPDXFlat):
    def __init__(
            self, *,
            ndim: int = 128,
            metric: str = "l2sq",
            normalize: bool = True
    ) -> None:
        super().__init__(ndim=ndim, metric=metric, normalize=normalize)
        self.preprocessor = Preprocessor()
        self.pdx_index = _IndexBONDFlat()
        self.block_sizes = PDXConstants.PDXEARCH_VECTOR_SIZE
        self.pe = PredicateEvaluator()

    def preprocess(self, data, inplace: bool = True):
        return self.preprocessor.preprocess(data, inplace=inplace, normalize=self.normalize)

    def add_persist(self, data, path: str):
        self._to_pdx(data, self.block_sizes, bond=True)
        self._persist(path)

    def persist(self, path: str):  # TODO: Rename
        self._persist(path)

    def add(self, data):
        self._to_pdx(data, self.block_sizes, bond=True)
        self.pdx_index.load(self.materialized_index)

    def restore(self, path: str):
        self.pdx_index.restore(path)

    def search(self, q: np.ndarray, knn: int):
        return self.pdx_index.search(q, knn)

    def evaluate_predicate(self, passing_tuples_ids):
        self.pe.evaluate_predicate(passing_tuples_ids, [x.indices for x in self.partitions])

    def filtered_search(self, q: np.ndarray, knn: int, nprobe: int = 16):
        if len(self.pe.n_passing_tuples) == 0 or len(self.pe.selection_vector) == 0:
            raise ValueError("Call .evaluate_predicate([<passing_tuples>]) first to generate the selection_vector of your predicate")
        return self.pdx_index.filtered_search(q, knn, self.pe.n_passing_tuples, self.pe.selection_vector)


class IndexPDXADSamplingFlat(BaseIndexPDXFlat):
    def __init__(self, *, ndim: int = 128, metric: str = "l2sq", normalize: bool = True) -> None:
        super().__init__(ndim=ndim, metric=metric, normalize=normalize)
        self.preprocessor = ADSampling(ndim)
        self.pdx_index = _IndexADSamplingFlat()
        self.block_sizes = PDXConstants.PDXEARCH_VECTOR_SIZE
        self.pe = PredicateEvaluator()

    def preprocess(self, data: np.array, inplace: bool = True):
        return self.preprocessor.preprocess(data, inplace=inplace, normalize=self.normalize)

    def add_persist(self, data: np.array, path: str, matrix_path: str):
        self._to_pdx(data, self.block_sizes)
        self._persist(path)
        self.preprocessor.store_metadata(matrix_path)

    def persist(self, path: str, matrix_path: str):  # TODO: Rename
        self._persist(path)
        self.preprocessor.store_metadata(matrix_path)

    def add(self, data):
        self._to_pdx(data, self.block_sizes)
        self.pdx_index.load(self.materialized_index, self.preprocessor.transformation_matrix)

    def restore(self, path: str, matrix_path: str):
        self.pdx_index.restore(path, matrix_path)

    def search(self, q: np.ndarray, knn: int):
        return self.pdx_index.search(q, knn)

    def set_pruning_confidence(self, confidence: float):
        self.pdx_index.set_pruning_confidence(confidence)

    def evaluate_predicate(self, passing_tuples_ids):
        self.pe.evaluate_predicate(passing_tuples_ids, [x.indices for x in self.partitions])

    def filtered_search(self, q: np.ndarray, knn: int, nprobe: int = 16):
        if len(self.pe.n_passing_tuples) == 0 or len(self.pe.selection_vector) == 0:
            raise ValueError("Call .evaluate_predicate([<passing_tuples>]) first to generate the selection_vector of your predicate")
        return self.pdx_index.filtered_search(q, knn, self.pe.n_passing_tuples, self.pe.selection_vector)


class IndexPDXFlat(BaseIndexPDXFlat):
    def __init__(self, *, ndim: int = 128, metric: str = "l2sq", normalize: bool = True) -> None:
        super().__init__(ndim, metric, normalize)
        self.pdx_index = _IndexPDXFlat()
        self.preprocessor = Preprocessor()
        self.block_sizes = PDXConstants.PDX_VECTOR_SIZE

    def add_persist(self, data, path: str):
        self._to_pdx(data, self.block_sizes, bond=True)
        self._persist(path)

    def preprocess(self, data, inplace: bool = True):
        return self.preprocessor.preprocess(data, inplace=inplace, normalize=self.normalize)

    def persist(self, path: str):  # TODO: Rename
        self._persist(path)

    def add(self, data):
        self._to_pdx(data, self.block_sizes, bond=True)
        self.pdx_index.load(self.materialized_index)

    def restore(self, path: str):
        self.pdx_index.restore(path)

    def search(self, q: np.ndarray, knn: int):
        return self.pdx_index.search(q, knn)