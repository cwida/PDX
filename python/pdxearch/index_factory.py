import numpy as np

from pdxearch.compiled import PDXIndex as _PDXIndex, load_index as _load_index
# from pdxearch.predicate_evaluator import PredicateEvaluator

METRIC_MAP = {"l2sq": 0, "cosine": 1, "ip": 2}
DEFAULT_NPROBE = 32


class IndexPDXIVF:
    """Single-level IVF index (F32)."""

    def __init__(
        self,
        *,
        num_dimensions: int,
        distance_metric: str = "l2sq",
        normalize: bool = True,
        seed: int = 42,
        num_clusters: int = 0,
        sampling_fraction: float = 0.0,
        kmeans_iters: int = 10,
        hierarchical_indexing: bool = True,
        n_threads: int = 0,
    ) -> None:
        self._index = _PDXIndex(
            "pdx_f32", num_dimensions, METRIC_MAP[distance_metric],
            seed, num_clusters, 0, normalize, sampling_fraction, kmeans_iters,
            hierarchical_indexing, n_threads,
        )
        # self.pe = PredicateEvaluator()

    def build(self, data: np.ndarray) -> None:
        self._index.build_index(np.ascontiguousarray(data, dtype=np.float32))

    def search(self, query: np.ndarray, knn: int, nprobe: int = DEFAULT_NPROBE):
        self._index.set_nprobe(nprobe)
        return self._index.search(np.ascontiguousarray(query, dtype=np.float32), knn)

    def filtered_search(self, query: np.ndarray, knn: int, row_ids: np.ndarray, nprobe: int = DEFAULT_NPROBE):
        self._index.set_nprobe(nprobe)
        return self._index.filtered_search(
            np.ascontiguousarray(query, dtype=np.float32), knn,
            np.ascontiguousarray(row_ids, dtype=np.uint64),
        )

    def append(self, row_id: int, embedding: np.ndarray) -> None:
        self._index.append(row_id, np.ascontiguousarray(embedding, dtype=np.float32))

    def delete(self, row_id: int) -> None:
        self._index.delete(row_id)

    def save(self, path: str) -> None:
        self._index.save(path)

    @property
    def num_dimensions(self) -> int:
        return self._index.get_num_dimensions()

    @property
    def num_clusters(self) -> int:
        return self._index.get_num_clusters()

    @property
    def in_memory_size_bytes(self) -> int:
        return self._index.get_in_memory_size_in_bytes()


class IndexPDXIVFSQ8:
    """Single-level IVF index (U8 scalar quantization)."""

    def __init__(
        self,
        *,
        num_dimensions: int,
        distance_metric: str = "l2sq",
        normalize: bool = True,
        seed: int = 42,
        num_clusters: int = 0,
        sampling_fraction: float = 0.0,
        kmeans_iters: int = 10,
        hierarchical_indexing: bool = True,
        n_threads: int = 0,
    ) -> None:
        self._index = _PDXIndex(
            "pdx_u8", num_dimensions, METRIC_MAP[distance_metric],
            seed, num_clusters, 0, normalize, sampling_fraction, kmeans_iters,
            hierarchical_indexing, n_threads,
        )
        # self.pe = PredicateEvaluator()

    def build(self, data: np.ndarray) -> None:
        self._index.build_index(np.ascontiguousarray(data, dtype=np.float32))

    def search(self, query: np.ndarray, knn: int, nprobe: int = DEFAULT_NPROBE):
        self._index.set_nprobe(nprobe)
        return self._index.search(np.ascontiguousarray(query, dtype=np.float32), knn)

    def filtered_search(self, query: np.ndarray, knn: int, row_ids: np.ndarray, nprobe: int = DEFAULT_NPROBE):
        self._index.set_nprobe(nprobe)
        return self._index.filtered_search(
            np.ascontiguousarray(query, dtype=np.float32), knn,
            np.ascontiguousarray(row_ids, dtype=np.uint64),
        )

    def append(self, row_id: int, embedding: np.ndarray) -> None:
        self._index.append(row_id, np.ascontiguousarray(embedding, dtype=np.float32))

    def delete(self, row_id: int) -> None:
        self._index.delete(row_id)

    def save(self, path: str) -> None:
        self._index.save(path)

    @property
    def num_dimensions(self) -> int:
        return self._index.get_num_dimensions()

    @property
    def num_clusters(self) -> int:
        return self._index.get_num_clusters()

    @property
    def in_memory_size_bytes(self) -> int:
        return self._index.get_in_memory_size_in_bytes()


class IndexPDXIVFTree:
    """Two-level IVF index (F32)."""

    def __init__(
        self,
        *,
        num_dimensions: int,
        distance_metric: str = "l2sq",
        normalize: bool = True,
        seed: int = 42,
        num_clusters: int = 0,
        num_meso_clusters: int = 0,
        sampling_fraction: float = 0.0,
        kmeans_iters: int = 10,
        hierarchical_indexing: bool = True,
        n_threads: int = 0,
    ) -> None:
        self._index = _PDXIndex(
            "pdx_tree_f32", num_dimensions, METRIC_MAP[distance_metric],
            seed, num_clusters, num_meso_clusters, normalize,
            sampling_fraction, kmeans_iters, hierarchical_indexing, n_threads,
        )
        # self.pe = PredicateEvaluator()

    def build(self, data: np.ndarray) -> None:
        self._index.build_index(np.ascontiguousarray(data, dtype=np.float32))

    def search(self, query: np.ndarray, knn: int, nprobe: int = DEFAULT_NPROBE):
        self._index.set_nprobe(nprobe)
        return self._index.search(np.ascontiguousarray(query, dtype=np.float32), knn)

    def filtered_search(self, query: np.ndarray, knn: int, row_ids: np.ndarray, nprobe: int = DEFAULT_NPROBE):
        self._index.set_nprobe(nprobe)
        return self._index.filtered_search(
            np.ascontiguousarray(query, dtype=np.float32), knn,
            np.ascontiguousarray(row_ids, dtype=np.uint64),
        )

    def append(self, row_id: int, embedding: np.ndarray) -> None:
        self._index.append(row_id, np.ascontiguousarray(embedding, dtype=np.float32))

    def delete(self, row_id: int) -> None:
        self._index.delete(row_id)

    def save(self, path: str) -> None:
        self._index.save(path)

    @property
    def num_dimensions(self) -> int:
        return self._index.get_num_dimensions()

    @property
    def num_clusters(self) -> int:
        return self._index.get_num_clusters()

    @property
    def in_memory_size_bytes(self) -> int:
        return self._index.get_in_memory_size_in_bytes()


class IndexPDXIVFTreeSQ8:
    """Two-level IVF index (U8 scalar quantization)."""

    def __init__(
        self,
        *,
        num_dimensions: int,
        distance_metric: str = "l2sq",
        normalize: bool = True,
        seed: int = 42,
        num_clusters: int = 0,
        num_meso_clusters: int = 0,
        sampling_fraction: float = 0.0,
        kmeans_iters: int = 10,
        hierarchical_indexing: bool = True,
        n_threads: int = 0,
    ) -> None:
        self._index = _PDXIndex(
            "pdx_tree_u8", num_dimensions, METRIC_MAP[distance_metric],
            seed, num_clusters, num_meso_clusters, normalize,
            sampling_fraction, kmeans_iters, hierarchical_indexing, n_threads,
        )
        # self.pe = PredicateEvaluator()

    def build(self, data: np.ndarray) -> None:
        self._index.build_index(np.ascontiguousarray(data, dtype=np.float32))

    def search(self, query: np.ndarray, knn: int, nprobe: int = DEFAULT_NPROBE):
        self._index.set_nprobe(nprobe)
        return self._index.search(np.ascontiguousarray(query, dtype=np.float32), knn)

    def filtered_search(self, query: np.ndarray, knn: int, row_ids: np.ndarray, nprobe: int = DEFAULT_NPROBE):
        self._index.set_nprobe(nprobe)
        return self._index.filtered_search(
            np.ascontiguousarray(query, dtype=np.float32), knn,
            np.ascontiguousarray(row_ids, dtype=np.uint64),
        )

    def append(self, row_id: int, embedding: np.ndarray) -> None:
        self._index.append(row_id, np.ascontiguousarray(embedding, dtype=np.float32))

    def delete(self, row_id: int) -> None:
        self._index.delete(row_id)

    def save(self, path: str) -> None:
        self._index.save(path)

    @property
    def num_dimensions(self) -> int:
        return self._index.get_num_dimensions()

    @property
    def num_clusters(self) -> int:
        return self._index.get_num_clusters()

    @property
    def in_memory_size_bytes(self) -> int:
        return self._index.get_in_memory_size_in_bytes()


def load_index(path: str):
    """Load a PDX index from a single file (auto-detects type)."""
    return _load_index(path)
