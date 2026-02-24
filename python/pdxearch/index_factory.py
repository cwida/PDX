import numpy as np

from pdxearch.compiled import PDXIndex as _PDXIndex, load_index as _load_index
# from pdxearch.predicate_evaluator import PredicateEvaluator

METRIC_MAP = {"l2sq": 0, "cosine": 1, "ip": 2}


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
    ) -> None:
        self._index = _PDXIndex(
            "pdx_f32", num_dimensions, METRIC_MAP[distance_metric],
            seed, num_clusters, 0, normalize, sampling_fraction, kmeans_iters,
        )
        # self.pe = PredicateEvaluator()

    def build(self, data: np.ndarray) -> None:
        self._index.build_index(np.ascontiguousarray(data, dtype=np.float32))

    def search(self, query: np.ndarray, knn: int, nprobe: int = 16):
        self._index.set_nprobe(nprobe)
        return self._index.search(np.ascontiguousarray(query, dtype=np.float32), knn)

    def save(self, path: str) -> None:
        self._index.save(path)

    # def evaluate_predicate(self, passing_tuples_ids):
    #     self.pe.evaluate_predicate(passing_tuples_ids, self._index.get_labels())

    # def filtered_search(self, query: np.ndarray, knn: int, nprobe: int = 16):
    #     if len(self.pe.n_passing_tuples) == 0 or len(self.pe.selection_vector) == 0:
    #         raise ValueError("Call .evaluate_predicate([<passing_tuples>]) first")
    #     self._index.set_nprobe(nprobe)
    #     return self._index.filtered_search(
    #         np.ascontiguousarray(query, dtype=np.float32), knn,
    #         self.pe.n_passing_tuples, self.pe.selection_vector,
    #     )

    @property
    def num_dimensions(self) -> int:
        return self._index.get_num_dimensions()

    @property
    def num_clusters(self) -> int:
        return self._index.get_num_clusters()


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
    ) -> None:
        self._index = _PDXIndex(
            "pdx_u8", num_dimensions, METRIC_MAP[distance_metric],
            seed, num_clusters, 0, normalize, sampling_fraction, kmeans_iters,
        )
        # self.pe = PredicateEvaluator()

    def build(self, data: np.ndarray) -> None:
        self._index.build_index(np.ascontiguousarray(data, dtype=np.float32))

    def search(self, query: np.ndarray, knn: int, nprobe: int = 16):
        self._index.set_nprobe(nprobe)
        return self._index.search(np.ascontiguousarray(query, dtype=np.float32), knn)

    def save(self, path: str) -> None:
        self._index.save(path)

    # def evaluate_predicate(self, passing_tuples_ids):
    #     self.pe.evaluate_predicate(passing_tuples_ids, self._index.get_labels())

    # def filtered_search(self, query: np.ndarray, knn: int, nprobe: int = 16):
    #     if len(self.pe.n_passing_tuples) == 0 or len(self.pe.selection_vector) == 0:
    #         raise ValueError("Call .evaluate_predicate([<passing_tuples>]) first")
    #     self._index.set_nprobe(nprobe)
    #     return self._index.filtered_search(
    #         np.ascontiguousarray(query, dtype=np.float32), knn,
    #         self.pe.n_passing_tuples, self.pe.selection_vector,
    #     )

    @property
    def num_dimensions(self) -> int:
        return self._index.get_num_dimensions()

    @property
    def num_clusters(self) -> int:
        return self._index.get_num_clusters()


class IndexPDXIVF2:
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
    ) -> None:
        self._index = _PDXIndex(
            "pdx_tree_f32", num_dimensions, METRIC_MAP[distance_metric],
            seed, num_clusters, num_meso_clusters, normalize,
            sampling_fraction, kmeans_iters,
        )
        # self.pe = PredicateEvaluator()

    def build(self, data: np.ndarray) -> None:
        self._index.build_index(np.ascontiguousarray(data, dtype=np.float32))

    def search(self, query: np.ndarray, knn: int, nprobe: int = 16):
        self._index.set_nprobe(nprobe)
        return self._index.search(np.ascontiguousarray(query, dtype=np.float32), knn)

    def save(self, path: str) -> None:
        self._index.save(path)

    # def evaluate_predicate(self, passing_tuples_ids):
    #     self.pe.evaluate_predicate(passing_tuples_ids, self._index.get_labels())

    # def filtered_search(self, query: np.ndarray, knn: int, nprobe: int = 16):
    #     if len(self.pe.n_passing_tuples) == 0 or len(self.pe.selection_vector) == 0:
    #         raise ValueError("Call .evaluate_predicate([<passing_tuples>]) first")
    #     self._index.set_nprobe(nprobe)
    #     return self._index.filtered_search(
    #         np.ascontiguousarray(query, dtype=np.float32), knn,
    #         self.pe.n_passing_tuples, self.pe.selection_vector,
    #     )

    @property
    def num_dimensions(self) -> int:
        return self._index.get_num_dimensions()

    @property
    def num_clusters(self) -> int:
        return self._index.get_num_clusters()


class IndexPDXIVF2SQ8:
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
    ) -> None:
        self._index = _PDXIndex(
            "pdx_tree_u8", num_dimensions, METRIC_MAP[distance_metric],
            seed, num_clusters, num_meso_clusters, normalize,
            sampling_fraction, kmeans_iters,
        )
        # self.pe = PredicateEvaluator()

    def build(self, data: np.ndarray) -> None:
        self._index.build_index(np.ascontiguousarray(data, dtype=np.float32))

    def search(self, query: np.ndarray, knn: int, nprobe: int = 16):
        self._index.set_nprobe(nprobe)
        return self._index.search(np.ascontiguousarray(query, dtype=np.float32), knn)

    def save(self, path: str) -> None:
        self._index.save(path)

    # def evaluate_predicate(self, passing_tuples_ids):
    #     self.pe.evaluate_predicate(passing_tuples_ids, self._index.get_labels())

    # def filtered_search(self, query: np.ndarray, knn: int, nprobe: int = 16):
    #     if len(self.pe.n_passing_tuples) == 0 or len(self.pe.selection_vector) == 0:
    #         raise ValueError("Call .evaluate_predicate([<passing_tuples>]) first")
    #     self._index.set_nprobe(nprobe)
    #     return self._index.filtered_search(
    #         np.ascontiguousarray(query, dtype=np.float32), knn,
    #         self.pe.n_passing_tuples, self.pe.selection_vector,
    #     )

    @property
    def num_dimensions(self) -> int:
        return self._index.get_num_dimensions()

    @property
    def num_clusters(self) -> int:
        return self._index.get_num_clusters()


def load_index(path: str):
    """Load a PDX index from a single file (auto-detects type)."""
    return _load_index(path)
