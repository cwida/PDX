import faiss
import numpy as np

from multiprocessing import cpu_count
faiss.omp_set_num_threads(cpu_count() // 2)

#
# Wrapper of FAISS FlatIVF index
# TODO: Support more distance metrics
# TODO: Replace FAISS implementation with a propietary one
#


class IVF:
    def __init__(
            self,
            ndim: int = 0,
            metric: str = "l2sq",
            nbuckets: int = 4
    ) -> None:
        if metric not in ["l2sq"]:
            raise Exception("Distance metric not supported yet")
        self.ndim = ndim
        self.nbuckets = nbuckets
        self.metric = metric
        self.centroids = np.array([], dtype=np.float32)

        match self.metric:
            case "l2sq":
                quantizer = faiss.IndexFlatL2(int(self.ndim))
            case _:
                quantizer = faiss.IndexFlatL2(int(self.ndim))
        self.index: faiss.IndexIVFFlat = faiss.IndexIVFFlat(quantizer, int(self.ndim), int(self.nbuckets))

    def train(self, data, **kwargs):
        self.index.train(data, **kwargs)

    def add(self, data, **kwargs):
        self.index.add(data, **kwargs)

    def get_inverted_list_size(self, list_id):
        return self.index.invlists.list_size(list_id)

    def get_inverted_list_ids(self, list_id):
        return self.index.invlists.get_ids(list_id)

    def get_inverted_list_metadata(self, list_id):
        num_list_embeddings = self.get_inverted_list_size(list_id)
        list_ids = faiss.rev_swig_ptr(self.get_inverted_list_ids(list_id), num_list_embeddings)
        return num_list_embeddings, list_ids

    def persist_core(self, path: str):
        faiss.write_index(self.index, path)


class HNSW:
    pass
