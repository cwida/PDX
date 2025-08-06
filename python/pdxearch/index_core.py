import numpy as np
import struct

from multiprocessing import cpu_count

#
# Wrapper of FAISS FlatIVF index
# TODO: Support more distance metrics
# TODO: Replace FAISS implementation with a propietary one
#

class PartitionsUtils:
    @staticmethod
    def write_centroids_and_labels(centroids, labels, path):
        data = bytearray()
        num_centroids = len(centroids)
        data.extend(np.int32(num_centroids).tobytes("C"))
        data.extend(centroids.tobytes("C"))
        for l in labels:
            labels_n = len(l)
            data.extend(np.int32(labels_n).tobytes("C"))
            data.extend(l.tobytes("C"))
        with open(path + '.bin', "wb") as file:
            file.write(bytes(data))

    @staticmethod
    def read_centroids_and_labels(ndim, f):
        # Read number of centroids
        num_centroids_bytes = f.read(4)
        num_centroids = struct.unpack('<I', num_centroids_bytes)[0]

        # Read centroids
        centroids = []
        for _ in range(num_centroids):
            centroid_bytes = f.read(4 * ndim)
            centroid = struct.unpack('<' + 'f' * ndim, centroid_bytes)
            centroids.append(list(centroid))

        # Read labels
        labels = []
        for _ in range(len(centroids)):
            length_bytes = f.read(4)
            length = struct.unpack('<I', length_bytes)[0]
            label_bytes = f.read(8 * length)
            label = struct.unpack('<' + 'q' * length, label_bytes)
            labels.append(list(label))
        return centroids, labels

class IVF(PartitionsUtils):
    import faiss
    def __init__(
            self,
            ndim: int = 0,
            metric: str = "l2sq",
            nbuckets: int = 4
    ) -> None:
        IVF.faiss.omp_set_num_threads(cpu_count())
        if metric not in ["l2sq"]:
            raise Exception("Distance metric not supported yet")
        self.ndim = ndim
        self.labels = []
        self.nbuckets = nbuckets
        self.metric = metric
        self.centroids = np.array([], dtype=np.float32)

        match self.metric:
            case "l2sq":
                quantizer = IVF.faiss.IndexFlatL2(int(self.ndim))
            case _:
                quantizer = IVF.faiss.IndexFlatL2(int(self.ndim))
        self.index: IVF.faiss.IndexIVFFlat = IVF.faiss.IndexIVFFlat(quantizer, int(self.ndim), int(self.nbuckets))


    def train(self, data, **kwargs):
        self.index.train(data, **kwargs)

    def add(self, data, **kwargs):
        self.index.add(data, **kwargs)
        self.centroids = self.index.quantizer.reconstruct_n(0, self.index.nlist)
        self.fill_labels()

    def read_index(self, path):
        with open(path + '.bin', 'rb') as f:
            self.centroids, self.labels = self.read_centroids_and_labels(self.ndim, f)
        self.nbuckets = len(self.labels)
        # print('L1 buckets: ', self.nbuckets)

    def fill_labels(self):
        for i in range(self.nbuckets):
            _, cur_labels = self.get_inverted_list_metadata(i)
            self.labels.append(cur_labels)

    def get_inverted_list_size(self, list_id):
        return self.index.invlists.list_size(list_id)

    def get_inverted_list_ids(self, list_id):
        return self.index.invlists.get_ids(list_id)

    def get_inverted_list_metadata(self, list_id):
        num_list_embeddings = self.get_inverted_list_size(list_id)
        list_ids = IVF.faiss.rev_swig_ptr(self.get_inverted_list_ids(list_id), num_list_embeddings)
        return num_list_embeddings, list_ids

    def persist_core(self, path: str):
        self.write_centroids_and_labels(self.centroids, self.labels, path)


class IMI(PartitionsUtils):
    import faiss
    def __init__(
            self,
            ndim: int = 0,
            metric: str = "l2sq",
            nbuckets: int = 4,
            nbuckets_l0: int = 32
    ) -> None:
        IMI.faiss.omp_set_num_threads(cpu_count())
        if metric not in ["l2sq"]:
            raise Exception("Distance metric not supported yet")
        self.ndim = ndim
        self.nbuckets = nbuckets
        self.nbuckets_l0 = nbuckets_l0
        self.metric = metric
        self.labels = []
        self.labels_l0 = []
        self.centroids = np.array([], dtype=np.float32)
        self.centroids_l0 = np.array([], dtype=np.float32)

        match self.metric:
            case "l2sq":
                quantizer = IMI.faiss.IndexFlatL2(int(self.ndim))
                quantizer_l0 = IMI.faiss.IndexFlatL2(int(self.ndim))
            case _:
                quantizer = IMI.faiss.IndexFlatL2(int(self.ndim))
                quantizer_l0 = IMI.faiss.IndexFlatL2(int(self.ndim))
        self.index: IMI.faiss.IndexIVFFlat = IMI.faiss.IndexIVFFlat(quantizer, int(self.ndim), int(self.nbuckets))
        self.index_l0: IMI.faiss.IndexIVFFlat = IMI.faiss.IndexIVFFlat(quantizer_l0, int(self.ndim), int(self.nbuckets_l0))

    def train(self, data, **kwargs):
        self.index.train(data, **kwargs)

    def add(self, data, **kwargs):
        self.index.add(data, **kwargs)
        self.fill_labels(level=1)

    def read_index(self, path, path_l0):
        with open(path + '.bin', 'rb') as f:
            self.centroids, self.labels = self.read_centroids_and_labels(self.ndim, f)
        self.nbuckets = len(self.labels)

        with open(path_l0 + '.bin', 'rb') as f:
            self.centroids_l0, self.labels_l0 = self.read_centroids_and_labels(self.ndim, f)
        self.nbuckets_l0 = len(self.labels_l0)

        # print('L0 buckets: ', self.nbuckets_l0)
        # print('L1 buckets: ', self.nbuckets)

    def fill_labels(self, level=1):
        if level == 1:
            for i in range(self.nbuckets):
                _, cur_labels = self.get_inverted_list_metadata(i, level=1)
                self.labels.append(cur_labels)
        else:
            for i in range(self.nbuckets_l0):
                _, cur_labels = self.get_inverted_list_metadata(i, level=0)
                self.labels_l0.append(cur_labels)

    def train_add_l0(self, **kwargs):
        self.centroids = self.index.quantizer.reconstruct_n(0, self.index.nlist)
        # print(len(self.centroids), self.nbuckets_l0)
        self.index_l0.train(self.centroids, **kwargs)
        self.index_l0.add(self.centroids, **kwargs)
        self.fill_labels(level=0)
        self.centroids_l0 = self.index_l0.quantizer.reconstruct_n(0, self.index_l0.nlist)

    def get_inverted_list_size(self, list_id, level=1):
        if level == 1:
            return self.index.invlists.list_size(list_id)
        else:
            return self.index_l0.invlists.list_size(list_id)

    def get_inverted_list_ids(self, list_id, level=1):
        if level == 1:
            return self.index.invlists.get_ids(list_id)
        else:
            return self.index_l0.invlists.get_ids(list_id)

    def get_inverted_list_metadata(self, list_id, level=1):
        num_list_embeddings = self.get_inverted_list_size(list_id, level)
        list_ids = IMI.faiss.rev_swig_ptr(self.get_inverted_list_ids(list_id, level), num_list_embeddings)
        return num_list_embeddings, list_ids

    def persist_core(self, path: str, path_l0: str):
        self.write_centroids_and_labels(self.centroids, self.labels, path)
        self.write_centroids_and_labels(self.centroids_l0, self.labels_l0, path_l0)
        IMI.faiss.write_index(self.index, path)

class HNSW:
    pass
