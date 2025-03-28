import numpy as np
import sys
import math

from typing import List
from pdxearch.index_core import IVF
from pdxearch.constants import PDXConstants


class Partition:
    def __init__(self):
        self.num_embeddings = 0
        self.indices = np.array([])
        self.blocks = []

#
# Transformer of the IVFFlat index to the different layouts (Nary, Dual Nary Block, or PDX)
#
class BaseIndexPDXIVF:
    def __init__(
            self,
            ndim: int,
            metric: str,
            nbuckets: int = 16
    ):
        self.ndim = ndim
        self.dtype = np.float32
        self.nbuckets = nbuckets
        self.core_index = IVF(ndim, metric, nbuckets)
        self.means: np.array = None
        self.centroids: np.array = np.array([], dtype=np.float32)
        self.partitions: List[Partition] = []
        self.num_partitions: int = 0
        self.materialized_index = None

    def train(self, data: np.array, **kwargs):
        self.core_index.train(data, **kwargs)

    def add(self, data: np.array, **kwargs):
        self.core_index.add(data, **kwargs)

    # Separate the data in the PDX blocks
    # TODO: Most probably this can be much more efficient
    def _to_pdx(self, data: np.array, _type='pdx', centroids_preprocessor=None, use_original_centroids=False, **kwargs):
        self.partitions = []
        self.means = data.mean(axis=0, dtype=np.float32)
        self.centroids = np.array([], dtype=np.float32)
        if use_original_centroids:
            self.centroids = self.core_index.index.quantizer.reconstruct_n(0, self.core_index.index.nlist)
            if centroids_preprocessor is not None:
                centroids_preprocessor.preprocess(self.centroids, inplace=True)
        for list_id in range(self.core_index.index.nlist):
            num_list_embeddings, list_ids = self.core_index.get_inverted_list_metadata(list_id)
            partition = Partition()
            partition.num_embeddings = num_list_embeddings
            partition.indices = np.zeros((partition.num_embeddings,), dtype=np.uint32)
            for embedding_index in range(partition.num_embeddings):
                partition.indices[embedding_index] = list_ids[embedding_index]
            # Tight blocks of 64, not used on IVF indexes for now
            # if _type == 'pdx' and kwargs.get('blockify', False):
            #     left_to_write = partition.num_embeddings
            #     already_written = 0
            #     while left_to_write > PDXConstants.PDX_VECTOR_SIZE:
            #         partition.blocks.append(data[partition.indices[already_written: already_written + PDXConstants.PDX_VECTOR_SIZE], :])
            #         already_written += PDXConstants.PDX_VECTOR_SIZE
            #         left_to_write -= PDXConstants.PDX_VECTOR_SIZE
            #     if left_to_write != 0:
            #         partition.blocks.append(data[partition.indices[already_written:], :])
            # else:  # Variable block size
            partition.blocks.append(data[partition.indices, :])
            if not use_original_centroids:
                partition_centroids = np.mean(data[partition.indices, :], axis=0, dtype=np.float32)
                self.centroids = np.append(self.centroids, partition_centroids)
            self.partitions.append(partition)
        self.num_partitions = len(self.partitions)
        self._materialize_index(_type, **kwargs)

    # Materialize the index with the given layout
    # TODO: Most probably this can be much more efficient
    def _materialize_index(self, _type='pdx', **kwargs):
        data = bytearray()
        data.extend(np.int32(self.ndim).tobytes("C"))
        data.extend(np.int32(self.num_partitions).tobytes("C"))
        for i in range(self.num_partitions):
            data.extend(np.int32(self.partitions[i].num_embeddings).tobytes("C"))
        if _type == 'dsm':
            whole = []
            for i in range(self.num_partitions):
                for p in range(len(self.partitions[i].blocks)):
                    whole.append(self.partitions[i].blocks[p])
            whole_f = np.concatenate(whole, axis=0)
            data.extend(whole_f.tobytes("F"))
        else:
            for i in range(self.num_partitions):
                for p in range(len(self.partitions[i].blocks)):
                    if _type == 'pdx':
                        data.extend(self.partitions[i].blocks[p].tobytes("F"))  # PDX
                    elif _type == 'pdx-4':
                        tmp_block = self.partitions[i].blocks[p]
                        for k in range(0, self.ndim, 4):
                            data.extend(tmp_block[:, k:k+4].tobytes("C"))
                    elif _type == 'n-ary':
                        data.extend(self.partitions[i].blocks[p].tobytes("C"))
                    elif _type == 'dual':
                        delta_d = kwargs.get('delta_d', PDXConstants.DEFAULT_DELTA_D)
                        data.extend(self.partitions[i].blocks[p][:, :delta_d].tobytes("C"))
                        data.extend(self.partitions[i].blocks[p][:, delta_d:].tobytes("C"))
        for i in range(self.num_partitions):
            data.extend(self.partitions[i].indices.tobytes("C"))
        data.extend(self.means.tobytes("C"))
        is_ivf = True
        data.extend(is_ivf.to_bytes(1, sys.byteorder))
        # Since centroids not many, we store them twice to have a dual layout
        # This is part of our EXPERIMENTAL feature in which we want to do Centroids pruning
        # We do not currently do so in PDXearch, but we DO use the PDX layout to do a vertical search
        # on the centroids
        data.extend(self.centroids.tobytes("C"))  # Nary format
        if _type == 'pdx' or _type == 'dsm':
            # PDX format
            centroids_written = 0
            reshaped_centroids = np.reshape(self.centroids, (self.num_partitions, self.ndim))
            while centroids_written != self.num_partitions:
                if centroids_written + PDXConstants.PDX_CENTROIDS_VECTOR_SIZE > self.num_partitions:
                    data.extend(reshaped_centroids[centroids_written:, :].tobytes("F"))
                    centroids_written = self.num_partitions
                else:
                    data.extend(
                        reshaped_centroids[
                            centroids_written: centroids_written + PDXConstants.PDX_CENTROIDS_VECTOR_SIZE, :
                        ].tobytes("F")
                    )
                    centroids_written += PDXConstants.PDX_CENTROIDS_VECTOR_SIZE
        self.materialized_index = bytes(data)

    def _persist(self, path: str):
        if self.materialized_index is None:
            raise Exception('The index have not been created')
        with open(path, "wb") as file:
            file.write(self.materialized_index)

#
# Transformer of collections to the different layouts (Nary, Dual Nary Block, or PDX)
#
class BaseIndexPDXFlat:
    def __init__(
            self,
            ndim: int,
            metric: str,
    ):
        self.ndim = ndim
        self.dtype = np.float32
        self.means: np.array = None
        self.partitions: List[Partition] = []
        self.num_partitions: int = 0
        self.materialized_index = None

    # Separate the data in the PDX blocks
    # TODO: Most probably this can be much more efficient
    def _to_pdx(self, data: np.array, size_partition: int, _type='pdx', **kwargs):
        self.partitions = []
        self.means = data.mean(axis=0, dtype=np.float32)
        num_embeddings = len(data)

        num_partitions = math.ceil(num_embeddings / size_partition)

        indices = np.arange(0, num_embeddings, dtype=np.uint32)
        np.random.shuffle(indices)

        shuffle_index = 0
        for partition_index in range(num_partitions):
            partition = Partition()
            partition.num_embeddings = min(num_embeddings - shuffle_index, size_partition)
            partition.indices = indices[shuffle_index: shuffle_index + partition.num_embeddings]
            # Each partition is one block
            partition.blocks.append(data[partition.indices, :])
            shuffle_index += partition.num_embeddings
            self.partitions.append(partition)
        self.num_partitions = len(self.partitions)
        self._materialize_index(_type, **kwargs)

    # Materialize the index with the given layout
    # TODO: Most probably this can be much more efficient
    def _materialize_index(self, _type='pdx', **kwargs):
        data = bytearray()
        data.extend(np.int32(self.ndim).tobytes("C"))
        data.extend(np.int32(self.num_partitions).tobytes("C"))
        for i in range(self.num_partitions):
            data.extend(np.int32(self.partitions[i].num_embeddings).tobytes("C"))
        if _type == 'dsm':
            whole = []
            for i in range(self.num_partitions):
                for p in range(len(self.partitions[i].blocks)):
                    whole.append(self.partitions[i].blocks[p])
            whole_f = np.concatenate(whole, axis=0)
            data.extend(whole_f.tobytes("F"))
        else:
            for i in range(self.num_partitions):
                for p in range(len(self.partitions[i].blocks)):
                    if _type == 'pdx':
                        data.extend(self.partitions[i].blocks[p].tobytes("F"))  # PDX
                    elif _type == 'pdx-4':
                        tmp_block = self.partitions[i].blocks[p]
                        for k in range(0, self.ndim, 4):
                            data.extend(tmp_block[:, k:k+4].tobytes("C"))
                    elif _type == 'n-ary':
                        data.extend(self.partitions[i].blocks[p].tobytes("C"))
                    elif _type == 'dual':
                        delta_d = kwargs.get('delta_d', PDXConstants.DEFAULT_DELTA_D)
                        data.extend(self.partitions[i].blocks[p][:, :delta_d].tobytes("C"))
                        data.extend(self.partitions[i].blocks[p][:, delta_d:].tobytes("C"))
        for i in range(self.num_partitions):
            data.extend(self.partitions[i].indices.tobytes("C"))
        data.extend(self.means.tobytes("C"))
        is_ivf = False
        data.extend(is_ivf.to_bytes(1, sys.byteorder))
        self.materialized_index = bytes(data)

    def _persist(self, path: str):
        if self.materialized_index is None:
            raise Exception('The index have not been created')
        with open(path, "wb") as file:
            file.write(self.materialized_index)

