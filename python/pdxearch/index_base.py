import numpy as np
import sys
import math
from typing import List
from pdxearch.index_core import IVF, IMI
from pdxearch.constants import PDXConstants

class Partition:
    def __init__(self):
        self.num_embeddings = 0
        self.indices = np.array([])
        self.blocks = []

class BaseIndexPDXIMI:
    def __init__(
            self,
            ndim: int,
            metric: str,
            nbuckets: int = 16,
            nbuckets_l0: int = 2,
            normalize: bool = True
    ):
        if ndim < 128:
            raise Exception('`ndim` must be >= 128')
        if ndim % 4 != 0:
            raise Exception('`ndim` must be multiple of 4')
        self.ndim = ndim
        self.normalize = normalize
        self.dtype = np.float32
        self.nbuckets = nbuckets
        self.nbuckets_l0 = nbuckets_l0
        self.for_base = 0.0
        self.scale_factor = 0.0

        self.core_index = IMI(ndim, metric, nbuckets, nbuckets_l0)

        self.means: np.array = None
        self.centroids: np.array = np.array([], dtype=np.float32)
        self.partitions: List[Partition] = []
        self.num_partitions: int = 0

        self.means_l0: np.array = None
        self.centroids_l0: np.array = np.array([], dtype=np.float32)
        self.partitions_l0: List[Partition] = []
        self.num_partitions_l0: int = 0

        self.materialized_index = None

    def train(self, data: np.array, **kwargs):
        self.core_index.train(data, **kwargs)

    def add(self, data: np.array, **kwargs):
        self.core_index.add(data, **kwargs)

    def train_add_l0(self,  **kwargs):
        self.core_index.train_add_l0(**kwargs)

    # Separate the data in the PDX blocks
    # TODO: Most probably this can be much more efficient
    def _to_pdx(self, data: np.array, _type='pdx', centroids_preprocessor=None, use_original_centroids=False, **kwargs):
        use_sq = kwargs.get('quantize', False)
        self.partitions = []
        self.partitions_l0 = []
        self.means = data.mean(axis=0, dtype=np.float32)
        self.centroids = np.array(self.core_index.centroids, dtype=np.float32)
        self.centroids_l0 = np.array(self.core_index.centroids_l0, dtype=np.float32)
        if use_original_centroids:
            if centroids_preprocessor is not None:
                centroids_preprocessor.preprocess(self.centroids, inplace=True)
                centroids_preprocessor.preprocess(self.centroids_l0, inplace=True)
        for list_id in range(self.core_index.nbuckets_l0):
            list_ids = self.core_index.labels_l0[list_id]
            num_list_embeddings = len(list_ids)
            partition = Partition()
            partition.num_embeddings = num_list_embeddings
            partition.indices = np.zeros((partition.num_embeddings,), dtype=np.uint32)
            for embedding_index in range(partition.num_embeddings):
                partition.indices[embedding_index] = list_ids[embedding_index]
            partition.blocks.append(self.centroids[partition.indices, :])
            self.partitions_l0.append(partition)
        self.num_partitions_l0 = len(self.partitions_l0)
        if use_sq:
            data_max = data.max()
            data_min = data.min()
            data_range = data_max - data_min
            data = data - data_min
            global_scale_factor = float(PDXConstants.U8_MAX) / data_range
            self.scale_factor = global_scale_factor
            self.for_base = data_min

        for list_id in range(self.core_index.nbuckets):
            list_ids = self.core_index.labels[list_id]
            num_list_embeddings = len(list_ids)
            partition = Partition()
            partition.num_embeddings = num_list_embeddings
            partition.indices = np.zeros((partition.num_embeddings,), dtype=np.uint32)
            for embedding_index in range(partition.num_embeddings):
                partition.indices[embedding_index] = list_ids[embedding_index]
            if (_type == 'pdx-v4-h') and use_sq:
                # TODO: Move outside (?)
                pre_data = data[partition.indices, :]
                pre_data = pre_data * global_scale_factor
                pre_data = pre_data.round(decimals=0).astype(dtype=np.int32)
                for_data = pre_data
                for_data = for_data.astype(dtype=np.uint8)  # Always using np.uint8
                partition.blocks.append(for_data)
            else:
                partition.blocks.append(data[partition.indices, :])
            self.partitions.append(partition)
        self.num_partitions = len(self.partitions)
        self._materialize_index(_type, **kwargs)

    # Materialize the index with the given layout
    # TODO: Most probably this can be much more efficient
    def _materialize_index(self, _type='pdx', **kwargs):
        data = bytearray()
        data.extend(np.int32(self.ndim).tobytes("C"))
        data.extend(np.int32(self.num_partitions).tobytes("C"))
        data.extend(np.int32(self.num_partitions_l0).tobytes("C"))

        h_dims = int(self.ndim * PDXConstants.VERTICAL_PROPORTION_DIM)
        v_dims = self.ndim - h_dims
        if h_dims % PDXConstants.HORIZONTAL_DIMENSIONS_GROUPING != 0:
            h_dims = round(h_dims / PDXConstants.HORIZONTAL_DIMENSIONS_GROUPING) * PDXConstants.HORIZONTAL_DIMENSIONS_GROUPING
            v_dims = self.ndim - h_dims
        h_dims_block = kwargs.get('h_dims_block', PDXConstants.HORIZONTAL_DIMENSIONS_GROUPING)
        if v_dims == 0:
            h_dims = PDXConstants.HORIZONTAL_DIMENSIONS_GROUPING
            v_dims = self.ndim - h_dims

        # L0
        for i in range(self.num_partitions_l0):
            data.extend(np.int32(self.partitions_l0[i].num_embeddings).tobytes("C"))
        for i in range(self.num_partitions_l0):
            for p in range(len(self.partitions_l0[i].blocks)):
                vertical_block = self.partitions_l0[i].blocks[p][:, :v_dims]
                rows, _ = vertical_block.shape
                data.extend(vertical_block.tobytes("F"))  # PDX vertical block
                pdx_h_block = self.partitions_l0[i].blocks[p][:, v_dims:].reshape(rows, -1, h_dims_block).transpose(1, 0, 2).reshape(-1)
                data.extend(pdx_h_block.tobytes("C")) # PDX horizontal block to improve sequential access
        for i in range(self.num_partitions_l0):
            data.extend(self.partitions_l0[i].indices.tobytes("C"))

        # L1
        for i in range(self.num_partitions):
            data.extend(np.int32(self.partitions[i].num_embeddings).tobytes("C"))
        for i in range(self.num_partitions):
            for p in range(len(self.partitions[i].blocks)):
                assert h_dims % PDXConstants.X4_GROUPING == 0
                assert v_dims % PDXConstants.X4_GROUPING == 0
                assert h_dims + v_dims == self.ndim
                assert v_dims != 0
                assert h_dims != 0
                assert h_dims % PDXConstants.HORIZONTAL_DIMENSIONS_GROUPING == 0
                if _type == 'pdx':
                    vertical_block = self.partitions[i].blocks[p][:, :v_dims]
                    rows, _ = vertical_block.shape
                    data.extend(vertical_block.tobytes("F"))  # PDX vertical block
                    pdx_h_block = self.partitions[i].blocks[p][:, v_dims:].reshape(rows, -1, h_dims_block).transpose(1, 0, 2).reshape(-1)
                    data.extend(pdx_h_block.tobytes("C")) # PDX horizontal block to improve sequential access
                elif _type == 'pdx-v4-h':
                    tmp_block = self.partitions[i].blocks[p][:, :v_dims]
                    rows, _ = tmp_block.shape
                    pdx_4_block = tmp_block.reshape(rows, -1, 4).transpose(1, 0, 2).reshape(-1)
                    assert h_dims % h_dims_block == 0
                    data.extend(pdx_4_block.tobytes("C"))
                    # Horizontal block (rest)
                    pdx_h_block = self.partitions[i].blocks[p][:, v_dims:].reshape(rows, -1, h_dims_block).transpose(1, 0, 2).reshape(-1)
                    data.extend(pdx_h_block.tobytes("C"))

        for i in range(self.num_partitions):
            data.extend(self.partitions[i].indices.tobytes("C"))
        data.extend(self.normalize.to_bytes(1, sys.byteorder))
        # TODO: Support other multiples of 64
        if len(self.centroids_l0) == PDXConstants.PDX_CENTROIDS_VECTOR_SIZE:
            data.extend(self.centroids_l0.tobytes("F")) # PDX
        else:
            data.extend(self.centroids_l0.tobytes("C"))  # Nary format
        if (_type == 'pdx-v4-h') and kwargs.get('quantize', False):
            data.extend(np.float32(self.for_base).tobytes("C"))
            data.extend(np.float32(self.scale_factor).tobytes("C"))
        self.materialized_index = bytes(data)

    def _persist(self, path: str):
        if self.materialized_index is None:
            raise Exception('The index have not been created')
        with open(path, "wb") as file:
            file.write(self.materialized_index)

class BaseIndexPDXIVF:
    def __init__(
            self,
            ndim: int,
            metric: str,
            nbuckets: int = 16,
            normalize: bool = True
    ):
        if ndim < 128:
            raise Exception('`ndim` must be >= 128')
        if ndim % 4 != 0:
            raise Exception('`ndim` must be multiple of 4')
        self.ndim = ndim
        self.normalize = normalize
        self.dtype = np.float32
        self.nbuckets = nbuckets
        self.core_index = IVF(ndim, metric, nbuckets)
        self.means: np.array = None
        self.centroids: np.array = np.array([], dtype=np.float32)
        self.partitions: List[Partition] = []
        self.num_partitions: int = 0
        self.materialized_index = None
        self.for_base = 0.0
        self.scale_factor = 0.0

    def train(self, data: np.array, **kwargs):
        self.core_index.train(data, **kwargs)

    def add(self, data: np.array, **kwargs):
        self.core_index.add(data, **kwargs)

    # Separate the data in the PDX blocks
    # TODO: Most probably this can be much more efficient
    def _to_pdx(self, data: np.array, _type='pdx', centroids_preprocessor=None, use_original_centroids=False, **kwargs):
        use_sq = kwargs.get('quantize', False)
        self.partitions = []
        self.means = data.mean(axis=0, dtype=np.float32)
        self.centroids = np.array(self.core_index.centroids, dtype=np.float32)
        if use_original_centroids and centroids_preprocessor is not None:
            centroids_preprocessor.preprocess(self.centroids, inplace=True)
        if use_sq:
            data_max = data.max()
            data_min = data.min()
            data_range = data_max - data_min
            data = data - data_min
            self.for_base = data_min
            global_scale_factor = float(PDXConstants.U8_MAX) / data_range
        for list_id in range(self.core_index.nbuckets):
            list_ids = self.core_index.labels[list_id]
            num_list_embeddings = len(list_ids)
            partition = Partition()
            partition.num_embeddings = num_list_embeddings
            partition.indices = np.zeros((partition.num_embeddings,), dtype=np.uint32)
            for embedding_index in range(partition.num_embeddings):
                partition.indices[embedding_index] = list_ids[embedding_index]
            if (_type == 'pdx-v4-h') and use_sq:
                # TODO: Get out
                pre_data = data[partition.indices, :]
                pre_data = pre_data * global_scale_factor
                pre_data = pre_data.round(decimals=0).astype(dtype=np.int32)
                for_data = pre_data
                self.scale_factor = global_scale_factor
                for_data = for_data.astype(dtype=np.uint8) # Always using np.uint8
                partition.blocks.append(for_data)
            else:
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

        h_dims = int(self.ndim * PDXConstants.VERTICAL_PROPORTION_DIM)
        v_dims = self.ndim - h_dims
        if h_dims % PDXConstants.HORIZONTAL_DIMENSIONS_GROUPING != 0:
            h_dims = round(h_dims / PDXConstants.HORIZONTAL_DIMENSIONS_GROUPING) * PDXConstants.HORIZONTAL_DIMENSIONS_GROUPING
            v_dims = self.ndim - h_dims
        h_dims_block = kwargs.get('h_dims_block', PDXConstants.HORIZONTAL_DIMENSIONS_GROUPING)
        if v_dims == 0:
            h_dims = PDXConstants.HORIZONTAL_DIMENSIONS_GROUPING
            v_dims = self.ndim - h_dims

        for i in range(self.num_partitions):
            data.extend(np.int32(self.partitions[i].num_embeddings).tobytes("C"))

        for i in range(self.num_partitions):
            for p in range(len(self.partitions[i].blocks)):
                assert h_dims % PDXConstants.X4_GROUPING == 0
                assert v_dims % PDXConstants.X4_GROUPING == 0
                assert h_dims + v_dims == self.ndim
                assert v_dims != 0
                assert h_dims != 0
                assert h_dims % PDXConstants.HORIZONTAL_DIMENSIONS_GROUPING == 0
                if _type == 'pdx':
                    if kwargs.get('bond', False):
                        data.extend(self.partitions[i].blocks[p].tobytes("F"))
                    else:
                        vertical_block = self.partitions[i].blocks[p][:, :v_dims]
                        rows, _ = vertical_block.shape
                        data.extend(vertical_block.tobytes("F"))  # PDX vertical block
                        pdx_h_block = self.partitions[i].blocks[p][:, v_dims:].reshape(rows, -1, h_dims_block).transpose(1, 0, 2).reshape(-1)
                        data.extend(pdx_h_block.tobytes("C")) # PDX horizontal block to improve sequential access
                elif _type == 'pdx-v4-h':
                    tmp_block = self.partitions[i].blocks[p][:, :v_dims]
                    rows, _ = tmp_block.shape
                    pdx_4_block = tmp_block.reshape(rows, -1, 4).transpose(1, 0, 2).reshape(-1)
                    assert h_dims % h_dims_block == 0
                    # Vertical Dimensions
                    data.extend(pdx_4_block.tobytes("C"))
                    # Horizontal block (rest)
                    pdx_h_block = self.partitions[i].blocks[p][:, v_dims:].reshape(rows, -1, h_dims_block).transpose(1, 0, 2).reshape(-1)
                    data.extend(pdx_h_block.tobytes("C"))
        for i in range(self.num_partitions):
            data.extend(self.partitions[i].indices.tobytes("C"))
        if _type == 'pdx':
            data.extend(self.means.tobytes("C"))
        is_ivf = True
        data.extend(self.normalize.to_bytes(1, sys.byteorder))
        data.extend(is_ivf.to_bytes(1, sys.byteorder))
        # Since centroids not many, we store them twice to have a dual layout
        data.extend(self.centroids.tobytes("C"))  # Nary format
        if _type == 'pdx':
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
        if (_type == 'pdx-v4-h') and kwargs.get('quantize', False):
            data.extend(np.float32(self.for_base).tobytes("C"))
            data.extend(np.float32(self.scale_factor).tobytes("C"))
        self.materialized_index = bytes(data)

    def _persist(self, path: str):
        if self.materialized_index is None:
            raise Exception('The index have not been created')
        with open(path, "wb") as file:
            file.write(self.materialized_index)


class BaseIndexPDXFlat:
    def __init__(
            self,
            ndim: int,
            metric: str,
            normalize: bool = True,
    ):
        if ndim < 128:
            raise Exception('`ndim` must be >= 128')
        if ndim % 4 != 0:
            raise Exception('`ndim` must be multiple of 4')
        self.ndim = ndim
        self.dtype = np.float32
        self.normalize = normalize
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

        if kwargs.get('randomize', True):
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
        h_dims = int(self.ndim * PDXConstants.VERTICAL_PROPORTION_DIM)
        v_dims = self.ndim - h_dims
        if h_dims % PDXConstants.HORIZONTAL_DIMENSIONS_GROUPING != 0:
            h_dims = round(h_dims / PDXConstants.HORIZONTAL_DIMENSIONS_GROUPING) * PDXConstants.HORIZONTAL_DIMENSIONS_GROUPING
            v_dims = self.ndim - h_dims
        h_dims_block = kwargs.get('h_dims_block', PDXConstants.HORIZONTAL_DIMENSIONS_GROUPING)
        if v_dims == 0:
            h_dims = PDXConstants.HORIZONTAL_DIMENSIONS_GROUPING
            v_dims = self.ndim - h_dims

        for i in range(self.num_partitions):
            data.extend(np.int32(self.partitions[i].num_embeddings).tobytes("C"))
        for i in range(self.num_partitions):
            for p in range(len(self.partitions[i].blocks)):
                if _type == 'pdx':
                    # In Bond we use fully decomposed
                    if kwargs.get('bond', False):
                        data.extend(self.partitions[i].blocks[p].tobytes("F"))
                    else: # In ADSampling we use a hybrid layout
                        vertical_block = self.partitions[i].blocks[p][:, :v_dims]
                        rows, _ = vertical_block.shape
                        data.extend(vertical_block.tobytes("F"))  # PDX vertical block
                        pdx_h_block = self.partitions[i].blocks[p][:, v_dims:].reshape(rows, -1, h_dims_block).transpose(1, 0, 2).reshape(-1)
                        data.extend(pdx_h_block.tobytes("C")) # PDX horizontal block to improve sequential access
        for i in range(self.num_partitions):
            data.extend(self.partitions[i].indices.tobytes("C"))
        data.extend(self.means.tobytes("C"))
        is_ivf = False
        data.extend(self.normalize.to_bytes(1, sys.byteorder))
        data.extend(is_ivf.to_bytes(1, sys.byteorder))
        self.materialized_index = bytes(data)

    def _persist(self, path: str):
        if self.materialized_index is None:
            raise Exception('The index have not been created')
        with open(path, "wb") as file:
            file.write(self.materialized_index)

