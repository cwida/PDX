import numpy as np
import sys
import math
from bitarray import bitarray

from typing import List
from pdxearch.index_core import IVF
from pdxearch.constants import PDXConstants
from pdxearch.fastlanes_pack import *


class Partition:
    def __init__(self):
        self.num_embeddings = 0
        self.indices = np.array([])
        self.for_bases = np.array([])
        self.blocks = []

#
# Transformer of the IVFFlat index to the different layouts (Nary, Dual Nary Block, or PDX)
#
class BaseIndexPDXIVF:
    def __init__(
            self,
            ndim: int,
            metric: str,
            nbuckets: int = 16,
            normalize: bool = True
    ):
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

    def train(self, data: np.array, **kwargs):
        self.core_index.train(data, **kwargs)

    def add(self, data: np.array, **kwargs):
        self.core_index.add(data, **kwargs)

    # Separate the data in the PDX blocks
    # TODO: Most probably this can be much more efficient
    def _to_pdx(self, data: np.array, _type='pdx', centroids_preprocessor=None, use_original_centroids=False, **kwargs):
        exceptions_count = 0
        lep_bw = kwargs.get('lep_bw', 8)
        use_lep = kwargs.get('lep', False)
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
            if (_type == 'pdx-4' or _type == 'pdx-v4-h') and use_lep:
                # TODO: Call to LEP class to determine exponent, FOR base, and bit-width
                pre_data = data[partition.indices, :]
                if lep_bw == 7:
                    pre_data = pre_data * 1000
                    pre_data = pre_data * 0.5
                    pre_data = pre_data.round(decimals=0).astype(dtype=np.int32)
                    for_bases = np.min(pre_data, axis=0).astype(dtype=np.int32)
                    for_data = pre_data - for_bases
                    lep_min = 0
                    lep_max = 127
                elif lep_bw == 6:  # TODO: More smart min/max
                    # Method 1, Clip means:
                    # pre_data = pre_data * 1000
                    # pre_data = pre_data.round(decimals=0).astype(dtype=np.int32)
                    # for_bases = np.mean(pre_data, axis=0).astype(dtype=np.int32) - 32  # 32 (64 / 2) below the mean
                    # for_data = pre_data - for_bases
                    # lep_min = 0
                    # lep_max = 64
                    # Method 2: Scale all:
                    pre_data = pre_data * 1000
                    pre_data = pre_data * 0.25
                    pre_data = pre_data.round(decimals=0).astype(dtype=np.int32)
                    for_bases = np.min(pre_data, axis=0).astype(dtype=np.int32)
                    for_data = pre_data - for_bases
                    lep_min = 0
                    lep_max = 64
                elif lep_bw == 4:   # LEP-4
                    pre_data = pre_data * 1000
                    pre_data = pre_data * 0.0625
                    pre_data = pre_data.round(decimals=0).astype(dtype=np.int32)
                    for_bases = np.min(pre_data, axis=0).astype(dtype=np.int32)
                    for_data = pre_data - for_bases
                    lep_min = 0
                    lep_max = 16
                else:  # LEP-8
                    pre_data = pre_data * 1000
                    pre_data = pre_data * 1.5 # TODO: Fix
                    pre_data = pre_data.round(decimals=0).astype(dtype=np.int32)
                    for_bases = np.min(pre_data, axis=0).astype(dtype=np.int32)
                    for_data = pre_data - for_bases
                    lep_min = 0
                    lep_max = 255
                if np.any(for_data > lep_max) or np.any(for_data < lep_min):  # TODO: We are assuming data fits in 8-bits
                    if lep_bw == 8 or lep_bw == 7:  # I am only interested of knowing this if we use 8-bits
                        print(f'LEP overflow when converting to uint8 in partition {list_id}')
                    exceptions_count += ((for_data > lep_max) | (for_data < lep_min)).sum()
                    # TODO: Support exceptions
                    for_data = np.clip(for_data, lep_min, lep_max)  # Perhaps using a mask (e.g. matrix & 0x3F) is faster
                    # perhaps we can clip for now, instead of exceptions and see how recall is
                    # for_data = np.clip(for_data, None, 255)
                # Always using np.uint8
                for_data = for_data.astype(dtype=np.uint8)
                partition.for_bases = for_bases
                # Tight blocks of 64, reintroducing them for uint8
                if kwargs.get('blockify', False):
                    left_to_write = partition.num_embeddings
                    already_written = 0
                    while left_to_write > PDXConstants.PDX_VECTOR_SIZE:
                        partition.blocks.append(for_data[already_written: already_written + PDXConstants.PDX_VECTOR_SIZE, :])
                        already_written += PDXConstants.PDX_VECTOR_SIZE
                        left_to_write -= PDXConstants.PDX_VECTOR_SIZE
                    if left_to_write != 0:
                        partition.blocks.append(for_data[already_written:, :])
                else:
                    partition.blocks.append(for_data)
            else:
                partition.blocks.append(data[partition.indices, :])
            if not use_original_centroids:
                partition_centroids = np.mean(data[partition.indices, :], axis=0, dtype=np.float32)
                self.centroids = np.append(self.centroids, partition_centroids)
            self.partitions.append(partition)
        self.num_partitions = len(self.partitions)
        print(f'LEP-{lep_bw} Compression finished. Found: {exceptions_count} exceptions')
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
                # if i % 100 == 0:
                #     print(f"{i}/{self.num_partitions} partitions processed")
                for p in range(len(self.partitions[i].blocks)):
                    if _type == 'pdx':
                        data.extend(self.partitions[i].blocks[p].tobytes("F"))  # PDX
                    elif _type == 'pdx-v4-h':
                        h_dims = int(self.ndim * 0.75)
                        v_dims = self.ndim - h_dims
                        assert h_dims % 4 == 0
                        assert v_dims % 4 == 0
                        assert h_dims > v_dims
                        assert v_dims % 64 == 0
                        tmp_block = self.partitions[i].blocks[p][:, :v_dims]
                        rows, _ = tmp_block.shape
                        pdx_4_block = tmp_block.reshape(rows, -1, 4).transpose(1, 0, 2).reshape(-1)
                        lep_bw = kwargs.get('lep_bw', 8)
                        h_dims_block = kwargs.get('h_dims_block', 64)
                        assert h_dims % h_dims_block == 0
                        if lep_bw == 8 or lep_bw == 7:  # Vertical Dimensions
                            data.extend(pdx_4_block.tobytes("C"))
                        # Horizontal block (rest)
                        pdx_64_block = self.partitions[i].blocks[p][:, v_dims:].reshape(rows, -1, h_dims_block).transpose(1, 0, 2).reshape(-1)
                        data.extend(pdx_64_block.tobytes("C"))
                    elif _type == 'pdx-4':
                        tmp_block = self.partitions[i].blocks[p]
                        rows, _ = tmp_block.shape
                        pdx_4_block = tmp_block.reshape(rows, -1, 4).transpose(1, 0, 2).reshape(-1)
                        lep_bw = kwargs.get('lep_bw', 8)
                        # For lep_bw = 7, we just store it as 8 bits
                        # While we don't get smaller size we get faster kernels (potentially)
                        if lep_bw == 8 or lep_bw == 7:
                            data.extend(pdx_4_block.tobytes("C"))
                        else:
                            total_values = len(pdx_4_block)
                            padding_length = (1024 - total_values % 1024) % 1024  # Fastlanes needs 1024 values
                            pdx_4_block = np.pad(pdx_4_block, (0, padding_length), mode='constant', constant_values=0)
                            out_array = np.zeros(packed_length(len(pdx_4_block), lep_bw), dtype=np.uint8)
                            compression_unit_length = packed_length(1024, lep_bw)
                            if lep_bw == 4:
                                compression_unit_offset = 0
                                for i in range(0, len(pdx_4_block), 1024):
                                    #pack_4bit(pdx_4_block[i: i+1024], out_array[compression_unit_offset: compression_unit_offset+compression_unit_length])
                                    pack_4bit_symmetric(pdx_4_block[i: i+1024], out_array[compression_unit_offset: compression_unit_offset+compression_unit_length])
                                    compression_unit_offset += compression_unit_length
                            elif lep_bw == 6:
                                compression_unit_offset = 0
                                for i in range(0, len(pdx_4_block), 1024):
                                    pack_6bit(pdx_4_block[i: i+1024], out_array[compression_unit_offset: compression_unit_offset+compression_unit_length])
                                    # pack_4bit_symmetric(pdx_4_block[i: i+1024], out_array[compression_unit_offset: compression_unit_offset+compression_unit_length])
                                    compression_unit_offset += compression_unit_length
                            data.extend(out_array.tobytes("C"))
                            # if len(tmp_bitarray):  # Byte aligned every 4 dimensions
                            #     data.extend(tmp_bitarray.tobytes())
                        # if len(tmp_bitarray):  # Byte aligned every partition
                        #     data.extend(tmp_bitarray.tobytes())
                    elif _type == 'n-ary':
                        data.extend(self.partitions[i].blocks[p].tobytes("C"))
                    elif _type == 'dual':
                        delta_d = kwargs.get('delta_d', PDXConstants.DEFAULT_DELTA_D)
                        data.extend(self.partitions[i].blocks[p][:, :delta_d].tobytes("C"))
                        data.extend(self.partitions[i].blocks[p][:, delta_d:].tobytes("C"))
        for i in range(self.num_partitions):
            data.extend(self.partitions[i].indices.tobytes("C"))
        if (_type == 'pdx-4' or _type == 'pdx-v4-h') and kwargs.get('lep', False):
            for i in range(self.num_partitions):
                data.extend(self.partitions[i].for_bases.tobytes("C"))
        data.extend(self.means.tobytes("C"))
        is_ivf = True
        data.extend(self.normalize.to_bytes(1, sys.byteorder))
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
            normalize: bool = True,
    ):
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
        data.extend(self.normalize.to_bytes(1, sys.byteorder))
        data.extend(is_ivf.to_bytes(1, sys.byteorder))
        self.materialized_index = bytes(data)

    def _persist(self, path: str):
        if self.materialized_index is None:
            raise Exception('The index have not been created')
        with open(path, "wb") as file:
            file.write(self.materialized_index)

