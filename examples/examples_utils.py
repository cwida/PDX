import time
import os
import h5py
import numpy as np


class TicToc:
    def __init__(self, ms=True, verbose=False):
        self._tic = time.time()
        self.ms = ms
        self.verbose = verbose

    def reset(self):
        self.tic()

    def tic(self) -> None:
        self._tic = time.time()

    def toc(self, tic=True):
        tictoc = time.time() - self._tic
        if self.ms: tictoc *= 1000
        if tic: self._tic = time.time()
        return tictoc


def read_hdf5_data(path):
    hdf5_file_name = os.path.join(path)
    hdf5_file = h5py.File(hdf5_file_name, "r")
    return np.array(hdf5_file["train"], dtype=np.float32), np.array(hdf5_file["test"], dtype=np.float32)
