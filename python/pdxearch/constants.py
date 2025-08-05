from enum import Enum
import ctypes.util

# Some constants
class PDXConstants:
    PDX_VECTOR_SIZE = 64
    PDX_CENTROIDS_VECTOR_SIZE = 64
    D_THRESHOLD_FOR_DCT_ROTATION = 512
    HORIZONTAL_DIMENSIONS_GROUPING = 64
    X4_GROUPING = 4
    U8_MAX = 255 # TODO: Fix for Intel can only by max of 127 (there are mathematical workarounds)
    VERTICAL_PROPORTION_DIM = 0.75
    PDXEARCH_VECTOR_SIZE = 10240 # Probably ADSampling can do less and find benefits
    SUPPORTED_METRICS = [
        "l2sq"
    ]
    HAS_FFTW = ctypes.util.find_library("fftw3f") is not None


class PDXDistanceMetrics(Enum):
    l2sq = 1

