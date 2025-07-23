from enum import Enum

# Some constants
class PDXConstants:
    PDX_VECTOR_SIZE = 64
    PDX_CENTROIDS_VECTOR_SIZE = 64
    DEFAULT_DELTA_D = 32
    HORIZONTAL_DIMENSIONS_GROUPING = 64
    X4_GROUPING = 4
    U8_MAX = 255 # TODO: Fix for Intel can only by max of 127 (there are mathematical workarounds)
    VERTICAL_PROPORTION_DIM = 0.75
    PDXEARCH_VECTOR_SIZE = 10240  # TODO: Parametrize per algorithm, probably ADSampling can do less
    SUPPORTED_METRICS = [
        "l2sq"
    ]


class PDXDistanceMetrics(Enum):
    l2sq = 1

