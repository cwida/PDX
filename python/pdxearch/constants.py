from enum import Enum

# Some constants
class PDXConstants:
    PDX_VECTOR_SIZE = 64
    PDX_CENTROIDS_VECTOR_SIZE = 64
    DEFAULT_DELTA_D = 32
    PDXEARCH_VECTOR_SIZE = 10240  # TODO: Parametrize per algorithm, probably ADSampling can do less
    SUPPORTED_METRICS = [
        "l2sq"
    ]


class PDXDistanceMetrics(Enum):
    l2sq = 1

