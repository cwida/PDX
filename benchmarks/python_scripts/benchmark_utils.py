import time
import os
import csv
import math
import numpy as np

SOURCE_DIR = os.getcwd()
ARCHITECTURE = os.environ.get('PDX_ARCH', 'DEFAULT')
RESULTS_DIRECTORY = os.path.join(SOURCE_DIR, "benchmarks", "results", ARCHITECTURE)
KNN = 100
N_MEASURE_RUNS = 1

IVF_NPROBES = [
    512, 256, 224, 192, 160, 144, 128,
    112, 96, 80, 64, 56, 48, 40,
    32, 28, 26, 24, 22, 20, 18, 16, 14, 12, 10, 8, 6, 4, 2
]

EF_SEARCHES = [
    512, 384, 256, 224, 192, 160, 144, 128,
    112, 96, 80, 64, 56, 48, 40,
    32, 28, 26, 24, 22, 20, 18, 16, 14, 12, 10, 8, 6, 4, 2
]


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


# We remove extreme outliers on both sides (Q3 + 1.5*IQR & Q1 - 1.5*IQR)
def save_results(runtimes: list, results_path, metadata):
    write_header = True
    if os.path.exists(results_path):
        write_header = False

    all_min_runtime = min(runtimes)
    all_max_runtime = max(runtimes)
    all_sum_runtime = sum(runtimes)

    min_runtime = math.inf
    max_runtime = -math.inf
    sum_runtime = 0

    Q1 = int(len(runtimes) / 4)
    Q2 = int(len(runtimes) / 2)
    Q3 = Q1 + Q2
    runtimes.sort()
    iqr = runtimes[Q3] - runtimes[Q1]
    accounted_queries = 0
    for runtime in runtimes:
        if runtime > runtimes[Q3] + 1.5 * iqr:
            continue
        if runtime < runtimes[Q1] - 1.5 * iqr:
            continue
        min_runtime = min(min_runtime, runtime)
        max_runtime = max(max_runtime, runtime)
        sum_runtime += runtime
        accounted_queries += 1

    all_avg_runtime = all_sum_runtime / len(runtimes)
    avg_runtime = sum_runtime / accounted_queries
    print(metadata.get('dataset'), " --------------")
    print('Total avg:', np.average(np.array(runtimes)))
    print('Total median:', np.median(np.array(runtimes)))
    print('avg:', avg_runtime)
    print('max:', max_runtime)
    print('min:', min_runtime)
    print('recall:', metadata.get('recall'))

    f = open(results_path, 'a')
    writer = csv.writer(f)
    if write_header:
        writer.writerow([
            "dataset", "algorithm", "avg", "max", "min", "recall",
            "knn", "n_queries", "num_measure_runs",
            "avg_all", "max_all", "min_all",
            "ef_search", "M", "ef_construction", "ivf_nprobe"
        ])
    writer.writerow([
        metadata.get('dataset'), metadata.get('algorithm'), avg_runtime, max_runtime, min_runtime,
        metadata.get('recall'),
        KNN, metadata.get('n_queries'), N_MEASURE_RUNS,
        all_avg_runtime, all_max_runtime, all_min_runtime,
        metadata.get('ef_search', 0), metadata.get('M', 0), metadata.get('ef_construction', 0),
        metadata.get('ivf_nprobe', 0)
    ])
    f.close()


def disable_multithreading():
    os.environ['MKL_NUM_THREADS'] = "1"
    os.environ['NUMEXPR_NUM_THREADS'] = "1"
    os.environ['OMP_NUM_THREADS'] = "1"
    os.environ['VECLIB_MAXIMUM_THREADS'] = "1"
    os.environ['OPENBLAS_NUM_THREADS'] = "1"
