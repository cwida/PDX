from benchmark_utils import *
from setup_scripts.setup_utils import *
from setup_scripts.setup_settings import *
from setup_scripts.WrapperBruteForce import BruteForceUsearch, BruteForceSKLearn

disable_multithreading()

if __name__ == '__main__':
    RESULTS_PATH = os.path.join(RESULTS_DIRECTORY, "EXACT_USEARCH.csv")
    arg_dataset = ""
    if len(sys.argv) > 1:
        arg_dataset = sys.argv[1]
    for dataset in DATASETS:
        if len(arg_dataset) and dataset != arg_dataset:
            continue
        runtimes = []
        clock = TicToc()

        data, queries = read_hdf5_data(dataset)
        data = np.ascontiguousarray(data)

        searcher = BruteForceUsearch("euclidean")
        searcher_gt = BruteForceSKLearn("euclidean")
        for _ in range(N_MEASURE_RUNS):
            for q in queries:
                q = np.ascontiguousarray(q)
                clock.tic()
                searcher.query(data, q, KNN)
                runtimes.append(clock.toc())
        metadata = {
            'dataset': dataset,
            'n_queries': len(queries),
            'algorithm': 'usearch',
            'recall': 1.0
        }
        save_results(runtimes, RESULTS_PATH, metadata)


