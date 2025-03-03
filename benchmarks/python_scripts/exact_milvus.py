from benchmark_utils import *
from setup_scripts.setup_utils import *
from setup_scripts.WrapperBruteForce import BruteForceMilvus

disable_multithreading()

if __name__ == '__main__':
    RESULTS_PATH = os.path.join(RESULTS_DIRECTORY, "EXACT_MILVUS.csv")
    arg_dataset = ""
    if len(sys.argv) > 1:
        arg_dataset = sys.argv[1]
    for dataset in DATASETS:
        if len(arg_dataset) and dataset != arg_dataset:
            continue
        print('Milvus:', dataset, '====================================')
        runtimes = []
        clock = TicToc()

        data, queries = read_hdf5_data(dataset)
        data = np.ascontiguousarray(data)

        searcher = BruteForceMilvus("euclidean", len(data[0]), dataset)
        # DELETE LATER:
        searcher.just_created = True
        searcher.fit(data)
        searcher.release()
        continue
        # UNTIL HERE

        searcher_gt = BruteForceSKLearn("euclidean")

        for _ in range(N_MEASURE_RUNS):
            for q in queries:
                q = np.ascontiguousarray(q)
                q_for_milvus = [q]  # Single query
                clock.tic()
                searcher.query(q_for_milvus, KNN)
                runtimes.append(clock.toc())
        metadata = {
            'dataset': dataset,
            'n_queries': len(queries),
            'algorithm': 'milvus',
            'recall': 1.0
        }
        save_results(runtimes, RESULTS_PATH, metadata)

        searcher.release()


