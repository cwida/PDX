import json
import sys
from setup_utils import *
from setup_settings import *
from benchmark_utils import *
from WrapperMilvus import *

disable_multithreading()


if __name__ == '__main__':
    RESULTS_PATH = os.path.join(RESULTS_DIRECTORY, "IVF_MILVUS.csv")
    IVF_NPROBE = 0
    arg_dataset = ""
    if len(sys.argv) > 1:
        arg_dataset = sys.argv[1]
    if len(sys.argv) > 2:
        IVF_NPROBE = int(sys.argv[2])  # controls recall of search
    for dataset in DATASETS:
        if len(arg_dataset) and dataset != arg_dataset:
            continue
        print('Milvus:', dataset, '====================================')
        dimensionality = DIMENSIONALITIES[dataset]
        gt_name = os.path.join(SEMANTIC_GROUND_TRUTH_PATH, get_ground_truth_filename(dataset, KNN))
        searcher = IVFMilvus("euclidean", dimensionality, dataset)

        total_nlist = searcher.get_index_n_buckets()

        queries = read_hdf5_test_data(dataset)

        for ivf_nprobe in IVF_NPROBES:
            print('Nprobe: ', ivf_nprobe)
            if IVF_NPROBE > 0 and IVF_NPROBE != ivf_nprobe:
                continue
            if ivf_nprobe > total_nlist:
                continue
            runtimes = []
            recalls = []
            clock = TicToc()

            # Measure time with the rest of the queries only executed once
            print('Querying Measure...')
            for i in range(N_MEASURE_RUNS):
                for q in queries:
                    q = np.ascontiguousarray(q)
                    q_for_milvus = [q]  # Single query
                    clock.tic()
                    searcher.query_index(q_for_milvus, KNN, nprobe=ivf_nprobe)
                    runtimes.append(clock.toc())

            # Measure recall afterwards to not affect cache
            print('Loading GT...')
            gt = json.load(open(gt_name, 'r'))

            print('Measuring recall')
            query_i = 0
            for q in queries:
                q_for_milvus = [q]  # Single query
                points = searcher.query_index(q_for_milvus, KNN, nprobe=ivf_nprobe)
                indices = [p['id'] for p in points[0]]
                recalls.append(float(len(set(indices).intersection(set(gt[str(query_i)])))) / KNN)
                query_i += 1

            metadata = {
                'dataset': dataset,
                'n_queries': len(queries),
                'algorithm': 'ivf_milvus',
                'recall': sum(recalls) / float(len(recalls)),
                'ivf_nprobe': ivf_nprobe
            }
            save_results(runtimes, RESULTS_PATH, metadata)
        searcher.release()