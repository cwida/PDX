import faiss
import json
import sys
from sklearn import preprocessing
from benchmark_utils import *
from setup_utils import *
from setup_settings import *

DATASETS_TO_USE = [
    'openai-1536-angular',
    'agnews-mxbai-1024-euclidean',
    'instructorxl-arxiv-768',
    'simplewiki-openai-3072-normalized',
    'msong-420',
    'llama-128-ip',
]
if __name__ == '__main__':
    RESULTS_PATH = os.path.join(RESULTS_DIRECTORY, "IVF_FAISS.csv")
    arg_dataset = ""
    IVF_NPROBE = 0
    if len(sys.argv) > 1:
        arg_dataset = sys.argv[1]
    if len(sys.argv) > 2:
        IVF_NPROBE = int(sys.argv[2])  # controls recall of search
    if not len(DATASETS_TO_USE): DATASETS_TO_USE = DATASETS
    for dataset in DATASETS_TO_USE:
        if len(arg_dataset) and dataset != arg_dataset:
            continue
        dimensionality = DIMENSIONALITIES[dataset]
        index_name = os.path.join(CORE_INDEXES_FAISS, get_core_index_filename(dataset))
        gt_name = os.path.join(SEMANTIC_GROUND_TRUTH_PATH, get_ground_truth_filename(dataset, 100))

        disable_multithreading()
        faiss.omp_set_num_threads(1)

        queries = read_hdf5_test_data(dataset)
        queries = preprocessing.normalize(queries, axis=1, norm='l2')

        print('Restoring index...')
        index = faiss.read_index(index_name)
        print('Index restored...')

        nprobes_to_use = []
        if IVF_NPROBE:
            nprobes_to_use = [IVF_NPROBE]
        else :
            nprobes_to_use = IVF_NPROBES

        for ivf_nprobe in nprobes_to_use:
            print('Nprobe: ', ivf_nprobe)
            if IVF_NPROBE > 0 and IVF_NPROBE != ivf_nprobe:
                continue
            if ivf_nprobe > index.nlist:
                continue
            runtimes = []
            recalls = []
            clock = TicToc()
            index.nprobe = ivf_nprobe

            print('Querying Measure...')
            for i in range(N_MEASURE_RUNS):
                for q in queries:
                    q = np.ascontiguousarray(np.array([q]))
                    clock.tic()
                    index.search(q, KNN)
                    runtimes.append(clock.toc())

            # Measure recall afterwards to not affect cache
            gt = json.load(open(gt_name, 'r'))
            query_i = 0
            for q in queries:
                _, matches = index.search(np.ascontiguousarray(np.array([q])), KNN)
                recalls.append(float(len(set(matches[0]).intersection(set(gt[str(query_i)][:KNN])))) / KNN)
                query_i += 1

            metadata = {
                'dataset': dataset,
                'n_queries': len(queries),
                'algorithm': 'ivf_faiss',
                'recall': sum(recalls) / float(len(recalls)),
                'ivf_nprobe': ivf_nprobe
            }
            save_results(runtimes, RESULTS_PATH, metadata)


