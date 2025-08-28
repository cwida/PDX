import numpy as np

class PredicateEvaluator:
    def __init__(self):
        self.n_passing_tuples = np.array([], dtype=np.uint32)
        self.selection_vector = np.array([], dtype=np.uint8)

    def evaluate_predicate(self, passing_tuples_ids: np.ndarray, labels: np.ndarray):
        all_ordered_labels = np.concatenate([np.array(sub) for sub in labels])
        selection_vector = np.zeros(all_ordered_labels.shape[0], dtype=np.uint8)
        selection_vector_pos = np.isin(all_ordered_labels, passing_tuples_ids).nonzero()[0]
        selection_vector[selection_vector_pos] = 1

        n_passing_tuples = np.zeros((len(labels)), dtype=np.uint32)
        offset = 0
        for i, label_list in enumerate(labels):
            n_labels = len(label_list)
            n_passing_tuples[i] = np.sum(selection_vector[offset:offset + n_labels])
            offset += n_labels
        self.n_passing_tuples = n_passing_tuples
        self.selection_vector = selection_vector
