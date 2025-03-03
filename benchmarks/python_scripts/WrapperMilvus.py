from pymilvus import MilvusClient


class IVFMilvus:
    def __init__(self, metric, dimension, dataset, force_create=False):
        if metric not in ("euclidean"):
            raise NotImplementedError("IVFMilvus doesn't support metric %s" % metric)
        self.metric = metric

        self._metric = metric
        self.name = "IVFMILVUS()"
        self._nbrs = None
        self.dimension = dimension
        self.force_create = force_create

        # self.client = MilvusClient("milvus_demo.db")
        self.client = MilvusClient("http://localhost:19530")
        self.dataset = dataset.replace("-", "_") + '_ivf'
        self.just_created = False

        if self.force_create:
            self.client.drop_collection(collection_name=self.dataset)

        # If the collection doesn't exist, we must create it
        if not self.client.has_collection(collection_name=self.dataset):
            self.client.drop_collection(collection_name=self.dataset)
            self.client.create_collection(
                collection_name=self.dataset,
                metric_type="L2",
                dimension=self.dimension,
                segment_row_limit=4000000,
            )
            self.just_created = True
        else:
            print('Loading collection')
            self.client.load_collection(self.dataset)

    def fit(self, X):
        if self.just_created or self.force_create:  # Batch insert if it's a new collection
            print('Generating')
            data = [
                {"id": i, "vector": X[i]}
                for i in range(len(X))
            ]
            print('Inserting')
            i = 0
            for _ in range(0, len(data), 1000):
                print(_)
                if i + 1000 > len(data):
                    break
                self.client.insert(collection_name=self.dataset, data=data[i: i+1000])
                i += 1000
            if i != len(data):
                self.client.insert(collection_name=self.dataset, data=data[i: len(data)])
        # self.client.load_collection(self.dataset)

    def add(self, X):
        print('Adding rest')
        data = [
            {"id": i, "vector": X[i]}
            for i in range(len(X))
        ]
        print('Inserting rest')
        i = 0
        for _ in range(0, len(data), 1000):
            print(_)
            if i + 1000 > len(data):
                break
            self.client.insert(collection_name=self.dataset, data=data[i: i+1000])
            i += 1000
        if i != len(data):
            self.client.insert(collection_name=self.dataset, data=data[i: len(data)])

    def create_index(self, nlist):
        if self.just_created or self.force_create:
            print(nlist)
            print(self.client.list_indexes(collection_name=self.dataset))
            default_idx_name = self.client.list_indexes(collection_name=self.dataset)[0]
            self.client.release_collection(
                collection_name=self.dataset
            )
            print('Dropping index')
            self.client.drop_index(
                collection_name=self.dataset,
                index_name=default_idx_name
            )
            index_params = [{
                "field_name": "vector",
                "index_name": "benchmarking_ivf",
                "index_type": "IVF_FLAT",
                "metric_type": "L2",  # L2 (Euclidean Distance) or IP (Inner Product)
                "params": {"nlist": nlist},  # Number of clusters for IVF
            }]
            print('Creating IVF')
            self.client.create_index(
                collection_name=self.dataset,
                index_params=index_params
            )
        else:
            print('IVF index already created')
        # self.client.load_collection(self.dataset)

    def get_index_n_buckets(self):
        res = self.client.describe_index(
            collection_name=self.dataset,
            index_name="benchmarking_ivf"
        )
        return int(res['nlist'])


    def query_index(self, v, n, nprobe):
        results = self.client.search(
            collection_name=self.dataset,
            data=v,
            search_params={
                "metric_type": "L2",
                "params": {
                    "nprobe": nprobe
                }
            },
            limit=n
        )
        return results

    def query(self, v, n):
        results = self.client.search(
            collection_name=self.dataset,
            data=v,
            search_params={"metric_type": "L2", "params": {}},  # Leave parameters empty for exact search
            limit=n
        )
        return results

    def release(self):
        self.client.release_collection(
            collection_name=self.dataset
        )
