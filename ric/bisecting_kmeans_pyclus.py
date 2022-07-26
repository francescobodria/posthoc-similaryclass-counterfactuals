import heapq
import numpy as np

from pyclustering.cluster.kmeans import kmeans
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
from pyclustering.utils.metric import type_metric, distance_metric

from scipy.spatial.distance import pdist, cdist

from haversine import haversine


class BisectingKMeans:
    
    def __init__(
            self,
            k=2,
            min_cluster_size=1,
            min_split_size=3,
            max_distance_thr=-np.inf,
            max_nbr_clusters_iter=100,
            max_iter=300,
            metric="euclidean",
            random_state=None
        ):
        
        self.k = k
        self.min_cluster_size = min_cluster_size
        self.min_split_size = min_split_size
        self.max_distance_thr = max_distance_thr
        self.max_nbr_clusters_iter = max_nbr_clusters_iter
        self.max_iter = max_iter
        self.metric = metric
        self.random_state = random_state

        if self.metric == 'euclidean':
            self.metric_pyclus = distance_metric(type_metric.EUCLIDEAN)
            self.metric_dist = self.metric
        elif self.metric == 'euclidean_square':
            self.metric_pyclus = distance_metric(type_metric.EUCLIDEAN_SQUARE)
            self.metric_dist = 'sqeuclidean'
        elif self.metric == 'manhattan':
            self.metric_pyclus = distance_metric(type_metric.MANHATTAN)
            self.metric_dist = 'cityblock'
        elif self.metric == 'chebyshev':
            self.metric_pyclus = distance_metric(type_metric.CHEBYSHEV)
            self.metric_dist = self.metric
        elif self.metric == 'minkowski':
            self.metric_pyclus = distance_metric(type_metric.MINKOWSKI)
            self.metric_dist = self.metric
        elif self.metric == 'canberra':
            self.metric_pyclus = distance_metric(type_metric.CANBERRA)
            self.metric_dist = self.metric
        # elif self.metric == 'chisquare':
        #     self.metric_pyclus = distance_metric(type_metric.CHI_SQUARE)
        #     self.metric_dist = self.metric
        # elif self.metric == 'gower':
        #     self.metric_pyclus = distance_metric(type_metric.GOWER)
        #     self.metric_dist = self.metric
        elif self.metric == 'haversine':
            self.metric_pyclus = distance_metric(type_metric.USER_DEFINED, func=haversine)
            self.metric_dist = haversine
        else:
            raise Exception('Unknown metric %s' % self.metric)

        self.labels_ = None
        self.cluster_centers_ = None
        self.sse_ = None
        self.sse_list_ = None
        self.sse_history_ = None
        self.n_iter_ = None

    def fit(self, X):

        self.labels_ = -np.ones(len(X)).astype(int)
        self.cluster_centers_ = list()
        self.sse_list_ = list()
        self.sse_history_ = list()
        self.n_iter_ = 0

        queue = list()
        heapq.heappush(queue, (-len(X), np.arange(len(X))))

        cluster_id = 0
        for i in range(self.max_iter):
            # print('iter', i, len(queue), cluster_id)

            indexes = heapq.heappop(queue)[1]
            # print(len(indexes))
            X_it = X[indexes]

            initial_centers = kmeans_plusplus_initializer(X_it, self.k, random_state=self.random_state).initialize()
            # print('initial_centers', initial_centers)
            kmeans_instance = kmeans(X_it, initial_centers, metric=self.metric_pyclus)
            kmeans_instance.process()

            clusters_it = kmeans_instance.get_clusters()
            # print('nbr clusters', len(clusters_it))
            # print(clusters_it)
            centers_it = kmeans_instance.get_centers()
            sse_it = kmeans_instance.get_total_wce()
            self.sse_history_.append(sse_it)

            for j in range(self.k):
                C_j = X[indexes[clusters_it[j]]]
                # print(j, len(C_j))

                if len(C_j) <= self.min_cluster_size:
                    # print('a')
                    continue  # cluster with less than min_cluster_size points, i.e., noise

                add_to_result_set = len(C_j) <= self.min_split_size

                if not add_to_result_set:
                    max_intra_cluster_dist = np.max(pdist(C_j, metric=self.metric_dist))
                    add_to_result_set = max_intra_cluster_dist <= self.max_distance_thr

                if add_to_result_set:
                    # print('b')
                    cluster_sse = np.sum(cdist(np.array([centers_it[j]]), C_j))
                    self.labels_[indexes[clusters_it[j]]] = cluster_id
                    self.cluster_centers_.append(centers_it[j])
                    self.sse_list_.append(cluster_sse)
                    cluster_id += 1
                else:
                    # print(-len(C_j))
                    # heapq.heappush(queue, (-len(C_j), indexes[clusters_it[j]]))
                    cluster_sse = np.sum(cdist(np.array([centers_it[j]]), C_j))
                    heapq.heappush(queue, (-cluster_sse, indexes[clusters_it[j]]))

            # print(cluster_id, self.max_nbr_clusters_iter)
            current_nbr_clusters = cluster_id+1 + len(queue)
            if len(queue) == 0 or current_nbr_clusters >= self.max_nbr_clusters_iter:
                break

            self.n_iter_ = i
            # print('-----\n')

        # print('----->', len(self.cluster_centers_))

        while len(queue) > 0:   # handles clusters remained in the queue
            indexes = heapq.heappop(queue)[1]
            C = X[indexes]
            center = np.mean(C, axis=0)
            cluster_sse = np.sum(cdist(np.array([center]), C))
            self.labels_[indexes] = cluster_id
            self.cluster_centers_.append(center)
            self.sse_list_.append(cluster_sse)
            cluster_id += 1

        self.sse_ = np.sum(self.sse_list_)
        self.cluster_centers_ = np.array(self.cluster_centers_)

    def fit_predict(self, X):
        self.fit(X)
        return self.labels_
