import heapq
from sklearn.cluster import KMeans

class BisectingKMeans:
    
    def __init__(
            self,
            k=2,
            min_cluster_size=1,
            min_split_size=2,
            max_distance_thr=0.0,
            max_nbr_clusters_iter=100,
            max_iter=300,
            random_state=None
        ):
        
        self.k = k
        self.min_cluster_size = min_cluster_size
        self.min_split_size = min_split_size
        self.max_distance_thr = max_distance_thr
        self.max_nbr_clusters_iter = max_nbr_clusters_iter
        self.max_iter = max_iter
        self.random_state = random_state

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

        nbr_clusters = 0
        for i in range(self.max_iter):
            #print('iter', i, len(queue), nbr_clusters)
            
            if len(queue) == 0:
                break
            
            indexes = heapq.heappop(queue)[1]
            X_it = X[indexes]
            #print(len(X_it), len(X), type(indexes))

            kmeans = KMeans(n_clusters=self.k)
            kmeans.fit(X_it)
            
            _, sizes = np.unique(kmeans.labels_, return_counts=True)
            if np.any(sizes < self.min_cluster_size):
                continue
            
            self.labels_[indexes] = kmeans.labels_ + nbr_clusters

            self.sse_history_.append(kmeans.inertia_)
            nbr_clusters += self.k

            for j in range(self.k):
                C_j = X[indexes[kmeans.labels_ == j]]
                #print(j, len(C_j), '<----')
                
                cluster_sse = np.sum(cdist(np.array([kmeans.cluster_centers_[j]]), C_j))
                cluster_sse += np.random.random()*0.000001
                
                if self.max_distance_thr > 0.0:
                    max_intra_cluster_dist = np.max(pdist(C_j))
                else:
                    max_intra_cluster_dist = np.inf
                
                if len(C_j) <= self.min_split_size or max_intra_cluster_dist <= self.max_distance_thr: # if a cluster must not be split anymore
                    self.cluster_centers_.append(kmeans.cluster_centers_[j])
                    self.sse_list_.append(cluster_sse)
                else:
                    heapq.heappush(queue, (-cluster_sse, indexes[kmeans.labels_ == j]))

            if len(queue) == 0 or nbr_clusters >= self.max_nbr_clusters_iter:
                break

            self.n_iter_ = i

        self.sse_ = np.sum(self.sse_list_)
        self.cluster_centers_ = np.array(self.cluster_centers_)

    def fit_predict(self, X):
        self.fit(X)
        return self.labels_
