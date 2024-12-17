import numpy as np
from collections import Counter
from .Metrics import Metric, Metrics
class KNN:
    def __init__(self,distance_metric:Metrics, k:int=3,weighted=False, **kwargs):
        assert k >0 and k%2>0
        self.p = kwargs['p']
        self.k = k
        self.weighted = weighted
        if distance_metric == Metrics.Lp:
            if not self.p:
                return ValueError("please set the value for 'p'")
            
            self.metric = Metric(metric=distance_metric)
        if not distance_metric == Metrics.Lp:
            self.metric = Metric(metric=distance_metric)

    def fit(self, X_train:np.ndarray, y_train:np.ndarray):
        
        # assert X_train.shape[0] == y_train.shape[0]
        self.train_shape = X_train.shape

        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test:np.ndarray):
        # assert X_test.shape[0] == self.train_shape
        predictions = [self._predict(x) for x in X_test]
        return np.array(predictions)

    def _predict(self, x:np.ndarray):
        # Compute distances between x and all examples in the training set
        # assert x.shape[1:] == self.train_shape[1:]
        if self.p :
            distances = [self.metric.calc(x, x_train, self.p) for x_train in self.X_train]
        if not self.p:
            distances = [self.metric.calc(x, x_train) for x_train in self.X_train]

        # Get the k nearest samples, their labels
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
       
        # Calculating the weighted vote
        if self.weighted:
            k_nearest_distances = [distances[i] for i in  k_indices]
            weights = [1 / (d + 1e-5) for d in k_nearest_distances]
            # calculating the voted weight of all labels
            weighted_votes = Counter() 
            for label, weight in zip(k_nearest_labels, weights): 
                weighted_votes[label] += weight 
            # print(len(weighted_votes.keys()))
            # returning the most weighted vote
            # print(weighted_votes)
            return weighted_votes.most_common(1)[0][0]
        
        # Return the most common class label
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]

