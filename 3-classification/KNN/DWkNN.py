# Weighted KNN

import numpy as np
from collections import Counter
from .Metrics import Metric, Metrics

class DWkNN:
    def __init__(self,distance_metric:Metrics, k:int=3, **kwargs):
        assert k >0 and k%2>0
        self.p = kwargs['p'] if kwargs and kwargs['p'] else None
        self.k = k
        if distance_metric == Metrics.Lp:
            if not self.p:
                return ValueError("please set the value for 'p'")
            
            self.metric = Metric(metric=distance_metric)
        if not distance_metric == Metrics.Lp:
            self.metric = Metric(metric=distance_metric)

    def fit(self, X_train:np.ndarray, y_train:np.ndarray):
        
        # assert X_train.shape[0] == y_train.shape[0]
        # self.train_shape = X_train.shape

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
        k_nearest_distances = [distances[i] for i in  k_indices]

        
        inverse_distances = [1 / (d + 1e-5) for d in k_nearest_distances]
        weighted_sum = sum(inv_dist * label for inv_dist, label in zip(inverse_distances, k_nearest_labels))
        sum_of_weights = sum(inverse_distances)
        
        return weighted_sum / sum_of_weights 
       
        # return weighted_votes.most_common(1)[0][0]

