import numpy as np
from collections import Counter
from .Metrics import Metric, Metrics
class KNN:
    def __init__(self,
                 distance_metric:Metrics, 
                 k:int=3,
                 weighted=False, 
                 centeralized=False, 
                 **kwargs):
        
        assert k >0 and k%2>0
        # currently we dont compute the centralized weighted KNN
        assert not weighted or not centeralized

        self.p = kwargs['p'] if kwargs and kwargs['p'] else None
        self.k = k 
        self.weighted = weighted
        self.centeralized = centeralized
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
        assert self.k < len(X_train)
        # computing the centers of classes(mean)
        

        if self.centeralized:
            labels = set(self.y_train)
            leaders = []
            assert self.k <= len(labels)
            l = len(self.X_train)

            for label in labels:
                mean = np.zeros(shape=X_train[0].shape)
                # number of elements in the "label" class
                n_class = 0
                for i in range(l):
                    if y_train[i] == label:    
                        mean += X_train[i]
                        n_class +=1
                # print(mean)
                leaders.append((label, mean/n_class, n_class))
        
            self.leaders = leaders

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
       
        # Calculating centralized weighted vote
        if self.centeralized and self.weighted:
            pass

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
        
        # calculating the votes w.r.t center of classes
        if self.centeralized:
            if self.p :
                distances = [self.metric.calc(x, leader[1], self.p) for leader in self.leaders]
            if not self.p:
                distances = [self.metric.calc(x, leader[1]) for leader in self.leaders]
            
            # Get the k nearest samples, their labels
            k_indices = np.argsort(distances)[:self.k]
            return self.leaders[k_indices[-1]][0]
            # k_nearest_labels = [self.leaders[i][0] for i in k_indices]

        # Return the most common class label
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]

    def score(self, y_test, y_pred):
        """
        the R² score
        R^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}
        """
        ss_total = np.sum((y_test - np.mean(y_test)) ** 2) 
        ss_residual = np.sum((y_test - y_pred) ** 2) 
        return 1 - (ss_residual / ss_total)