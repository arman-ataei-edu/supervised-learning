import math
import numpy as np
from enum import Enum

from typing import Dict

class Metrics (Enum):
    Man = '_Manhatan'
    Euc = '_Uclidean'
    MinK = '_Minkowski'
    Lp = '_Lp'
 

class Metric:
    def __init__(self, metric:Metrics):
        metric = metric.value
        metrics = {
            '_Manhatan':self._Manhatan,
            '_Uclidean': self._Uclidean,
            '_Lp': self._Lp,
            '_Minkowski': self._Minkowski
        }
        # print(metric)
        self.calc = metrics[metric]
    
    def _Manhatan(self, u:np.ndarray, v:np.ndarray):
        pass

    def _Uclidean(self, u:np.ndarray, v:np.ndarray):
        assert u.shape == v.shape
        l = u.size 
        s = 0
        for i in range(l):
            s += (abs(u[i]-v[i]))**2
        
        return np.sqrt(s) 
    
    def _Lp(self, u:np.ndarray, v:np.ndarray, p:float=1):
        # print(u.shape, v.shape)
        assert u.shape == v.shape
        assert p>=1
        # print(u,v,p)
        l = u.size 
        # print(l)
        s = 0
        for i in range(l):
            # print(u[i], v[i])
            # print(u[i]-v[i])
            s += math.pow(abs(u[i]-v[i]), p)
            # print(s)
        # print(s)
        u_v_dist = np.power(s, 1/p)
        # print(u_v_dist)
        return u_v_dist
    
    def _Minkowski(self, u:np.ndarray, v:np.ndarray):
        assert u.shape == v.shape
        l = u.size 
        s = 0
        for i in range(l):
            pass

        u_v_dist = 0
        return u_v_dist