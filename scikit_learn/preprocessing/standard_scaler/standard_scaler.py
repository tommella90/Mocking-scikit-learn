import numpy as np
from ...base import BaseEstimator, TransformerMixin
from ...utils.extmath import (
    _mean_std_axis0
)

class StandardScaler(BaseEstimator, TransformerMixin): 

    def __init__(self, attr1=False, attr2=False): 
        self.mean_ = None
        self.std_ = None
        self.attr1 = False
        self.attr2 = False

    def _reset(self):
        self.mean_ = None
        self.std_ = None
        pass 
    
    def fit(self, X, y=None):
        self._reset()
        X = self._validate_data(X)
        self.mean_, self.std_ = _mean_std_axis0(X, weights=None)
        if self.print:
            print(f"mean: {self.mean_}, std: {self.std_}")
        return self

    def transform(self, X, y=None):
        X = self._validate_data(X)

        if self.mean_ is None or self.std_ is None:
            raise ValueError("StandardScaler is not fitted yet")

        X -= self.mean_
        X /= self.std_
        return X        

    def inverse_transform(self, X, y=None):
        X = self._validate_data(X)
        if self.mean_ is None or self.std_ is None:
            raise ValueError("StandardScaler is not fitted yet")
        X *= self.std_
        X += self.mean_
        return X
    
