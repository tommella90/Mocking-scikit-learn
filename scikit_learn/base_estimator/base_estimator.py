
import numpy as np
import pandas as pd

from ..utils.validation import (
    _num_features,
    _get_feature_names
)  

class BaseEstimator:

    def __init__(self, model=None):
        self.model = model

    def _get_param_names(self):
        # Get the parameter names of the model
        return [name for name in dir(self.model) if name.endswith('_') and not name.startswith('__')]

    def get_params(self): 
        # Get the parameters of the model
        out = dict()
        names = self._get_param_names()
        for key in names: 
            out[key] = getattr(self.model, key)
        return out
    
    def set_params(self, **params): 
        # Set the parameters of the model
        valid_params = self.get_params()
        for key, value in params.items(): 
            if key not in valid_params:
                raise ValueError(f"Invalid parameter {key} for estimator {self.__class__.__name__}")
            setattr(self.model, key, value)     
            valid_params[key] = value
        return self
    
    def __getstate__(self) -> object:
        # __getstate__ is called when you use the pickle module to serialize the object
        # __dict__ is a dictionary that contains all the instance attributes
        state = self.__dict__.copy()
        return state
    
    def __setstate__(self, state) -> None:
        # __setstate__ is called when you use the pickle module to deserialize the object
        self.__dict__.update(state)
        return self

    def _validate_data(self, X, y):
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of samples")
        return X, y
    
    def _check_n_features(self, X):
        try:
            n_features = _num_features(X)
        except TypeError as e: 
            raise ValueError(f"X should be array-like or sequence, got {type(X)}") from e
        return n_features
    
    def _check_feature_names(self, X):
        feature_names = _get_feature_names(X)
        return feature_names

