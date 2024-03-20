
import numpy as np
import inspect

from ..utils.validation import (
    _num_features,
    _get_feature_names
)  

class BaseEstimator:

    def _get_param_names(self):
        # Use inspect.signature() to get the parameters of the __init__ method
        init_signature = inspect.signature(self.__init__)
        # Exclude 'self' and return a list of parameter names
        return [p.name for p in init_signature.parameters.values() if p.name != 'self']

    def get_params(self, deep=True):
        out = dict()
        for key in self._get_param_names():
            value = getattr(self, key, None)  # Use None as default if attribute is not set
            if deep and hasattr(value, 'get_params'):
                deep_items = value.get_params().items()
                out.update((key + '__' + k, val) for k, val in deep_items)
            else:
                out[key] = value
        return out

    def set_params(self, **params): 
        if not params:
            return self
        # Set the parameters of the model
        valid_params = self.get_params()
        for key, value in params.items(): 
            if key not in valid_params:
                raise ValueError(f"Invalid parameter {key} for estimator {self.__class__.__name__}")
            setattr(self, key, value)     
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
    
    def _validate_names(self, names):
        if len(set(names)) != len(names):
            raise ValueError("Names should be unique")
        
        invalid_names = [name for name in names if not name.isidentifier()]
        if invalid_names:
            raise ValueError(f"Invalid names: {invalid_names}")

    def _validate_data(self, X, y=None, **params):
        if y is not None:
            if X.shape[0] != y.shape[0]:
                raise ValueError("X and y must have the same number of samples")
            return X, y
        return X
    
    def _check_n_features(self, X):
        try:
            n_features = _num_features(X)
        except TypeError as e: 
            raise ValueError(f"X should be array-like or sequence, got {type(X)}") from e
        return n_features
    
    def _check_feature_names(self, X):
        feature_names = _get_feature_names(X)
        return feature_names


class TransformerMixin(): 
    def fit_transform(self, X, y=None, **fit_params): 
        if y is None: 
            return self.fit(X, **fit_params).transform(X)
        else:
            return self.fit(X, y, **fit_params).transform(X, y)


class RegressorMixin(): 
    def score(self, X, y): 
        y_pred = self.predict(X)
        return r2_score(y, y_pred)


@staticmethod
def r2_score(y, y_pred):
    return 1 - np.sum((y - y_pred)**2) / np.sum((y - np.mean(y))**2)
