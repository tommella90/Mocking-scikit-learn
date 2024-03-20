#%%
from ..base import BaseEstimator
from itertools import islice

__all__ = ['Pipeline'] 
    

class Pipeline(BaseEstimator):

    def __init__(self, steps):
        super().__init__()  
        self.steps = steps

    def _validate_steps(self):
        names, estimators = zip(*self.steps)

        self._validate_names(names)

        transformers = estimators[:-1]
        estimator = estimators[-1]

        for t in transformers: 
            if not (hasattr(t, "fit") or hasattr(t, "fit_transform")) or not hasattr(t, "transform"):
                raise TypeError("All intermediate steps should be transformers (and have fit and transform methods)", 
                                f"{t} attributes found are: {t.__dict__.keys()}")
        
        if not (hasattr(estimator, "fit") or hasattr(estimator, "predict")):
            raise TypeError("The final estimator should be a transformer (and implement the fit and predict methods)", 
                            f"{estimator} attributes found are: {estimator.__dict__.keys()}")
            

    @property
    def _final_estimator(self):
        """Return the final estimator."""
        return self.steps[-1][1]


    def fit(self, X, y=None, **params): 
        self._validate_steps()

        for name, transformer in self.steps[:-1]:
            transformer.fit(X, y)
            X = transformer.fit_transform(X, y, **params)

        self._final_estimator.fit(X, y, **params)
        return self


    def fit_predict(self, X, y=None, **fit_params):
        self.fit(X, y, **fit_params)
        return self.predict(X, **fit_params)


    def predict(self, X, **params):
        for name, transformer in self.steps[:-1]:
            X = transformer.transform(X)
        return self._final_estimator.predict(X, **params)
    
    def get_params(self, deep=True):
        values = getattr(self, "steps")
        out = dict()
        for key, value in values:
            params = value.get_params()
            for param in params:
                out[key + "__" + param] = param
        return out
    
    @property
    def _named_steps(self):
        return dict(self.steps)

def set_params(self, **params):
    for name, value in params.items():
        if '__' in name:
            step, param = name.split('__', 1)
            if step not in self._named_steps:
                raise ValueError(f"Step '{step}' not found in pipeline steps")
            step_obj = self._named_steps[step]
            if hasattr(step_obj, 'set_params'):
                step_obj.set_params(**{param: value})
            else:
                setattr(step_obj, param, value)
        else:
            setattr(self, name, value)
    return self

    def _get_params_values(self, **params):
        params = self.get_params(**params)
        out = dict()
        for k, value in params.items():
            out[k] = value
        return out
    
    def get_final_estimator_params(self, **params):
        return self._final_estimator.get_params(**params)


    def _route_params(self, fit_params):
        routed_params = {}
        for name, params in fit_params.items():
            if '__' in name:
                step, param = name.split('__', 1)
                if step not in dict(self.steps):
                    raise ValueError(f"Step {step} not found in pipeline steps")
                if step not in routed_params:
                    routed_params[step] = {}
                routed_params[step][param] = params
            else:
                # Handle or ignore parameters without a step prefix
                pass  # Placeholder for handling global parameters or raising warning/error
        return routed_params
