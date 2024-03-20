#%%
import numpy as np
from ..base import BaseEstimator, RegressorMixin

class LinearRegression(BaseEstimator, RegressorMixin):

    def __init__(self):
        self.coef_ = None
        self.intercept_ = None

    def fit(self, X, y):
        # Calculate coefficients (coef_) and intercept
        X, y = self._validate_data(X, y)
        X_with_intercept = np.c_[np.ones(len(X)), X]  # Add intercept term
        beta = np.linalg.inv(X_with_intercept.T @ X_with_intercept) @ X_with_intercept.T @ y
        self.intercept_ = beta[0]
        self.coef_ = beta[1:]


        if self.coef_ is None or self.intercept_ is None:
            raise ValueError("LinearRegression is not fitted yet")

        return self

    def predict(self, X, return_params_as_tuple=False):
        # For simplicity, let's assume X is a 2D numpy array
        # Use the learned coefficients and intercept to make predictions
        X = self._validate_data(X)

        if self.coef_ is None or self.intercept_ is None:
            raise ValueError("LinearRegression is not fitted yet")
        
        if return_params_as_tuple:
            PARAMS = (self.coef_, self.intercept_)
            return np.dot(X, self.coef_) + self.intercept_, PARAMS

        return np.dot(X, self.coef_) + self.intercept_


