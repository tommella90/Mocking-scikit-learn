
#%%
import numpy as np

from ..base import BaseEstimator
from ..linear_regression import LinearRegression

def test_fit(): 
    np.random.seed(0)
    X = np.random.rand(100, 1) 
    y = 1.5 * X[:, 0] + np.random.randn(100) * 0.2

    model = LinearRegression()
    model.fit(X, y)

    assert model.coef_ is not None, "Coefficients are not set"
    assert model.intercept_ is not None, "Intercept is not set"

    np.testing.assert_allclose(
        actual=model.coef_, 
        desired=1.5,
        atol=0.3,
        err_msg="Coefficient is not as expected"
    )
    
    np.testing.assert_allclose(
        actual=model.intercept_, 
        desired=0, 
        atol=0.3, 
        err_msg="Intercept is not as expected"
    )

def test_predict():
    np.random.seed(0)
    X = np.random.rand(100, 1) 
    y = 1.5 * X[:, 0] + np.random.randn(100) * 0.2

    model = LinearRegression()
    model.fit(X, y)

    assert model.predict(X) is not None, "Predictions are not set"

    np.testing.assert_allclose(
        actual=model.predict(X), 
        desired=1.5 * X[:, 0], 
        atol=0.3, 
        err_msg="Predictions are not as expected"
    )



