import numpy as np
from ...preprocessing.standard_scaler import StandardScaler


def test_standard_scaler(): 
        np.random.seed(0)
        X = np.random.rand(100, 1) 
        y = 1.5 * X[:, 0] + np.random.randn(100) * 0.2

        scaler = StandardScaler()
        scaler.fit(X)
        X = scaler.transform(X)

        assert X is not None, "Transformed data is not set" 
        assert X.shape == (100, 1), "Transformed data is not of expected shape"
        
        # assert that the mean of the transformed data is close to 0
        np.testing.assert_allclose(
            actual=X.mean(), 
            desired=0, 
            atol=0.3, 
            err_msg="Mean of transformed data is not as expected"
        )

        # assert that the standard deviation of the transformed data is close to 1
        np.testing.assert_allclose(
            actual=X.std(), 
            desired=1, 
            atol=0.3, 
            err_msg="Standard deviation of transformed data is not as expected"
        )
