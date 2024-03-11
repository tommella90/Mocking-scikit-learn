import numpy as np
import pandas as pd

def _num_features(X):
    # Get the number of features in X
    if not hasattr(X, '__len__') and not hasattr(X, 'shape'):
        if not hasattr(X, '__array__'):
            raise TypeError("Expected sequence or array-like, got %s" % type(X))
        # convert other types to array
        X = np.asarray(X) 

    if hasattr(X, 'shape'):
        if not hasattr(X.shape, "__len__") or len(X.shape)<=1:
            raise TypeError("X should be 2D array")
        return X.shape[1]
    
    
def _get_feature_names(X):
    # Get the feature names
    feature_names = None
    type_X = type(X)
    if type_X is pd.DataFrame:
        feature_names = np.asarray(X.columns)
    elif hasattr(X, "__dataframe__"):
        df_column_names = X.__dataframe__().column_names()
        feature_names = np.asarray(df_column_names)
    if feature_names is None or len(feature_names) == 0:
        raise ValueError("Could not determine feature names")