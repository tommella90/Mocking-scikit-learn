import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils import check_array
from sklearn.utils.validation import FLOAT_DTYPES


def _get_column_as_tensor(s: pd.Series):
    """Get every normal or TensorArray column as a 2D array."""
    try:
        return s.tensor.values
    except AttributeError:  # normal column
        return s.values.reshape(-1, 1)


def _as_list_of_str(columns):
    """Return none, one or more columns as a list."""
    columns = columns if columns else []
    if isinstance(columns, str):
        columns = [columns]
    return columns


class CallableMixin:
    """Makes transformer callable."""

    def __call__(self, *args, **kwargs):
        return self.transform(*args, **kwargs)  # type: ignore


class IdentityRegressor(BaseEstimator, RegressorMixin, CallableMixin):
    """Use to turn a Pipeline into an estimator."""

    def fit(self, X, y=None):
        return self

    def predict(self, X):
        return X

    def fit_predict(self, X, y=None):
        self.fit(X, y)
        return self.predict(X)


def transform_selected(X, transform, dtype, selected="all", copy=True, retain_order=False):
    """Apply a transform function to portion of selected features."""
    X = check_array(X, accept_sparse="csc", copy=copy, dtype=FLOAT_DTYPES)

    if sparse.issparse(X) and retain_order:
        raise ValueError("The retain_order option can only be set to True " "for dense matrices.")

    if isinstance(selected, str) and selected == "all":
        return transform(X)

    if len(selected) == 0:
        return X

    n_features = X.shape[1]
    ind = np.arange(n_features)
    sel = np.zeros(n_features, dtype=bool)
    sel[np.asarray(selected)] = True
    not_sel = np.logical_not(sel)
    n_selected = np.sum(sel)

    if n_selected == 0:
        # No features selected.
        return X
    elif n_selected == n_features:
        # All features selected.
        return transform(X)
    else:
        X_sel = transform(X[:, ind[sel]])
        # The columns of X which are not transformed need
        # to be casted to the desire dtype before concatenation.
        # Otherwise, the stacking will cast to the higher-precision dtype.
        X_not_sel = X[:, ind[not_sel]].astype(dtype, copy=False)

    if retain_order:
        if X_sel.shape[1] + X_not_sel.shape[1] != n_features:
            raise ValueError(
                "The retain_order option can only be set to True "
                "if the dimensions of the input array match the "
                "dimensions of the transformed array."
            )

        # Fancy indexing not supported for sparse matrices
        X[:, ind[sel]] = X_sel
        return X

    if sparse.issparse(X_sel) or sparse.issparse(X_not_sel):
        return sparse.hstack((X_sel, X_not_sel))
    else:
        return np.hstack((X_sel, X_not_sel))
