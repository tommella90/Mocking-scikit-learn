#%%

import numpy as np

def _mean_std_axis0(X, weights=None):
    if X.dtype not in [np.float32, np.float64]:
        X = X.astype(np.float64)  # Corrected typo: np.flaot64 to np.float64
    
    if weights is None:
        means = np.mean(X, axis=0)
        stds = np.std(X, axis=0)
    else:
        normalized_weights = weights / np.sum(weights)
        means = np.average(X, axis=0, weights=normalized_weights)
        
        # Calculate weighted variance (the average of squared deviations from the mean)
        # Note: np.average can also compute the weighted average of squared deviations directly
        squared_deviations = (X - means)**2
        weighted_squared_deviations = np.average(squared_deviations, axis=0, weights=normalized_weights)
        
        # Standard deviation is the square root of variance
        stds = np.sqrt(weighted_squared_deviations)
    
    return means, stds




import numpy as np
random_data = np.random.rand(100, 5)
mean, std = _mean_std_axis0(random_data)
print(mean)
# %%
