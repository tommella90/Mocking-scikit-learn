#%%

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from create_data import CreateData
from linear_regression import LinearRegression
# from base_estimator import BaseEstimator

import os

def is_package(directory):
    """Check if a directory contains an __init__.py file, indicating it's a Python package."""
    return "__init__.py" in os.listdir(directory)

# Get the current working directory
current_directory = os.getcwd()

# List all items in the current directory
items = os.listdir(current_directory)

print("Python packages in '", current_directory, "':")
for item in items:
    item_path = os.path.join(current_directory, item)
    if os.path.isdir(item_path) and is_package(item_path):
        print(item)

#%%
import linear_regression as lr
dir(lr)

#%%
data_generator = CreateData(n_samples=100, n_features=1, noise=0.2)
X, y = data_generator.create_data()
data_generator.plot_data(X, y)

slr = lr.LinearRegression()
slr.fit(X, y)
y_pred = slr.predict(X)

plt.plot(X[:, 0], y, 'o', label='X[:, 0]')
plt.plot(X, y_pred, 'o', label='X[:, 0]')


slr.__dict__

# %%
base_estimator = BaseEstimator(slr)
print(base_estimator.get_params())

base_estimator.set_params(intercept_=True)
print(base_estimator.get_params())

getattr(slr, 'intercept_')


#%%
state = slr.__getstate__()
state['model_personal_name'] = 'MyLinearRegression'
state
slr.__setstate__(state)
slr.__dict__

# %%
import pickle

with open('pickle/slr.pkl', 'wb') as f:
    pickle.dump(slr, f)

# %%
with open('pickle/slr.pkl', 'rb') as f:
    deserialized_slr = pickle.load(f)

deserialized_slr.__dict__

# %%
slr._check_n_features(X)
# %%
df = pd.DataFrame(X, columns=['feature_1'])

#%%
df.__dataframe__().column_names()


# %%


# %%
X = np.array([[1, 2], [2, 3], [3, 5]])
y = np.array([1, 2, 3])
model = LinearRegression()
model.fit(X, y)


# %%
