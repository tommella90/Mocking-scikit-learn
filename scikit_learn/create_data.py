import numpy as np
import matplotlib.pyplot as plt


class CreateData:
    def __init__(self, n_samples, n_features, noise):
        self.n_samples = n_samples
        self.n_features = n_features
        self.noise = noise

    def create_data(self):
        np.random.seed(0)
        n_samples = 100
        X = np.random.rand(n_samples, 1) 
        y = 1.5 * X[:, 0] + np.random.randn(n_samples) * 0.2
        return X, y

    def plot_data(self, X, y):
        plt.plot(X, y, 'o', label='X[:, 0]')
        plt.legend(loc='best')
        plt.show()
    

