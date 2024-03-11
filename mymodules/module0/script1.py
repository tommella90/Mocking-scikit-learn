import numpy as np

class MockStandardizer:
    def __init__(self):
        print('MockPreprocessor.__init__ called')

    def standardize_data(self, data):
        data_standardized = (data - np.mean(data)) / np.std(data)
        return data_standardized