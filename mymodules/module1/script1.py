import numpy as np

class MockNormalizer:
    def __init__(self):
        print('MockPreprocessor.__init__ called')

    def normalize_data(self, data):
        data_normalized = data / np.max(data)
        return data_normalized