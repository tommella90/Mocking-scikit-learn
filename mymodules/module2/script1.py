# relative import
from ..module1.script1 import MockNormalizer

# absolute import
from mymodules.module0.script1 import MockStandardizer

# class MockCalculator
class MockCalculator:
    def __init__(self):
        print('MockCalculator.__init__ called')

    def multiply(self, data, multiplier=2):
        normalizer = MockNormalizer()
        data_normalized = normalizer.normalize_data(data)
        result = data_normalized * multiplier
        return result
    
    def divide(self, data, divisor=2):
        standardizer = MockStandardizer()
        data_standardized = standardizer.standardize_data(data)
        result = data_standardized / divisor
        return result 
