import scipy
from scipy import stats

print(f"✅ SciPy version: {scipy.__version__}")
# Example usage of stats to ensure it's resolved
print(f"✅ Stats module test: {stats.norm.cdf(0)}")