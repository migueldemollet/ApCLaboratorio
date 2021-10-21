from sklearn.datasets import make_regression
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats
import seaborn as sns


x = [1,2,3,4,5,6,7,1,1,1,1,1]
y = [1,2,3,5,6,7,5,5,5,5,55,5]

plt.figure()
plt.bar(x,y)
plt.show()