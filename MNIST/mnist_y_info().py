# ------------------------------------------------------- Basic setup ---------------------------------------------------------
import numpy as np
%matplotlib inline
import matplotlib
import matplotlib.pyplot as plt

# for classification
from sklearn.linear_model import SGDClassifier

# for cross validation
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone

# ---------------------------------------------------- Load and split MNIST ----------------------------------------------------
# The code in this section is from "Hands-On Machine Learning with Scikit-Learn & TensorFlow" by Aurélien Géron.

# load MNIST
from sklearn.datasets import fetch_openml
mnist_784 = fetch_openml('mnist_784', version=1)

# split data into X (image data) and y (label data)
X,  y = mnist_784["data"], mnist_784['target']
X = X.astype(np.int8)
y = y.astype(np.int8)


# ---------------------------------------------------- Code here. ----------------------------------------------------

def mnist_y_info():
  sum = 0
  len_y = len(y)
  sum_pct = 0
  nums = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']

  print('---- MNIST y label info ----\n')
  print('{:<10} {:<9} {}'.format('digit', 'count', 'percent'))
  print('{:<10} {:<9} {}'.format('------', '------', '-------'))

  for i in range(10):
    count = len(y[y == i])
    pct = round(count / len_y * 100, 3)
    print('{} ({}):  {:<7}  {}%'.format(i, nums[i], count, pct))
    sum = sum + count
    sum_pct = sum_pct + pct

  print('\ntotal count: ', sum)  
  print('total pct: {} %'.format(round(sum_pct, 2)))

mnist_info()
