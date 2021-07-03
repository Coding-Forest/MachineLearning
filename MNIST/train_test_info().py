# ------------------------------------------------------- Basic setup ---------------------------------------------------------
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone

from sklearn.linear_model import SGDClassifier

%matplotlib inline
import matplotlib
import matplotlib.pyplot as plt

# ---------------------------------------------------- Load and split MNIST ----------------------------------------------------
# Load data.
from sklearn.datasets import fetch_openml
mnist_784 = fetch_openml('mnist_784', version=1)

# split data into X (image data) and y (label data)
X,  y = mnist_784["data"], mnist_784['target']

# Further split X and y into train and test sets
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]
y_train = y_train.astype(np.int8)   # this is very IMPORTANT! y sets must be converted from str into int. 
y_test = y_test.astype(np.int8)


# ---------------------------------------------------- the function ----------------------------------------------------

def train_test_info(show_array=False):

  print('============= Train set =============')
  print('----- X train -----')
  print('Unique value count: ', len(np.unique(X_train)))
  if (show_array == True):
    print('Unique value array: \n{}'.format(np.unique(X_train)))

  print('\n----- y train -----')
  print('Unique value count: ', len(np.unique(y_train)))
  if (show_array == True): 
    print('Unique value array: {}'.format(np.unique(y_train)))


  print('\n\n============= Test set =============')
  print('----- X test -----')
  print('Unique value count: ', len(np.unique(X_test)))
  if (show_array == True):
    print('Unique value array: \n{}'.format(np.unique(X_test)))

  print('\n----- y test -----')
  print('Unique value count: ', len(np.unique(y_test)))
  if (show_array == True):
    print('Unique value array: {}'.format(np.unique(y_test)))


# As for X (both train and test), their arrays range from 0 - 255 (total count 256)
# because they express the intensity of the binary colour (white to black).

# As for y (both train and test), their arrays range from 0 - 9 (total count 10)
# because they are labels for each one-digit integers. 

train_test_info(show_array=True)
train_test_info()
