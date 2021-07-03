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

# Further split X and y into train and test sets
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]
y_train = y_train.astype(np.int8)   # this is very IMPORTANT! y sets must be converted from str into int. 
y_test = y_test.astype(np.int8)

# suffle index. 
shuffle_index = np.random.permutation(60000)
X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]

# True for all 5s, False for all other digits.
y_train_5 = (y_train == 5)
y_test_5 = (y_test == 5)

# Create a binary classifier.
sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train, y_train_5)

# -------------------- Above is the prerequisite code to ensure the below cross_validation function work. ------------------------
# --------------------- Below is the function that shows info about the MNIST dataset. -------------------------------------------

def cross_validation():

  skfolds = StratifiedKFold(n_splits=3, random_state=None)
  i = 1

  # for the two data sets in the 3-fold subsets for X_ and y_trains,
  for train_index, test_index in skfolds.split(X_train, y_train_5):
    clone_clf = clone(sgd_clf)  # creates a clone of sgd classifier that binary-sorts 5 or non-5's.
                                # why do we need a new clone for every fold...?
                                # is it because once a clf is trained, it may output different result
                                # based on its previous training?
    print('-------- FOLD {} train set --------'.format(i))
    X_train_folds = X_train[train_index]
    y_train_folds = (y_train_5[train_index])
    print('X_train shape: ', X_train_folds.shape)
    print('X_train array: \n', X_train_folds, '\n')
    print('y_train shape: ', y_train_folds.shape)
    print('y_train array: ', y_train_folds, '\n')
    print('y_train unique values: ', np.unique(y_train_folds))
    print('y_train value counts: {} True\'s, {} False\'s \n'.format(
        len(y_train_folds[y_train_folds == True]), 
        len(y_train_folds[y_train_folds == False])))

    print('-------- FOLD {} test set --------'.format(i))
    X_test_fold = X_train[test_index]
    y_test_fold = (y_train_5[test_index])
    print('X_test shape: ', X_test_fold.shape)
    print('X_test array: \n', X_test_fold, '\n')
    print('y_test shape: ', y_test_fold.shape)    
    print('y_test array: ', y_test_fold, '\n')
    print('y_test unique values: ', np.unique(y_test_fold))    
    print('y_test value counts: {} True\'s, {} False\'s \n'.format(
        len(y_test_fold[y_test_fold == True]), 
        len(y_test_fold[y_test_fold == False])))

    print('-------- FOLD {} Prediction overview --------'.format(i))
    clone_clf.fit(X_train_folds, y_train_folds)
    y_pred = clone_clf.predict(X_test_fold)   
    print('Prediction array: ', y_pred)
    n_correct = sum(y_pred == y_test_fold)  # if prediction (clf's guessing job) is equal to y test fold (correct answer set)
    print('correct prediction count: {} out of {} samples'.format(n_correct, len(X_test_fold)))
    print('correct prediction ratio: {} %'.format(n_correct / len(y_pred)))
    print('\n====================== End of FOLD {} =====================\n\n'.format(i))
    i = i + 1

  print('======================= Index info =======================')
  print('train index \n-----------')
  print('length: ', len(train_index))
  print('array: ', train_index, '\n')
  print('test index \n----------')
  print('length: ', len(test_index))
  print('array: ', test_index, '\n')

cross_validation()
