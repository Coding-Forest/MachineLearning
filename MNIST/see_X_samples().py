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
X = X.astype(np.int64)
X = X. reshape(70000, 28,28)



# ---------------------------------------------------- the visualiser function ----------------------------------------------------

import matplotlib.pyplot as plt
import seaborn as sn

# ---- hyper parameter conditions ----
# 'start' ranges between 0 - 69999.
# 'end' ranges bewteen 1-70000.
# start < end

def see_samples(start, end, annot=False):
      
  if (annot==True): 
    for i in range(start, end):
      plt.figure(figsize=(10,10))
      ax = sn.heatmap(X[i], 
                        annot=True,
                        fmt='d',
                        cbar=False)
      plt.axis('off')

  else:
    for i in range(start, end):
      plt.figure(figsize=(3,3))   
      ax = sn.heatmap(X[i], 
                      annot=False,
                      cbar=False)
      plt.axis('off')
      

# ---------------------------------------------------- End of function ----------------------------------------------------
      
      
# Change start & end numbers as you like. 
see_samples(50, 54)
see_samples(100,102, annot=True)
