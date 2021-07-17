import numpy as np
import math 

iris = [1.43, -0.4, 0.23]

def softmax(arr, decimal_place=None):
  # Calculate the softmaxed output
  softmax_vals = []
  denominator = sum(np.exp(iris))

  # Get the softmax value of every input
  for i in range(0, len(arr)):
    softmaxed = np.exp(arr[i]) / denominator
    softmax_vals.append(softmaxed)

  # round the decimals  
  if decimal_place == None:
    return softmax_vals
  else:
    return np.round(softmax_vals, decimal_place)

softmax(iris, 3) 
# array([0.684, 0.11 , 0.206])
