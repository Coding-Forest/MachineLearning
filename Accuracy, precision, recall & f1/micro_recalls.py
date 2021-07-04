import numpy as np

def micro_recalls(model):
  length = len(model)
  arr = np.array(model)   # convert list into array because transpose works with array only. 
  transposed_matrix = arr.transpose()
  micro_recalls = []

  for i in range(0, length):
    recall = transposed_matrix[i][i] / sum(transposed_matrix[i])
    micro_recalls.append(recall)
  return micro_recalls
