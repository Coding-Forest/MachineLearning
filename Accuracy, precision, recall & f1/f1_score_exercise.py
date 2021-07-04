""" 
# F-1 Score for Multi-Class Classification

## Exercise problems

Exercise 1. Prediction robots' performance comparison (Minsuk Heo 허민석, 2017))

robot1 = [[100, 80, 10, 10],
          [0, 9, 0, 1],
          [0, 1, 8, 1],
          [0, 1, 0, 9]]
robot2 = [[198, 2, 0, 0],
          [7, 1, 0, 2],
          [0, 8, 1, 1],
          [2, 3, 4, 1]]

-----------

Exercise 2. A sample (Baeldung, 2020)
samples = [[50, 3, 0, 0],
           [26, 8, 0, 1],
           [20, 2, 4, 0],
           [12, 0, 0, 1]]


References
- Minsuk Heo 허민석 (2017) [머신러닝] 다중 분류 모델 성능 측정 (accuracy, f1 score, precision, recall on multiclass classification) https://www.youtube.com/watch?v=8DbC39cvvis&t=576s
- Baeldung (2020) F-1 Score for Multi-Class Classification https://www.baeldung.com/cs/multi-class-f1-score
-------------------------------------------------------------------------------------------------------------------"""

import numpy as np

def micro_precisions(model):
  micro_precisions = []
  length = len(model)
  for i in range(0, length):
    precision = model[i][i] / sum(model[i])
    micro_precisions.append(precision)
  return micro_precisions


def micro_recalls(model):
  length = len(model)
  arr = np.array(model)   # convert list into array because transpose works with array only. 
  transposed_matrix = arr.transpose()
  micro_recalls = []
  for i in range(0, length):
    recall = transposed_matrix[i][i] / sum(transposed_matrix[i])
    micro_recalls.append(recall)
  return micro_recalls


def macro_precision(model):
  mac_precision = sum(micro_precisions(model)) / len(model)
  return mac_precision


def macro_recall(model):
  mac_recall = sum(micro_recalls(model))/ len(model)
  return mac_recall


def f1_score(model):
  mac_prec = macro_precision(model)
  mac_rec = macro_recall(model)
  f1 = 2 * (mac_prec * mac_rec) / (mac_prec + mac_rec)
  return f1
