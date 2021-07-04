def micro_precisions(model):
  micro_precisions = []
  length = len(model)
  for i in range(0, length):
    precision = model[i][i] / sum(model[i])
    micro_precisions.append(precision)
  return micro_precisions
