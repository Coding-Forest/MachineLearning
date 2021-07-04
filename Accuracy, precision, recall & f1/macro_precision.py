def macro_precision(model):
  mac_precision = sum(micro_precisions(model)) / len(model)
  return mac_precision
