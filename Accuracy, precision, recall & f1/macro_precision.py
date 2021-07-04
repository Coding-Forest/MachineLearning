def macro_precision(model):
  mac_precision = sum(micro_precisions(model))/4
  return mac_precision
