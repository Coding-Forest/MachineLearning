def macro_recall(model):
  mac_recall = sum(micro_recalls(model))/4
  return mac_recall
