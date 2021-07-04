def macro_recall(model):
  mac_recall = sum(micro_recalls(model)) / len(model)
  return mac_recall
