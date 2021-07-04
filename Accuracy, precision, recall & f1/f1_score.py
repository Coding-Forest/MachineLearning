def f1_score(model):
  mac_prec = macro_precision(model)
  mac_rec = macro_recall(model)
  f1 = 2 * (mac_prec * mac_rec) / (mac_prec + mac_rec)
  return f1
