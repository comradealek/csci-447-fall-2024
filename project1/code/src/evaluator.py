
def zeroOneLoss(cm : list[list[int]]) -> float:
  """
  Returns the zero one loss of a given confusion matrix
  :param cm:
      the confusion matrix to rate
  """
  try:
    n = 0
    h = 0
    for x in range(len(cm)):
      for y in range(len(cm[x])):
        if x == y:
          h += cm[x][y]
        n += cm[x][y]
    loss = (n - h) / n
  except:
    loss = 0
  return loss

def microPrecision(cm : list[list[int]]) -> float:
  """
  Returns the micro precision of a given confusion matrix
  :param cm:
      the confusion matrix to rate
  """
  try:
    tp = 0
    tpfp = 0
    for x in range(len(cm)):
      tp += cm[x][x]
      tpfp += sum(cm[x])
    loss = tp / tpfp
  except:
    loss = 0
  return loss

def microRecall(cm : list[list[int]]) -> float:
  """
  Returns the micro recall of a given confusion matrix
  :param cm:
      the confusion matrix to rate
  """
  try:
    tp = 0
    tpfn = 0
    for x in range(len(cm)):
      tp += cm[x][x]
      for y in range(len(cm)):
        tpfn += cm[y][x]
    loss = tp / tpfn
  except:
    loss = 0
  return loss

def macroPrecision(cm : list[list[int]]) -> float:
  """
  Returns the macro precision of a given confusion matrix
  :param cm:
      the confusion matrix to rate
  """
  try:
    loss = 0
    for x in range(len(cm)):
      loss += (cm[x][x] / sum(cm[x]))
    loss = loss / len(cm)
  except:
    loss = 0
  return loss

def macroRecall(cm : list[list[int]]) -> float:
  """
  Returns the macro recall of a given confusion matrix
  :param cm:
      the confusion matrix to rate
  """
  try:
    loss = 0
    for x in range(len(cm)):
      tp = cm[x][x]
      tpfn = 0
      for y in range(len(cm)):
        tpfn += cm[y][x]
      loss += (tp / tpfn)
    loss = loss / len(cm)
  except:
    return 0
  return loss

def macroFmeasure(cm : list[list[int]]) -> float:
  """
  Returns the macro F1 measure of a given confusion matrix
  :param cm:
      the confusion matrix to rate
  """
  precision = macroPrecision(cm)
  recall = macroPrecision(cm)
  try:
    fmeasure = 2 * (precision * recall) / (precision + recall)
  except:
    fmeasure = 0
  return fmeasure

def printMetrics(cm : list[list[int]]):
  """
  Prints the scores for a given confusion matrix

  :param cm:
      the confusion matrix to use for printing the scores
  """
  print(f'{"Precision:":11} {macroPrecision(cm):.4f}')
  print(f'{"Recall:":11} {macroRecall(cm):.4f}')
  print(f'{"Zero-One:":11} {zeroOneLoss(cm):.4f}')
  print(f'{"F Measure:":11} {macroFmeasure(cm):.4f}')