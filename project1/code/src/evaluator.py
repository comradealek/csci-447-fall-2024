
def zeroOneLoss(table : list[list[int]]) -> float:
  try:
    n = 0
    h = 0
    for x in range(len(table)):
      for y in range(len(table[x])):
        if x == y:
          h += table[x][y]
        n += table[x][y]
    loss = (n - h) / n
  except:
    loss = 0
  return loss

def microPrecision(table : list[list[int]]) -> float:
  try:
    tp = 0
    tpfp = 0
    for x in range(len(table)):
      tp += table[x][x]
      tpfp += sum(table[x])
    loss = tp / tpfp
  except:
    loss = 0
  return loss

def microRecall(table : list[list[int]]) -> float:
  try:
    tp = 0
    tpfn = 0
    for x in range(len(table)):
      tp += table[x][x]
      for y in range(len(table)):
        tpfn += table[y][x]
    loss = tp / tpfn
  except:
    loss = 0
  return loss

def macroPrecision(table : list[list[int]]) -> float:
  try:
    loss = 0
    for x in range(len(table)):
      loss += (table[x][x] / sum(table[x]))
    loss = loss / len(table)
  except:
    loss = 0
  return loss

def macroRecall(table : list[list[int]]) -> float:
  try:
    loss = 0
    for x in range(len(table)):
      tp = table[x][x]
      tpfn = 0
      for y in range(len(table)):
        tpfn += table[y][x]
      loss += (tp / tpfn)
    loss = loss / len(table)
  except:
    return 0
  return loss

def macroFmeasure(table : list[list[int]]) -> float:
  precision = macroPrecision(table)
  recall = macroPrecision(table)
  try:
    fmeasure = 2 * (precision * recall) / (precision + recall)
  except:
    fmeasure = 0
  return fmeasure

def printMetrics(table : list[list[int]]):
  print(f'{"Precision:":11} {macroPrecision(table):.4f}')
  print(f'{"Recall:":11} {macroRecall(table):.4f}')
  print(f'{"Zero-One:":11} {zeroOneLoss(table):.4f}')
  print(f'{"F Measure:":11} {macroFmeasure(table):.4f}')