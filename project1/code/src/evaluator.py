
def zeroOneLoss(table : list[list[int]]) -> float:
  n = 0
  h = 0
  for x in range(len(table)):
    for y in range(len(table[x])):
      if x == y:
        h += table[x][y]
      n += table[x][y]
  loss = h / n
  return loss

def microPrecision(table : list[list[int]]) -> float:
  tp = 0
  tpfp = 0
  for x in range(len(table)):
    tp += table[x][x]
    tpfp += sum(table[x])
  try:
    loss = tp / tpfp
  except:
    loss = -1
  return loss

def microRecall(table : list[list[int]]) -> float:
  tp = 0
  tpfn = 0
  for x in range(len(table)):
    tp += table[x][x]
    for y in range(len(table)):
      tpfn += table[y][x]
  try:
    loss = tp / tpfn
  except:
    loss = -1
  return loss

def macroPrecision(table : list[list[int]]) -> float:
  loss = 0
  for x in range(len(table)):
    try:
      loss += (table[x][x] / sum(table[x]))
    except:
      return -1
  loss = loss / len(table)
  return loss

def macroRecall(table : list[list[int]]) -> float:
  loss = 0
  for x in range(len(table)):
    tp = table[x][x]
    tpfn = 0
    for y in range(len(table)):
      tpfn += table[y][x]
    try:
      loss += (tp / tpfn)
    except:
      return -1
  loss = loss / len(table)
  return loss