import preprocessor as prpr
import numpy as np

class Classifier:
  def __init__(self, data : prpr.ProcessedData) -> None:
    self.table = []
    self.d = 0
    self.n = 0
    self.train(data)

  def train(self, data : prpr.ProcessedData, d = None, n = 1, debug = False):
    table = []
    table = [[0] for _ in range(data.numberOfClasses)]
    for i in range(data.numberOfClasses):
      table[i].extend([[0, 0] for _ in range(data.vectorLength - 1)])
    e = data.numberOfExamples
    if d == None:
      d = data.vectorLength - 1
    for i in range(0, e):
      row = data.vectorList[i]
      table[row[-1]][0] += 1
      for j in range(0, data.vectorLength - 1):
        table[row[-1]][j + 1][row[j]] += 1
    for i in range(data.numberOfClasses):
      C = table[i][0]
      for j in range(1, data.vectorLength):
        table[i][j][0] = (table[i][j][0] + n) / (C + d)
        table[i][j][1] = (table[i][j][1] + n) / (C + d)
      table[i][0] /= e
    self.table = table
    self.d = d
    if debug:
      self.printTable()

  def classify(self, vector : list[int], debug = False) -> int:
    classification = 0
    numberOfClasses = len(self.table)
    classestimates = []
    for x in range(0, numberOfClasses):
      classestimate = self.table[x][0]
      for y in range(0, len(vector) - 1):
        classestimate *= self.table[x][y + 1][vector[y]]
      classestimates.append(classestimate)
    for x in range(0, len(classestimates)):
      if classestimates[x] > classestimates[classification]:
        classification = x
    if debug:
      for x in classestimates:
        print(x)
    return classification
  
  def classifyLog(self, vector : list[int]) -> int:
    classification = 0
    numberOfClasses = len(self.table)
    classestimates = []
    for x in range(0, numberOfClasses):
      classestimate = self.table[x][0]
      for y in range(0, len(vector) - 1):
        classestimate += np.log(self.table[x][y + 1][vector[y]])
      classestimates.append(classestimate)
    for x in range(0, len(classestimates)):
      if classestimates[x] > classestimates[classification]:
        classification = x
    return classification

  def printTable(self):
    for row in self.table:
      print(row)