import preprocessor as prpr
import numpy as np

class Classifier:
  """
  
  """
  def __init__(self, data : prpr.ProcessedData) -> None:
    self.table = []
    self.d = [] #d values per attribute
    self.a = 0 #alpha
    self.e = 0 #number of examples
    self.train(data)

  def train(self, data : prpr.ProcessedData, debug = False):
    """
    Builds the table used for calculating classifications

    :param data: ProcessedData
        The data that is being used for building the model
    :param debug: bool
        Optional flag that will print the table when model is trained if it
        is True
    """
    table = [[0] for _ in range(data.numberOfClasses)]
    for i in range(data.numberOfClasses):
      table[i].extend([[0, 0] for _ in range(data.vectorLength - 1)])
    self.e = data.numberOfExamples
    a = 1
    d = []
    for svl in data.subvectorLengths:
      d.extend([svl for _ in range(svl)])
    for i in range(0, self.e):
      row = data.vectorList[i]
      table[row[-1]][0] += 1
      for j in range(0, data.vectorLength - 1):
        table[row[-1]][j + 1][row[j]] += 1
    self.table = table
    self.d = d
    self.a = a
    if debug:
      self.printTable()

  def classify(self, vector : list[int], debug = False) -> int:
    """
    Calculates scores for a given vector and returns the class with the highest
    score

    :param vector: list[int]
        the feature vector that is being classified
    :param debug: bool
        optional flag that prints the classestimates if True
    """
    classification = 0
    numberOfClasses = len(self.table)
    classestimates = []
    for x in range(0, numberOfClasses):
      classestimate = self.table[x][0] / self.e
      for y in range(0, len(vector) - 1):
        classestimate *= (self.table[x][y + 1][vector[y]] + self.a) / (self.table[x][0] + (self.d[y] * self.a))
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
      classestimate = np.log(self.table[x][0] / self.e)
      for y in range(0, len(vector) - 1):
        classestimate += np.log((self.table[x][y + 1][vector[y]] + self.a) / (self.table[x][0] + (self.d[y] * self.a)))
      classestimates.append(classestimate)
    for x in range(0, len(classestimates)):
      if classestimates[x] > classestimates[classification]:
        classification = x
    return classification

  def printTable(self):
    """
    Prints the counting table
    """
    for row in self.table:
      printstr = f'*[{row[0]}, '
      for i in range(1, len(row)):
        printstr += f'[{row[i][0]}, '
        printstr += f'{row[i][1]}]'
        if (i + 1) < len(row):
          printstr += ", "
      printstr += "]"
      print(printstr)
  
  def printWeightedTable(self):
    """
    Prints the table used for calculating the class score
    """
    for i in range(len(self.table)):
      printstr = f'*[{self.table[i][0] / self.e:.2f}, '
      for j in range(1, len(self.table[i])):
        printstr += f'[{(self.table[i][j][0] + self.a) / (self.table[i][0] + (self.d[j - 1] * self.a)):.2f}, '
        printstr += f'{(self.table[i][j][1] + self.a) / (self.table[i][0] + (self.d[j - 1] * self.a)):.2f}]'
        if (j + 1) < len(self.table[i]):
          printstr += ", "
      printstr += "]"
      print(printstr)
  
  def printDebugInfo(self):
    """
    Prints the values in the class
    """
    print(self.a)
    print(self.d)
    print(self.e)
    for i in range(len(self.table)):
      print(self.table[i][0])