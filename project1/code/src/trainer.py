import preprocessor as prpr

class Classifier:
  def __init__(self, data : prpr.ProcessedData) -> None:
    self.table = []
    self.d = 0
    self.train(data)
  
  def train(self, data : prpr.ProcessedData):
    self.d = data.vectorLength - 1
    e = data.numberOfExamples
    for x in range(0, e):
      vector = data.vectorList[x]
      vectorclass = vector[-1]
      while vectorclass >= len(self.table):
        head = [0]
        section = [[0,0] for _ in range(self.d)]
        head.extend(section)
        self.table.append(head)
      self.table[vectorclass][0] += 1
      for y in range(0, len(vector) - 1):
        attribute = vector[y]
        try:
          self.table[vectorclass][y + 1][attribute] += 1
        except:
          print(str(x) + " " + str(y))
          print(self.table[vectorclass])
    
    for x in range(0, len(self.table)):
      C = self.table[x][0]
      for y in range (1, self.d + 1):
        self.table[x][y][0] = (self.table[x][y][0] + 1) / (C + self.d)
        self.table[x][y][1] = (self.table[x][y][1] + 1) / (C + self.d)
      self.table[x][0] /= e
  
  def classify(self, vector : list[int]) -> int:
    classification = 0
    numberOfClasses = len(self.table)
    classestimates = []
    for x in range(0, numberOfClasses):
      classestimate = self.table[x][0]
      for y in range(0, self.d):
        classestimate *= self.table[x][y + 1][vector[y]]
      classestimates.append(classestimate)
    for x in range(0, len(classestimates)):
      if classestimates[x] > classestimates[classification]:
        classification = x
    return classification
  
  def printTable(self):
    for row in self.table:
      print(row)