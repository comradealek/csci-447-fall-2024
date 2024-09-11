import preprocessor as prpr
import copy
import numpy as np
import trainer as tr
import random

def tenfold(data : prpr.ProcessedData) -> list[prpr.ProcessedData]:
  return kfold(data, 10)

def kfold(data : prpr.ProcessedData, k : int, debug = False) -> list[prpr.ProcessedData]:
  dataList = []
  vectors = copy.deepcopy(data.vectorList)
  random.shuffle(vectors)
  vectors.sort(key=lambda x: int(x[-1]))
  splitVectorList = [[] for _ in range(k)]
  for i in range(data.numberOfExamples):
    splitVectorList[i % k].append(vectors[i])
  if debug:
    for chunk in splitVectorList:
      for vector in chunk:
        print(vector)
      print()
  for x in range(k):
    chunk = prpr.ProcessedData("__DEFAULT__")
    chunk.numberOfExamples = len(splitVectorList[x])
    chunk.vectorLength = data.vectorLength
    chunk.subvectorLengths = copy.copy(data.subvectorLengths)
    chunk.vectorList = list(splitVectorList[x])
    chunk.numberOfClasses = data.numberOfClasses
    dataList.append(chunk)
  return dataList

def crossvalidation(data : prpr.ProcessedData) -> list[list[int]]:
  k = 10
  validationTable = [([0] * data.numberOfClasses) for _ in range(data.numberOfClasses)]
  cleanDataList = kfold(data, k)
  for x in range(0, k):
    dataList = copy.copy(cleanDataList)
    testData = dataList.pop(x)
    trainingData = mergedata(dataList)
    classifier = tr.Classifier(trainingData)
    for vector in testData.vectorList:
      actualClass = vector[-1]
      predictedClass = classifier.classify(vector)
      validationTable[predictedClass][actualClass] += 1
  return validationTable

def mergedata(dataList : list[prpr.ProcessedData]) -> prpr.ProcessedData:
  totalData = prpr.ProcessedData("__DEFAULT__")
  totalData.vectorLength = dataList[0].vectorLength
  totalData.numberOfClasses = dataList[0].numberOfClasses
  totalData.subvectorLengths = dataList[0].subvectorLengths
  totalData.numberOfExamples = 0
  for data in dataList:
    totalData.vectorList.extend(data.vectorList)
    totalData.numberOfExamples += data.numberOfExamples
  return totalData

def printTable(data : prpr.ProcessedData, table : list[list[int]]):
  l = 0
  for name in data.classNames:
    if l < len(name):
      l = len(name)
  numL = 0
  for row in table:
    for val in row:
      n = int(np.ceil(np.log10(val + 1)))
      if n > numL:
        numL = n
  for x in range(len(table)):
    fstr = f'{(str(data.classNames[x]) + ":"):<{l + 2}} ['
    fstr += f'{table[x][0]:>{numL}}'
    for y in range(1, len(table[x])):
      fstr += f',{table[x][y]:>{numL + 1}}'
    fstr += ']'
    print(fstr)
