import preprocessor as prpr
import copy
import numpy as np
import trainer as tr

def tenfold(data : prpr.ProcessedData) -> list[prpr.ProcessedData]:
  return kfold(data, 10)

def kfold(data : prpr.ProcessedData, k : int) -> list[prpr.ProcessedData]:
  dataList = []
  data.shuffleVectors()
  splitVectorList = copy.deepcopy(data.vectorList)
  splitVectorList = np.array_split(splitVectorList, k)
  for x in range(0, k):
    chunk = prpr.blankData()
    chunk.numberOfExamples = len(splitVectorList[x])
    chunk.vectorLength = data.vectorLength
    chunk.subvectorLengths = copy.copy(data.subvectorLengths)
    chunk.vectorList = list(splitVectorList[x])
    chunk.numberOfClasses = data.numberOfClasses
    dataList.append(chunk)
  return dataList

def crossvalidation(data : prpr.ProcessedData) -> list[list[int]]:
  validationTable = [([0] * data.numberOfClasses) for _ in range(data.numberOfClasses)]
  cleanDataList = tenfold(data)
  for x in range(10):
    dataList = copy.copy(cleanDataList)
    testData = dataList.pop(x)
    trainingData = mergedata(dataList)
    classifier = tr.Classifier(trainingData)
    for vector in testData.vectorList:
      actualClass = vector[-1]
      predictedClass = classifier.classify(vector)
      validationTable[actualClass][predictedClass] += 1
  return validationTable

def mergedata(dataList : list[prpr.ProcessedData]) -> prpr.ProcessedData:
  totalData = prpr.blankData()
  totalData.vectorLength = dataList[0].vectorLength
  totalData.numberOfClasses = dataList[0].numberOfClasses
  totalData.subvectorLengths = dataList[0].subvectorLengths
  totalData.numberOfExamples = 0
  for data in dataList:
    totalData.vectorList.extend(data.vectorList)
    totalData.numberOfExamples += data.numberOfExamples
  return totalData