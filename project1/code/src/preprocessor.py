import io
import random
import copy
import pandas as pd

class ProcessedData:
  def __init__(self, filename : str):
    self.numberOfClasses = 0
    self.numberOfExamples = 0
    self.vectorLength = 0
    self.vectorList = list[list[int]]()
    self.subvectorLengths = list[int]()
    self.errorcode = 0
    self.classNames = list[str]()
    
    pdata = False
    if filename == "__DEFAULT__":
      return
    try:
      data_file = open("./data/" + filename + ".pdata", "r")

      select = input("Preprocessed data file found. Use it? y / n > ")
      if select == "n":
        raise
      pdata = True
    except:
      try:
        data_file = open("./data/" + filename + ".data", "r")
      except:
        print("failed to open file with name \"" + filename + "\"")
        self.errorcode = 1
        return
    if pdata:
      self.loadpdata(data_file)
    else:
      self.processfile(data_file)
      self.writetofile("./data/" + filename + ".pdata")
    self.vectorLength = len(self.vectorList[0])
  
  def printData(self):
    for vector in self.vectorList:
      print(vector)
    pass
  
  def writetofile(self, path):
    f = open(path, "w")
    #copy the subvector data
    f.write(str(self.classNames[0]))
    for x in range(1, len(self.classNames)):
      f.write("," + str(self.classNames[x]))
    f.write("\n")
    f.write(str(self.subvectorLengths[0]))
    for x in range(1, len(self.subvectorLengths)):
      f.write("," + str(self.subvectorLengths[x]))
    f.write("\n")
    #copy the vector data
    for li in self.vectorList:
      f.write(str(li[0]))
      for x in range(1, len(li)):
        f.write("," + str(li[x]))
      f.write("\n")
    f.close()
    pass
  
  def loadpdata(self, file: io.TextIOWrapper):
    #get the class names
    line = file.readline()
    line = line.strip()
    elems = line.split(',')
    for elem in elems:
      self.classNames.append(elem)
    #get the subvector lengths
    line = file.readline()
    line = line.strip()
    elems = line.split(',')
    for elem in elems:
      try:
        self.subvectorLengths.append(int(elem))
      except:
        print("Formatting error in file. Aborting")
        self.errorcode = 1
        return
    #get the rest of the data
    for line in file:
      line = line.strip()
      elems = line.split(',')
      vector = list()
      for elem in elems:
        try:
          vector.append(int(elem))
        except:
          print("Formatting error in file. Aborting")
          self.errorcode = 1
          return
        pass
      self.vectorList.append(vector)
    self.numberOfExamples = len(self.vectorList)
    for vector in self.vectorList:
      if vector[-1] + 1 > self.numberOfClasses:
        self.numberOfClasses = vector[-1] + 1
    self.vectorLength = len(self.vectorList[0])
    pass
  
  def processfile(self, file: io.TextIOWrapper, classColumnNum = None, columnCodes = None, missingAttribFlag = None, demo=False):
    #extract the column count from the data file
    line = str(file.readline())
    line = line.strip()
    elems = line.split(',')
    elementCount = len(elems)
    file.seek(0)
    
    #gets the column number for the class signifier
    if classColumnNum == None:
      code = 0
      while code == 0:
        inputNum = input("Which column corresponds to the class? > ")
        if inputNum.isdigit():
          classColumnNum = int(inputNum)
          if classColumnNum > elementCount:
            print("Out of range")
          else:
            code = 1
        else:
          print("Not a valid number")
    
    #flags each column as categorical, quantitative, or as the class signifier
    #-1 is for ignored columns, 0 is for continuous data, 1 is for categories, and 2 is for the class
    while columnCodes == None:
      columnCodes = list[int]()
      for x in range(0, elementCount):
        if x == classColumnNum - 1:
          columnCodes.append(2)
        else:
          inputValue = input("Is column " + str(x + 1) + " (c)ategorical, (q)uantitative, or (i)gnored > ")
          code = 0
          while code == 0:
            if inputValue == "c":
              columnCodes.append(1)
              code = 1
            elif inputValue == "q":
              columnCodes.append(0)
              code = 1
            elif inputValue == "i":
              columnCodes.append(-1)
              code = 1
            else:
              inputValue = input("Invalid entry. Please try again > ")
      #checks the validity of the flagging process
      if len(columnCodes) != elementCount:
        print("Error in processing file. Expected number of columns is " + str(elementCount) + " and number flagged is " + str(len(columnCodes)))
        columnCodes = None
    
    #ask for the missing attribute indicator
    if missingAttribFlag == None:
      missingAttribFlag = input("Please enter the flag for missing attributes > ")
    
    #for each column we initialize a list
    columnDataList = list[list]()
    for x in range(0, elementCount):
      if columnCodes[x] != -1:
        columnDataList.append(list[str]())
    
    #iterate through file line by line and add the elements to our data collections
    for line in file:
      line = line.strip()
      elems = line.split(",")
      
      x = 0
      y = 0
      for code in columnCodes:
        if code != -1:
          columnDataList[x].append(elems[y])
          x += 1
        else:
          pass
        y += 1
    
    #verify the lists are the same length
    prev = 0
    cur = 0
    for x in range(0, len(columnDataList)):
      prev = cur
      cur = len(columnDataList[x])
      if prev == 0:
        continue
      if prev != cur:
        print("value mismatch detected between column " + str(x) + " with length " + str(prev) + " and column " + str(x + 1) + " length " + str(cur))
        exit
    
    #process the data
    columnEncoder = list[list]()
    x = 0
    for code in columnCodes:
      if code == 0:
        sublist : list[str] = columnDataList[x]
        numlist = list[float]()
        missingValueIndices = list[int]()
        for index in range(0, len(sublist)):
          try:
            numlist.append(float(sublist[index]))
          except:
            if sublist[index] == missingAttribFlag:
              missingValueIndices.append(index)
        sortedlist = sorted(numlist)
        length = len(sortedlist)
        firstquartile = int(length * 0.25)
        median = int(length * 0.5)
        thirdquartile = int(length * 0.75)
        quartilelist = list([sortedlist[firstquartile], sortedlist[median], sortedlist[thirdquartile], sortedlist[-1]])
        columnEncoder.append(quartilelist)
        for index in missingValueIndices:
          numlist.insert(index, random.choice(quartilelist))
        columnDataList[x] = numlist
        x += 1
      elif code == 1 or code == 2:
        sublist = columnDataList[x]
        strset = set(sublist)
        strlist = list(strset)
        strlist = sorted(strlist)
        columnEncoder.append(strlist)
        x += 1
      elif code == -1:
        pass
    
    self.vectorList = [[] for _ in range(len(columnDataList[0]))]
    #use the columnEncoder to turn the data in the columDataList lists into feature vectors
    x = 0
    for code in columnCodes:
      if code == 0: #continuous values will get binned based on quartile
        subvectorLength = 4
        li = pd.qcut(columnDataList[x], q=subvectorLength, labels=False, duplicates='drop')
        if x == 0 and demo:
          democolumn = []
        for y in range(len(li)):
          subvector = [0] * subvectorLength
          subvector[li[y]] = 1
          self.vectorList[y].extend(subvector)
          if x == 0 and demo:
            democolumn.append(subvector)
          pass
        self.subvectorLengths.append(subvectorLength)
        x += 1
        pass
      if code == 1: #categorical values will get binned based on the string value
        subvectorLength = len(columnEncoder[x])
        for y in range(0, len(columnDataList[x])):
          subvector = [0] * subvectorLength
          subvector[columnEncoder[x].index(columnDataList[x][y])] = 1
          self.vectorList[y].extend(subvector)
          pass
        self.subvectorLengths.append(subvectorLength)
        x += 1
        pass
      if code == 2:
        self.numberOfClasses = len(columnEncoder[x])
        #we're going to skip the class column and save it for when everything else is encoded
        x += 1
        pass
      if code == -1:
        pass
    
    #code for printing the demo
    if demo:
      for i in range(30):
        print(f'{str(columnDataList[0][int(i * len(democolumn) / 30)]):2} -> {democolumn[int(i * len(democolumn) / 30)]}')
    
    #this is where we assign values to the class column
    x = classColumnNum - (columnCodes.count(-1) + 1)
    for y in range(0, len(columnDataList[0])):
      self.vectorList[y].append(columnEncoder[x].index(columnDataList[x][y]))
      pass
    self.subvectorLengths.append(1)
    self.classNames = columnEncoder[x]

    self.numberOfExamples = len(self.vectorList)
    self.vectorLength = len(self.vectorList[0])
    pass
  
  def shuffleVectors(self):
    random.shuffle(self.vectorList)

def shuffleElements(data : ProcessedData, factor : float) -> ProcessedData:
  #initialize the new data and copy over everything from the data
  newData = ProcessedData("__DEFAULT__")
  newData.subvectorLengths = copy.copy(data.subvectorLengths)
  newData.numberOfExamples = data.numberOfExamples
  newData.vectorList = copy.deepcopy(data.vectorList)
  newData.numberOfClasses = data.numberOfClasses
  newData.vectorLength = data.vectorLength
  newData.classNames = copy.copy(data.classNames)
  #apply the shuffling algorithm
  numberofattributes = len(data.subvectorLengths) - 1
  numberofshuffles = int(newData.numberOfExamples * numberofattributes * factor)
  for count in range(0, numberofshuffles):
    vectors = random.choices(newData.vectorList, k = 2)
    svlIndex = random.randint(0, numberofattributes - 1)
    attribLength = newData.subvectorLengths[svlIndex]
    attribIndex = 0
    for x in range(0, svlIndex):
      attribIndex += newData.subvectorLengths[x]
    temp = vectors[0][attribIndex : attribIndex + attribLength]
    for x in range(0, attribLength):
      try:
        vectors[0][attribIndex + x] = vectors[1][attribIndex + x]
        vectors[1][attribIndex + x] = temp[x]
      except:
        print(attribIndex + x)
        exit()
    pass
  return newData
  pass

def blankData() -> ProcessedData:
  return ProcessedData("__DEFAULT__")