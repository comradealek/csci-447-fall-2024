import io

class ProcessedData:
  numberOfExamples: int = 0
  vectorLength: int = 0
  
  def __init__(self, filename : str) -> None:
    try:
      data_file = open("./data/" + filename + ".pdata", "r")
    except:
      try:
        data_file = open("./data/" + filename + ".data", "r")
        self.processfile(data_file)
      except:
        print("file does not exist")
        exit
  
  def processfile(self, file: io.TextIOWrapper):
    line = file.readline()
    elems = line.split(", ")
    elementCount = elems.__len__()
    print(elems)
    
    #gets the column number for the class signifier
    columnNum: int
    code = 0
    while code == 0:
      inputNum = input("Which column corresponds to the class? > ")
      if inputNum.isdigit():
        columnNum = int(inputNum)
        if columnNum > self.numberOfExamples:
          print("Out of range")
        else:
          code = 1
      else:
        print("Not a valid number")
    
    #flags each column as categorical, quantitative, or as the class signifier
    columnCodes = []
    x = 0
    while x < elementCount:
      x += 1
      if x == columnNum:
        columnCodes.append(3)
      else:
        inputValue = input("Is this data (cat)egorical or (quant)itative > ")
        code = 0
        while code == 0:
          if inputValue == "cat":
            columnCodes.append(1)
            code = 1
            pass
          elif inputValue == "quant":
            columnCodes.append(2)
            code = 1
            pass
          else:
            inputValue = input("Invalid entry. Please try again > ")
    #checks the validity of the flagging process
    if columnCodes.__len__() != elementCount:
      print("Error in processing file. Expected number of columns is " + elementCount + " and number flagged is " + columnCodes.__len__())
      exit
    
    #for each categorical and class column, we initiailize a set, and for each quantitative column, we initialize a list
    columnDataList = []
    x = 0
    while x < elementCount:
      if columnCodes(x) == 1:
        columnDataList.append()
      elif columnCodes(x) == 2 or columnCodes(x) == 3:
        pass
      else:
        pass
      x += 1

#process data in the file
#shuffle data
#print it to a file in ./data/ with extension .pdata