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
    
    x = 0
    while x < elementCount:
      
      x += 1

#ask for a file name without extension
#check if there is a .pdata file present for the dataset
#ask for number of columns
#ask which column is the class
#for each other column ask for quantitative or qualitative
#process data in the file
#shuffle data
#print it to a file in ./data/ with extension .pdata