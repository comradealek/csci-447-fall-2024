import preprocessor as prpr
import trainer as tr
import kfxv

selection = input("Please select a .data or .pdata file for training > ")
testData = prpr.ProcessedData(selection)
validationTable = kfxv.crossvalidation(testData)
for row in validationTable:
  print(row)