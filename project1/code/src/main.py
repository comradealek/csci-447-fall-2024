import preprocessor as prpr
import trainer as tr
import kfxv
import evaluator as ev

selection = input("Please select a .data or .pdata file for training > ")
testData = prpr.ProcessedData(selection)
modifiedData = prpr.shuffleElements(testData, 0.30)


print("\nResults for unmodified test data")
validationTable = kfxv.crossvalidation(testData)
for row in validationTable:
  print(row)

print(ev.macroPrecision(validationTable))
print(ev.macroRecall(validationTable))
print(ev.zeroOneLoss(validationTable))

print("\nResults for modified test data")
validationTable = kfxv.crossvalidation(modifiedData)
for row in validationTable:
  print(row)

print(ev.macroPrecision(validationTable))
print(ev.macroRecall(validationTable))
print(ev.zeroOneLoss(validationTable))