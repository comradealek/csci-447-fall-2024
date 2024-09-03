import preprocessor as prpr
import trainer as tr
import kfxv
import evaluator as ev

code = 0
while code == 0:
  selection = input("Please select a .data or .pdata file for training > ")
  if selection == "":
    pass
  if selection == "exit":
    code = 1
    pass
  else:
    testData = prpr.ProcessedData(selection)
    if testData.errorcode != 0:
      continue
    modifiedData = prpr.shuffleElements(testData, 0.10)

    print("\nResults for unmodified test data")
    validationTable = kfxv.crossvalidation(testData)
    for x in range(len(validationTable)):
      print(str(testData.classNames[x]) + " " + str(validationTable[x]))

    print(ev.macroPrecision(validationTable))
    print(ev.macroRecall(validationTable))
    print(ev.zeroOneLoss(validationTable))
    print(ev.macroFmeasure(validationTable))

    print("\nResults for modified test data")
    validationTable = kfxv.crossvalidation(modifiedData)
    for row in validationTable:
      print(row)

    print(ev.zeroOneLoss(validationTable))
    print(ev.macroPrecision(validationTable))
    print(ev.macroRecall(validationTable))
    print(ev.macroFmeasure(validationTable))