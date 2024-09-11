import preprocessor as prpr
import trainer as tr
import kfxv
import evaluator as ev
import matplotlib.pyplot as plt
import numpy as np


def test(selection):
  testData = prpr.ProcessedData(selection)
  if testData.errorcode != 0:
    return
  modifiedData = prpr.shuffleElements(testData, 0.10)

  print("\nResults for unmodified test data")
  validationTable = kfxv.crossvalidation(testData)
  kfxv.printTable(testData, validationTable)
  ev.printMetrics(validationTable)

  print("\nResults for modified test data")
  validationTable = kfxv.crossvalidation(modifiedData)
  kfxv.printTable(testData, validationTable)
  ev.printMetrics(validationTable)

def printBarChart(labels, data, noisedata, title=None):
  x = np.arange(len(labels))
  width = 0.25
  plt.bar(x - (width/2), data, width)
  plt.bar(x + (width / 2), noisedata, width)
  plt.xticks(x, labels)
  plt.title(title)
  plt.legend(["clean", "noisy"])
  plt.xlabel("Data Sets")
  plt.ylim([0, 1])
  plt.show()

def fulltest():
  namelist = ["breast-cancer-wisconsin", "glass", "house-votes-84", "iris", "soybean-small"]
  precisionlist = [0.0 for _ in range(len(namelist))]
  recalllist =    [0.0 for _ in range(len(namelist))]
  zeroonelist =   [0.0 for _ in range(len(namelist))]
  fmeasurelist =  [0.0 for _ in range(len(namelist))]
  noiseprecisionlist = [0.0 for _ in range(len(namelist))]
  noiserecalllist =    [0.0 for _ in range(len(namelist))]
  noisezeroonelist =   [0.0 for _ in range(len(namelist))]
  noisefmeasurelist =  [0.0 for _ in range(len(namelist))]
  for i in range(len(namelist)):
    data = prpr.blankData()
    data.loadpdata(open("./data/" + namelist[i] + ".pdata", "r"))
    noisedata = prpr.shuffleElements(data, 0.10)
    table = kfxv.crossvalidation(data)
    noisetable = kfxv.crossvalidation(noisedata)

    """ print("Results for " + namelist[i] + ":")
    kfxv.printTable(data, table)
    ev.printMetrics(table)
    print() """

    precisionlist[i] = ev.macroPrecision(table)
    noiseprecisionlist[i] = ev.macroPrecision(noisetable)
    recalllist[i] = ev.macroRecall(table)
    noiserecalllist[i] = ev.macroRecall(noisetable)
    zeroonelist[i] = ev.zeroOneLoss(table)
    noisezeroonelist[i] = ev.zeroOneLoss(noisetable)
    fmeasurelist[i] = ev.macroFmeasure(table)
    noisefmeasurelist[i] = ev.macroFmeasure(noisetable)
  
  #Create the bar chart
  printBarChart(namelist, zeroonelist, noisezeroonelist, title="0-1 Loss")
  printBarChart(namelist, precisionlist, noiseprecisionlist, title="Precision")
  printBarChart(namelist, recalllist, noiserecalllist, title="Recall")
  printBarChart(namelist, fmeasurelist, noisefmeasurelist, "F Measure")
  return

code = 0
while code == 0:
  selection = input("Please select a .data or .pdata file for training > ")
  if selection == "":
    pass
  elif selection == "exit":
    exit()
  elif selection == "fulltest":
    fulltest()
  else:
    test(selection)