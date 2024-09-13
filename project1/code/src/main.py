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

def demo():
  data = prpr.blankData()
  print("\033c")
  input()
  print("\033c")
  data.processfile(open("./data/iris.data", "r"), classColumnNum=5, columnCodes=[0, 0, 0, 0, 2], missingAttribFlag="", demo=True)
  input()
  print("\033c")
  noisedata = prpr.shuffleElements(data, 0.30)
  classifier = tr.Classifier(data)
  classifier.printTable()
  input()
  classifier.printWeightedTable()
  input()
  print("\033c")
  cm = kfxv.democrossvalidation(data)
  cmwn = kfxv.democrossvalidation(noisedata)
  print("Confusion matrix for iris data:")
  kfxv.printTable(data, cm)
  print("Performance metrics for iris data:")
  ev.printMetrics(cm)
  print()
  print("Confusion matrix for iris data with noise:")
  kfxv.printTable(data, cmwn)
  print("Performance metrics for iris data with noise:")
  ev.printMetrics(cmwn)
  input()
  print("\033c")

def printBoxPlot(labels, data, noisedata, title=None):
  total = data + noisedata
  plt.boxplot(total, positions=[1, 2, 3, 4, 5, 7, 8 ,9 ,10 ,11], tick_labels=(labels+labels))
  plt.title(title)
  plt.show()

def fulltest():
  namelist = ["breast-cancer-wisconsin", "glass", "house-votes-84", "iris", "soybean-small"]
  
  #bunch of stuff for auto processing the data files
  classColumnList = [11, 11, 1, 5, 36]

  columnCodeList = [
    [-1] + [0 for _ in range(9)] + [2],
    [-1] + [0 for _ in range(5)] + [-1, -1, -1, 0] + [2],
    [2] + [1 for _ in range(16)],
    [0 for _ in range(4)] + [2],
    [1 for _ in range(10)] + [-1, 1] + [-1 for _ in range(7)] + [1 for _ in range(9)] + [-1 for _ in range(6)] + [1, 2]
  ]

  missingAttribList = [
    "?",
    "",
    "",
    "",
    ""
  ]
  
  precisionlist = [[] for _ in range(len(namelist))]
  recalllist =    [[] for _ in range(len(namelist))]
  zeroonelist =   [[] for _ in range(len(namelist))]
  fmeasurelist =  [[] for _ in range(len(namelist))]
  noiseprecisionlist = [[] for _ in range(len(namelist))]
  noiserecalllist =    [[] for _ in range(len(namelist))]
  noisezeroonelist =   [[] for _ in range(len(namelist))]
  noisefmeasurelist =  [[] for _ in range(len(namelist))]
  for i in range(len(namelist)):
    data = prpr.blankData()
    
    #process the data in the file
    try:
      data.loadpdata(open("./data/" + namelist[i] + ".pdata", "r"))
    except:
      data.processfile(
        open("./data/" + namelist[i] + ".data", "r"), 
        classColumnNum=classColumnList[i], 
        columnCodes=columnCodeList[i], 
        missingAttribFlag=missingAttribList[i]
      )
      data.writetofile(namelist[i] + ".pdata")
    
    #introduce noise
    noisedata = prpr.shuffleElements(data, 0.10)
    validationtable, tableList = kfxv.crossvalidation(data)
    noisevalidationtable, noisetableList = kfxv.crossvalidation(noisedata)

    for j in range(len(tableList)):
      precisionlist[i].append(ev.macroPrecision(tableList[j]))
      noiseprecisionlist[i].append(ev.macroPrecision(noisetableList[j]))
      recalllist[i].append(ev.macroRecall(tableList[j]))
      noiserecalllist[i].append(ev.macroRecall(noisetableList[j]))
      zeroonelist[i].append(ev.zeroOneLoss(tableList[j]))
      noisezeroonelist[i].append(ev.zeroOneLoss(noisetableList[j]))
      fmeasurelist[i].append(ev.macroFmeasure(tableList[j]))
      noisefmeasurelist[i].append(ev.macroFmeasure(noisetableList[j]))

    print("Results for " + namelist[i] + ":")
    kfxv.printTable(data, validationtable)
    print(f'{'precision:':10} {np.mean(precisionlist[i]):.4f}')
    print(f'{'recall:':10} {np.mean(recalllist[i]):.4f}')
    print(f'{'zero one:':10} {np.mean(zeroonelist[i]):.4f}')
    print(f'{'fmeasure:':10} {np.mean(fmeasurelist[i]):.4f}')
    print()
    print("Results for noise modified " + namelist[i] + ":")
    kfxv.printTable(noisedata, noisevalidationtable)
    print(f'{'precision:':10} {np.mean(noiseprecisionlist[i]):.4f}')
    print(f'{'recall:':10} {np.mean(noiserecalllist[i]):.4f}')
    print(f'{'zero one:':10} {np.mean(noisezeroonelist[i]):.4f}')
    print(f'{'fmeasure:':10} {np.mean(noisefmeasurelist[i]):.4f}')
    print()
  
  #Create the bar chart
  printBoxPlot(namelist, zeroonelist, noisezeroonelist, title="0-1 Loss")
  printBoxPlot(namelist, precisionlist, noiseprecisionlist, title="Precision")
  printBoxPlot(namelist, recalllist, noiserecalllist, title="Recall")
  printBoxPlot(namelist, fmeasurelist, noisefmeasurelist, title="F Measure")
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
  elif selection == "demo":
    demo()
    exit()
  else:
    test(selection)