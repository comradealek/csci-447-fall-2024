import preprocessor as prpr
import kfxv
import trainer as tr

data = prpr.blankData()
data.loadpdata(open("./glass" + ".pdata", "r"))
classifier = tr.Classifier(data)
classifier.printDebugInfo()
print()
classifier.printTable()
print()
classifier.printWeightedTable()
