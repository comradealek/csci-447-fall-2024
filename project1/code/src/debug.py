import preprocessor as prpr
import kfxv
import trainer as tr

data = prpr.blankData()
data.loadpdata(open("./data/" + "testdata" + ".pdata", "r"))
classifier = tr.Classifier(data)
classifier.printTable()