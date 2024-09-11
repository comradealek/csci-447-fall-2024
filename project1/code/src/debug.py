import preprocessor as prpr
import kfxv
import trainer as tr

data = prpr.blankData()
data.loadpdata(open("./data/" + "glass" + ".pdata", "r"))
table = kfxv.crossvalidation(data)
print(table)