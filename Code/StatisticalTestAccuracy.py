# To perform test from R in python

from rpy2.robjects.packages import importr
from rpy2.robjects.vectors import StrVector
import rpy2.robjects as ro
import rpy2.robjects.numpy2ri
import rpy2.robjects.pandas2ri

rpy2.robjects.numpy2ri.activate()
rpy2.robjects.pandas2ri.activate()

utils = importr("utils")
PMCMRplus = importr("PMCMRplus")
NSM3 = importr("NSM3")


#utils.chooseCRANmirror()
#packnames=("PCMRplus")
#utils.install_packages(StrVector(packnames))
## might not work -> do
#utils.install_packages("PMCMRplus")


import os
import pandas as pd
import numpy as np
from scipy.stats import friedmanchisquare



Algo = ["LR","C-SVMRBF","C-SVM","RF","ANN","KNN","Naive"]
Datasets = ["Adult","Alzheimer","Bank","Breast","Cell","Contra","Aus","Ger","Iris","Sonar","Wine","NBA"]

AccuracyAlzheimer = pd.read_csv("Alzheimer\AlzheimerAccuracy.csv", delimiter =",", header = 0 ,names = Algo)
AccuracyAdult = pd.read_csv("Adult\AdultAccuracy.csv", delimiter =",", header = 0 ,names = Algo)
AccuracyBank = pd.read_csv("Bank\BankAccuracy.csv", delimiter =",", header = 0 ,names = Algo)
AccuracyBreast = pd.read_csv("Breast\BreastAccuracy.csv", delimiter =",", header = 0 ,names = Algo)
AccuracyCell = pd.read_csv("Cell\CellAccuracy.csv", delimiter =",", header = 0 ,names = Algo)
AccuracyContra = pd.read_csv("Contraceptive\ContraAccuracy.csv", delimiter =",", header = 0 ,names = Algo)
AccuracyAus = pd.read_csv("CreditScoreAus\AusAccuracy.csv", delimiter =",", header = 0 ,names = Algo)
AccuracyGer = pd.read_csv("CreditScoreGer\GerAccuracy.csv", delimiter =",", header = 0 ,names = Algo)
AccuracyIris = pd.read_csv("Iris\IrisAccuracy.csv", delimiter =",", header = 0 ,names = Algo)
AccuracySonar = pd.read_csv("Sonar\SonarAccuracy.csv", delimiter =",", header = 0 ,names = Algo)
AccuracyWine = pd.read_csv("Wine\WineAccuracy.csv", delimiter =",", header = 0 ,names = Algo)

os.getcwd()
os.chdir("Basket")

AccuracyNBA = pd.read_csv("basketAccuracy.csv", delimiter =",", header = 0 ,names = Algo)



Adult = AccuracyAdult.mean()
Alzheimer = AccuracyAlzheimer.mean()
Bank = AccuracyBank.mean()
Breast = AccuracyBreast.mean()
Cell = AccuracyCell.mean()
Contra = AccuracyContra.mean()
Aus = AccuracyAus.mean()
Ger = AccuracyGer.mean()
Iris = AccuracyIris.mean()
Sonar = AccuracySonar.mean()
Wine = AccuracyWine.mean()
NBA = AccuracyNBA.mean()


TableAccuracy = pd.concat([pd.DataFrame(Adult),pd.DataFrame(Alzheimer),pd.DataFrame(Bank),
                           pd.DataFrame(Breast),pd.DataFrame(Cell),
                           pd.DataFrame(Contra),pd.DataFrame(Aus),
                           pd.DataFrame(Ger),pd.DataFrame(Iris),
                           pd.DataFrame(Sonar),pd.DataFrame(Wine),
                           pd.DataFrame(NBA)],axis = 1)

TableAccuracy.columns = Datasets
TableAccuracy = TableAccuracy.transpose()
TableAccuracy = np.round(TableAccuracy, decimals =10)
TableAccuracy

Pairwise1 =np.stack((np.array(TableAccuracy["LR"]),np.array(TableAccuracy["C-SVM"])))

PairwiseTotal = np.stack((np.array(TableAccuracy["LR"]),
                     np.array(TableAccuracy["C-SVMRBF"]),np.array(TableAccuracy["C-SVM"]),
                     np.array(TableAccuracy["RF"]),
                     np.array(TableAccuracy["ANN"]), 
                     np.array(TableAccuracy["KNN"]),
                     np.array(TableAccuracy["Naive"])))


nr,nc = PairwiseTotal.shape
BR = ro.r.matrix(PairwiseTotal, nrow=nr, ncol=nc)
BR = BR.transpose()
print(BR)

friedmanchisquare(TableAccuracy["LR"], TableAccuracy["C-SVMRBF"],TableAccuracy["C-SVM"] ,
                  TableAccuracy["RF"],TableAccuracy["ANN"], TableAccuracy["KNN"],
                  TableAccuracy["Naive"])

print(PMCMRplus.friedmanTest(BR))
print(PMCMRplus.friedmanTest(BR).rx2("statistic"))
print(PMCMRplus.friedmanTest(BR).rx2("p.value"))
print(PMCMRplus.friedmanTest(BR,dist="FDist"))
print(PMCMRplus.friedmanTest(BR,dist="FDist").rx2("statistic"))
print(PMCMRplus.friedmanTest(BR,dist="FDist").rx2("p.value"))

BRNum = ro.r.matrix(ro.r.c(BR),nrow = 12, ncol = 7,dimnames = ro.r.list(ro.r.c("Adult","Alzheimer","Bank","Breast","Cell","Contra","Aus","Ger","Iris","Sonar","Wine","NBA"),ro.r.c("LR","C-SVMRBF","C-SVM","RF","ANN","KNN","Naive")))


print(PMCMRplus.frdAllPairsNemenyiTest(BRNum))
print(PMCMRplus.frdAllPairsConoverTest(BRNum))

print(NSM3.pWNMT(BRNum,method="Asymptotic"))

#wilcoxon signed rank test pairwise on 1 dataset

stats = importr("stats")

def Wilcoxon(Algo1):
    vector1 = ro.r.matrix(np.array(TableAccuracy[Algo1]),ncol=1)
    for i in Algo :
        vector2 = ro.r.matrix(np.array(TableAccuracy[i]),ncol=1)
        print(i,stats.wilcox_test(vector1,vector2,paired = True).rx2("p.value"))


Wilcoxon("LR")





