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

MCCAlzheimer = pd.read_csv("Alzheimer\AlzheimerMCC.csv", delimiter =",", header = 0 ,names = Algo)
MCCAdult = pd.read_csv("Adult\AdultMCC.csv", delimiter =",", header = 0 ,names = Algo)
MCCBank = pd.read_csv("Bank\BankMCC.csv", delimiter =",", header = 0 ,names = Algo)
MCCBreast = pd.read_csv("Breast\BreastMCC.csv", delimiter =",", header = 0 ,names = Algo)
MCCCell = pd.read_csv("Cell\CellMCC.csv", delimiter =",", header = 0 ,names = Algo)
MCCContra = pd.read_csv("Contraceptive\ContraMCC.csv", delimiter =",", header = 0 ,names = Algo)
MCCAus = pd.read_csv("CreditScoreAus\AusMCC.csv", delimiter =",", header = 0 ,names = Algo)
MCCGer = pd.read_csv("CreditScoreGer\GerMCC.csv", delimiter =",", header = 0 ,names = Algo)
MCCIris = pd.read_csv("Iris\IrisMCC.csv", delimiter =",", header = 0 ,names = Algo)
MCCSonar = pd.read_csv("Sonar\SonarMCC.csv", delimiter =",", header = 0 ,names = Algo)
MCCWine = pd.read_csv("Wine\WineMCC.csv", delimiter =",", header = 0 ,names = Algo)

os.getcwd()
os.chdir("Basket")

MCCNBA = pd.read_csv("basketMCC.csv", delimiter =",", header = 0 ,names = Algo)



Adult = MCCAdult.mean()
Alzheimer = MCCAlzheimer.mean()
Bank = MCCBank.mean()
Breast = MCCBreast.mean()
Cell = MCCCell.mean()
Contra = MCCContra.mean()
Aus = MCCAus.mean()
Ger = MCCGer.mean()
Iris = MCCIris.mean()
Sonar = MCCSonar.mean()
Wine = MCCWine.mean()
NBA = MCCNBA.mean()


TableMCC = pd.concat([pd.DataFrame(Adult),pd.DataFrame(Alzheimer),pd.DataFrame(Bank),
                           pd.DataFrame(Breast),pd.DataFrame(Cell),
                           pd.DataFrame(Contra),pd.DataFrame(Aus),
                           pd.DataFrame(Ger),pd.DataFrame(Iris),
                           pd.DataFrame(Sonar),pd.DataFrame(Wine),
                           pd.DataFrame(NBA)],axis = 1)

TableMCC.columns = Datasets
TableMCC = TableMCC.transpose()
TableMCC = np.round(TableMCC, decimals =3)
TableMCC

Pairwise1 =np.stack((np.array(TableMCC["LR"]),np.array(TableMCC["C-SVM"])))

PairwiseTotal = np.stack((np.array(TableMCC["LR"]),
                     np.array(TableMCC["C-SVMRBF"]),np.array(TableMCC["C-SVM"]),
                     np.array(TableMCC["RF"]),
                     np.array(TableMCC["ANN"]), 
                     np.array(TableMCC["KNN"]),
                     np.array(TableMCC["Naive"])))


nr,nc = PairwiseTotal.shape
BR = ro.r.matrix(PairwiseTotal, nrow=nr, ncol=nc)
BR = BR.transpose()
print(BR)

friedmanchisquare(TableMCC["LR"], TableMCC["C-SVMRBF"],TableMCC["C-SVM"] ,
                  TableMCC["RF"],TableMCC["ANN"], TableMCC["KNN"],
                  TableMCC["Naive"])

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
vector1 = ro.r.matrix(np.array(TableMCC["ANN"]),ncol=1)
vector2 = ro.r.matrix(np.array(TableMCC["LR"]),ncol=1)
print(stats.wilcox_test(vector1,vector2,paired = True))
