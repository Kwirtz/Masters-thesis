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

F1MacroAlzheimer = pd.read_csv("Alzheimer\AlzheimerF1Macro.csv", delimiter =",", header = 0 ,names = Algo)
F1MacroAdult = pd.read_csv("Adult\AdultF1Macro.csv", delimiter =",", header = 0 ,names = Algo)
F1MacroBank = pd.read_csv("Bank\BankF1Macro.csv", delimiter =",", header = 0 ,names = Algo)
F1MacroBreast = pd.read_csv("Breast\BreastF1Macro.csv", delimiter =",", header = 0 ,names = Algo)
F1MacroCell = pd.read_csv("Cell\CellF1Macro.csv", delimiter =",", header = 0 ,names = Algo)
F1MacroContra = pd.read_csv("Contraceptive\ContraF1Macro.csv", delimiter =",", header = 0 ,names = Algo)
F1MacroAus = pd.read_csv("CreditScoreAus\AusF1Macro.csv", delimiter =",", header = 0 ,names = Algo)
F1MacroGer = pd.read_csv("CreditScoreGer\GerF1Macro.csv", delimiter =",", header = 0 ,names = Algo)
F1MacroIris = pd.read_csv("Iris\IrisF1Macro.csv", delimiter =",", header = 0 ,names = Algo)
F1MacroSonar = pd.read_csv("Sonar\SonarF1Macro.csv", delimiter =",", header = 0 ,names = Algo)
F1MacroWine = pd.read_csv("Wine\WineF1Macro.csv", delimiter =",", header = 0 ,names = Algo)

os.getcwd()
os.chdir("Basket")

F1MacroNBA = pd.read_csv("basketF1Macro.csv", delimiter =",", header = 0 ,names = Algo)



Adult = F1MacroAdult.mean()
Alzheimer = F1MacroAlzheimer.mean()
Bank = F1MacroBank.mean()
Breast = F1MacroBreast.mean()
Cell = F1MacroCell.mean()
Contra = F1MacroContra.mean()
Aus = F1MacroAus.mean()
Ger = F1MacroGer.mean()
Iris = F1MacroIris.mean()
Sonar = F1MacroSonar.mean()
Wine = F1MacroWine.mean()
NBA = F1MacroNBA.mean()


TableF1Macro = pd.concat([pd.DataFrame(Adult),pd.DataFrame(Alzheimer),pd.DataFrame(Bank),
                           pd.DataFrame(Breast),pd.DataFrame(Cell),
                           pd.DataFrame(Contra),pd.DataFrame(Aus),
                           pd.DataFrame(Ger),pd.DataFrame(Iris),
                           pd.DataFrame(Sonar),pd.DataFrame(Wine),
                           pd.DataFrame(NBA)],axis = 1)

TableF1Macro.columns = Datasets
TableF1Macro = TableF1Macro.transpose()
TableF1Macro = np.round(TableF1Macro, decimals = 3)
TableF1Macro

Pairwise1 =np.stack((np.array(TableF1Macro["LR"]),np.array(TableF1Macro["C-SVM"])))

PairwiseTotal = np.stack((np.array(TableF1Macro["LR"]),
                     np.array(TableF1Macro["C-SVMRBF"]),np.array(TableF1Macro["C-SVM"]),
                     np.array(TableF1Macro["RF"]),
                     np.array(TableF1Macro["ANN"]), 
                     np.array(TableF1Macro["KNN"]),
                     np.array(TableF1Macro["Naive"])))


nr,nc = PairwiseTotal.shape
BR = ro.r.matrix(PairwiseTotal, nrow=nr, ncol=nc)
BR = BR.transpose()
print(BR)

friedmanchisquare(TableF1Macro["LR"], TableF1Macro["C-SVMRBF"],TableF1Macro["C-SVM"] ,
                  TableF1Macro["RF"],TableF1Macro["ANN"], TableF1Macro["KNN"],
                  TableF1Macro["Naive"])

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
vector1 = ro.r.matrix(np.array(TableF1Macro["ANN"]),ncol=1)
vector2 = ro.r.matrix(np.array(TableF1Macro["C-SVM"]),ncol=1)
print(stats.wilcox_test(vector1,vector2,paired = True))