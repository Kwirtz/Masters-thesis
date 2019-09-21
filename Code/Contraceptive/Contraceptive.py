# Importing the libraries
import os
import sys

sys.path.insert(1, os.path.join(sys.path[0], '..'))

import pandas as pd

from SVM import SVM
from ANN import ANN
from Others import Training

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils import shuffle
import math

# Importing the dataset

headers = ["wifeage", "wifeeduc" , "husbeduc", "Children", "wiferel", "wifework","husboccup","living","Media","Contra"]
categorical = [ "wifeeduc" , "husbeduc", "wiferel", "wifework","husboccup","living","Media"]
dataset = shuffle(pd.read_csv("cmc.csv",delimiter=",",header= None, names = headers))

for i in categorical:
    dataset[i]=dataset[i].astype("category")

y = dataset.iloc[:, -1].values

for i in categorical:
    dataset = pd.concat([dataset,pd.get_dummies(dataset[i],prefix = i,drop_first= True)],axis = 1)
    dataset.drop([i],axis = 1, inplace = True)


dataset.drop(["Contra"],axis = 1, inplace = True)
X = dataset.iloc[:, :].values 


Label = pd.value_counts(y).to_frame().reset_index()
print(Label)
# SVM Models

ModelLinear = SVM(X, y)
ModelLinear.Linearparam()
df1_confmat, df1_f1_micro, df1_f1_macro, df1_accuracy, df1_MCC = ModelLinear.training("SVMLinear", ModelLinear.C)


ModelRBF = SVM(X, y)
ModelRBF.RBFparam()
df2_confmat, df2_f1_micro, df2_f1_macro, df2_accuracy, df2_MCC = ModelRBF.training("RBF", ModelRBF.C, ModelRBF.gamma)

#ANN models

ModelANN = ANN(X, y, Label)
ModelANN.Tune()
df3_confmat, df3_f1_micro, df3_f1_macro, df3_accuracy, df3_MCC = ModelANN.Train(ModelANN.batch, ModelANN.dropout1, ModelANN.dropout2, ModelANN.epochs)

# Others

clfLR = LogisticRegression(multi_class = "multinomial", solver ="lbfgs")
clfRF = RandomForestClassifier(n_estimators = 128, criterion ="gini", oob_score = False, random_state = 0, max_features = "sqrt")
clfKNN = KNeighborsClassifier(n_neighbors=math.floor(math.sqrt(len(y))))
clfGNB = GaussianNB()

df4_confmat, df4_f1_micro, df4_f1_macro, df4_accuracy, df4_MCC = Training(X, y, clfLR)
df5_confmat, df5_f1_micro, df5_f1_macro, df5_accuracy, df5_MCC = Training(X, y, clfRF, True)
df6_confmat, df6_f1_micro, df6_f1_macro, df6_accuracy, df6_MCC = Training(X, y, clfKNN)
df7_confmat, df7_f1_micro, df7_f1_macro, df7_accuracy, df7_MCC = Training(X, y, clfGNB)

# Save the scores

pd.concat([df1_confmat, df2_confmat, df3_confmat, df4_confmat,df5_confmat,df6_confmat,df7_confmat],
          axis=1).to_csv('ContraceptiveConfmat.csv')
pd.concat([df1_f1_micro, df2_f1_micro, df3_f1_micro, df4_f1_micro, df5_f1_micro, df6_f1_micro, df7_f1_micro],
          axis=1).to_csv('ContraceptiveF1Micro.csv')
pd.concat([df1_f1_macro, df2_f1_macro, df3_f1_macro, df4_f1_macro, df5_f1_macro, df6_f1_macro, df7_f1_macro],
          axis=1).to_csv('ContraceptiveF1Macro.csv')
pd.concat([df1_accuracy, df2_accuracy, df3_accuracy, df4_accuracy, df5_accuracy, df6_accuracy, df7_accuracy],
          axis=1).to_csv('ContraceptiveAccuracy.csv')
pd.concat([df1_MCC, df2_MCC, df3_MCC, df4_MCC, df5_MCC, df6_MCC, df7_MCC],
          axis=1).to_csv('ContraceptiveMCC.csv')


