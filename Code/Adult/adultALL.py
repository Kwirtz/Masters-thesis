from numpy.random import seed
seed(12345)

import pandas as pd
import numpy as np
import os
import math
from sklearn.tree import export_graphviz
from sklearn.utils import shuffle
from PIL import Image
import datetime as dt


# Models

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

# Evaluation

from sklearn.model_selection import KFold
from sklearn.metrics.classification import cohen_kappa_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import matthews_corrcoef
# Neural network

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout 


headers = ["Age", "workclass" , "fnlwgt", "education", "education-num", "marital-status","occupation","relationship","race","sex","capital-gain","capital-loss","hours-per-week","native-country","Income"]
categorical = ["workclass","education","marital-status","occupation","relationship","race","sex","native-country","Income"]

dataset = shuffle(pd.read_csv("adult.data" , sep = ",", header = None, skiprows= 1, names = headers))
dataset.replace([" ?"],np.nan, inplace = True)
dataset = dataset.dropna()

for i in categorical:
    dataset[i]=dataset[i].astype("category")



for i in categorical:
    dataset = pd.concat([dataset,pd.get_dummies(dataset[i],prefix = i,drop_first= True)],axis = 1)
    dataset.drop([i],axis = 1, inplace = True)

y = dataset.iloc[:].values[:,-1]
X = dataset.iloc[:,:].values [:, 0:-2]

Label = pd.value_counts(y).to_frame().reset_index()
print(Label)

cv = KFold(n_splits = 10, shuffle = True,random_state = 12345)


for train_index, test_index in cv.split(X):
    print(pd.value_counts(y[test_index]).to_frame().reset_index())



def Training_Logit_SVM_RF(classifier, RF = False):
    
    start_time = dt.datetime.now()
    print('Start learning at {}'.format(str(start_time)))
    i = 0
    
    df_f1_micro = []
    df_f1_macro = []
    df_accuracy = []
    df_MCC = []

    for train_index, test_index in cv.split(X):
        i += 1
        classifier.fit(X[train_index], y[train_index])
        ypred = classifier.predict(X[test_index])
        kappa_score = cohen_kappa_score(y[test_index], ypred)
        confmat = confusion_matrix(y[test_index], ypred)
        f1micro = f1_score(y[test_index], ypred,average="micro")
        f1macro = f1_score(y[test_index], ypred,average="macro")
        accuracy = accuracy_score(y[test_index], ypred)
        MCC = matthews_corrcoef(y[test_index], ypred)
        print("\nKappa score\n" ,kappa_score,"\n")
        print("\n confmat\n",confmat,"\n")
        print("\n f1 micro\n", f1micro,"\n")
        print("\n f1 macro\n", f1macro,"\n")
        print("\n accuracy\n", accuracy,"\n")
        
        if i == 1:
            
            df_confmat = pd.DataFrame(confmat)
            
            
            
        elif i > 1:
            
            temp = pd.DataFrame(confmat)
            df_confmat = df_confmat.append(temp)
        
        df_f1_micro.append(f1micro)
        df_f1_macro.append(f1macro)
        df_accuracy.append(accuracy)
        df_MCC.append(MCC)
    
    df_f1_micro = pd.DataFrame(df_f1_micro,columns=["f1micro"])
    df_f1_macro = pd.DataFrame(df_f1_macro,columns=["f1macro"])
    df_accuracy = pd.DataFrame(df_accuracy,columns=["accuracy"])
    df_MCC = pd.DataFrame(df_MCC, columns =["MCC"])
    end_time = dt.datetime.now() 
    
    print('Stop learning {}'.format(str(end_time)))
    elapsed_time= end_time - start_time
    print('Elapsed learning {}'.format(str(elapsed_time)))
    
    if RF == True:
        MakeTree(0,classifier)
    else:
        pass
    
    return df_confmat, df_f1_micro, df_f1_macro, df_accuracy, df_MCC



################################### Random Forest
###################################
###################################

# will be usefull to plot a tree

def MakeTree(number,classifier):
	
	export_graphviz(classifier.estimators_[number],
                feature_names=dataset.columns[0:-2],
                filled=True,
                rounded=True,
                out_file = "Tree.dot")


	os.system('dot -Tpng Tree.dot -o Tree.png') # need to dl graphviz and add to environment path
	im = Image.open("Tree.png")
	im.rotate(360).show()




################################### ANN
###################################
###################################

Node1 = math.floor(math.sqrt(((len(Label)+2)*len(y)))) + math.floor(2*math.sqrt((len(y))/(len(Label)+2)))
Node2 = math.floor(len(Label)*math.sqrt(len(y)/(len(Label)+2)))
_,inp = X.shape

def NeuralNetwork():
    i = 0
    
    df_f1_micro = []
    df_f1_macro = []
    df_accuracy = []
    df_MCC = []
    for train_index,test_index in cv.split(X):
        i +=1
        model = Sequential()
        model.add(Dense(units = Node1, kernel_initializer = 'uniform', activation = 'relu', input_dim = inp))
        model.add(Dropout(0.5))
        model.add(Dense(units = Node2, kernel_initializer = 'uniform', activation = 'relu'))
        model.add(Dropout(0.5))
        model.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
        model.compile(optimizer = "Adam", loss = 'binary_crossentropy', metrics = ['accuracy'])
        model.fit(X[train_index],y[train_index],epochs = 64, batch_size = 128, verbose = 1000)
        Y_pred = model.predict(X[test_index])
        y_pred = Y_pred >= 0.5
        
        confmat = confusion_matrix(y[test_index],y_pred)
        f1micro = f1_score(y[test_index], y_pred,average="micro")
        f1macro = f1_score(y[test_index], y_pred,average="macro")
        accuracy = accuracy_score(y[test_index], y_pred)
        MCC = matthews_corrcoef(y[test_index], y_pred)
        if accuracy < 0.5:
            print(accuracy)
            print(pd.value_counts(y[test_index]).to_frame().reset_index())
        else:
            pass
        if i == 1:
            
            df_confmat = pd.DataFrame(confmat)
            
            
        elif i > 1:
            
            temp = pd.DataFrame(confmat)
            df_confmat = df_confmat.append(temp)
        
        
        df_f1_micro.append(f1micro)
        df_f1_macro.append(f1macro)
        df_accuracy.append(accuracy)
        df_MCC.append(MCC)
    df_f1_micro = pd.DataFrame(df_f1_micro,columns=["f1micro"])
    df_f1_macro = pd.DataFrame(df_f1_macro,columns=["f1macro"])
    df_accuracy = pd.DataFrame(df_accuracy,columns=["accuracy"])
    df_MCC = pd.DataFrame(df_MCC, columns = ["MCC"])
    return df_confmat, df_f1_micro, df_f1_macro, df_accuracy, df_MCC





################################### Present the result
###################################
###################################


clfLR = LogisticRegression(multi_class = "multinomial", solver ="lbfgs")
clfRF = RandomForestClassifier(n_estimators = 128, criterion ="gini", oob_score = False, random_state = 0, max_features = "sqrt")
clfSVM = svm.LinearSVC(C = 1)
clfSVMKernel = svm.SVC(kernel='rbf', C= 10, gamma = 0.0000001 )
clfKNN = KNeighborsClassifier(n_neighbors=math.floor(math.sqrt(len(y))))
clfGNB = GaussianNB()

print("Logistic Regression Results")
print("")
df1_confmat, df1_f1_micro, df1_f1_macro, df1_accuracy, df1_MCC = Training_Logit_SVM_RF(clfLR)
print("")

print("SVM RBF Results")
print("")
df2_confmat, df2_f1_micro, df2_f1_macro, df2_accuracy, df2_MCC = Training_Logit_SVM_RF(clfSVMKernel)
print("")



print("SVM Linear")
print("")
df3_confmat, df3_f1_micro, df3_f1_macro, df3_accuracy, df3_MCC =Training_Logit_SVM_RF(clfSVM)
print("")

print("Random forest Results")
print("")
df4_confmat, df4_f1_micro, df4_f1_macro, df4_accuracy, df4_MCC = Training_Logit_SVM_RF(clfRF,True)
print("")


print("Neural Network Results")
print("")
df5_confmat, df5_f1_micro, df5_f1_macro, df5_accuracy, df5_MCC = NeuralNetwork()
print("")

print("KNN")
print("")
df6_confmat, df6_f1_micro, df6_f1_macro, df6_accuracy, df6_MCC = Training_Logit_SVM_RF(clfKNN)
print("")

print("Naive bayes")
print("")
df7_confmat, df7_f1_micro, df7_f1_macro, df7_accuracy, df7_MCC = Training_Logit_SVM_RF(clfGNB)
print("")



pd.concat([df1_confmat, df2_confmat, df3_confmat, df4_confmat,df5_confmat,df6_confmat,df7_confmat],
          axis=1).to_csv('AdultConfmat.csv')
pd.concat([df1_f1_micro, df2_f1_micro, df3_f1_micro, df4_f1_micro, df5_f1_micro, df6_f1_micro, df7_f1_micro],
          axis=1).to_csv('AdultF1Micro.csv')
pd.concat([df1_f1_macro, df2_f1_macro, df3_f1_macro, df4_f1_macro, df5_f1_macro, df6_f1_macro, df7_f1_macro],
          axis=1).to_csv('AdultF1Macro.csv')
pd.concat([df1_accuracy, df2_accuracy, df3_accuracy, df4_accuracy, df5_accuracy, df6_accuracy, df7_accuracy],
          axis=1).to_csv('AdultAccuracy.csv')
pd.concat([df1_MCC, df2_MCC, df3_MCC, df4_MCC, df5_MCC, df6_MCC, df7_MCC],
          axis=1).to_csv('AdultMCC.csv')

