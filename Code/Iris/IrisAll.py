from numpy.random import seed
seed(12345)

import pandas as pd
import numpy as np
import os
import math
from sklearn.tree import export_graphviz
from PIL import Image
import datetime as dt
from sklearn.utils import shuffle
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
from keras.layers import Dropout # not needed here
from keras.utils import to_categorical


headers = [1,2,3,4,"Species"]

dataset = shuffle(pd.read_csv("iris.csv",delimiter=",",header = None,names = headers))
dataset["Species"] = dataset["Species"].astype("category")
dataset["SpeciesNew"] = dataset["Species"].cat.codes

X = dataset.iloc[:, 0:-2].values 
y = dataset.iloc[:, -1].values

Label = pd.value_counts(y).to_frame().reset_index()
print(Label)

cv = KFold(n_splits = 10, shuffle = True,random_state = 12345)




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
    df_MCC = pd.DataFrame(df_MCC,columns = ["MCC"] ) 
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
                feature_names=pd.DataFrame(X).columns,
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
        #model.add(Dropout(0.3))
        #model.add(Dense(units = Node2, kernel_initializer = 'uniform', activation = 'relu'))
        #model.add(Dropout(0.1))
        model.add(Dense(units = len(Label), kernel_initializer = 'uniform', activation = 'softmax'))
        model.compile(optimizer = "Adam", loss = 'categorical_crossentropy', metrics = ['accuracy'])
        model.fit(X[train_index],to_categorical(y[train_index]),epochs=200, batch_size= 4, verbose = 0)
        Y_pred = model.predict(X[test_index])
        y_pred = np.argmax(Y_pred, axis=1) 
        print(y_pred)
        print(y[test_index])
        confmat = confusion_matrix(y[test_index],y_pred)
        f1micro = f1_score(y[test_index], y_pred,average="micro")
        f1macro = f1_score(y[test_index], y_pred,average="macro")
        accuracy = accuracy_score(y[test_index], y_pred)
        MCC = matthews_corrcoef(y[test_index], y_pred)
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
clfSVMKernel = svm.SVC(kernel='rbf', C= 100, gamma = 0.01 )
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
          axis=1).to_csv('IrisConfmat.csv')
pd.concat([df1_f1_micro, df2_f1_micro, df3_f1_micro, df4_f1_micro, df5_f1_micro, df6_f1_micro, df7_f1_micro],
          axis=1).to_csv('IrisF1Micro.csv')
pd.concat([df1_f1_macro, df2_f1_macro, df3_f1_macro, df4_f1_macro, df5_f1_macro, df6_f1_macro, df7_f1_macro],
          axis=1).to_csv('IrisF1Macro.csv')
pd.concat([df1_accuracy, df2_accuracy, df3_accuracy, df4_accuracy, df5_accuracy, df6_accuracy, df7_accuracy],
          axis=1).to_csv('IrisAccuracy.csv')
pd.concat([df1_MCC, df2_MCC, df3_MCC, df4_MCC, df5_MCC, df6_MCC, df7_MCC],
          axis=1).to_csv('IrisMCC.csv')

# If you want to have an email sent at the end
# only works for hotmail to hotmail

#import email
#import smtplib

#msg = email.message_from_string('')
#msg['From'] = "****" # Sender
#msg['To'] = "*****" # recipent
#msg['Subject'] = "Done !"
#password = "*****"
#s = smtplib.SMTP("smtp.live.com",587)
#s.ehlo() 
#s.starttls()
#s.ehlo()
#s.login('****',password)

#s.sendmail("*******", "*******", msg.as_string())

#s.quit()