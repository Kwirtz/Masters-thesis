from sklearn import svm
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.metrics.classification import cohen_kappa_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import matthews_corrcoef
import datetime as dt
import pandas as pd
import json

class SVM():
    
    def __init__(self, X, y):
        self.X = X
        self.y = y
        
    def RBFparam(self):
        if len(self.X) > 30000:
            gamma_range = [0.01, 0.1, 1]
            C_range = [0.1,1,10,10]
        else:
            gamma_range = [0.0000001, 0.00001, 0.01, 0.1, 1]
            C_range = [0.1,1,10,100]
        parameters = { 'C' : C_range, "gamma" : gamma_range}
        cv = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=12345)
        grid = GridSearchCV(svm.SVC(kernel = "rbf"), param_grid=parameters, cv=cv, verbose = 1000)
        grid.fit(self.X, self.y)
        print("The best parameters are %s with a score of %0.2f"
              % (grid.best_params_, grid.best_score_))
        self.C = grid.best_params_["C"]
        self.gamma = grid.best_params_["gamma"]
        param = {"accuracy":grid.best_score_,"C":self.C,"gamma":self.gamma}
        with open("param.json","r") as f:
            data = json.load(f)
        data["classifier"]["RBFSVM"] = param
        with open("param.json","w") as f:
            json.dump(data,f)
    
    def Linearparam(self):
        C_range = [0.1,1,10,100]
        parameters = {"C" : C_range}
        cv = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=12345)
        grid = GridSearchCV(svm.LinearSVC(), param_grid=parameters, cv=cv, verbose = 1)
        grid.fit(self.X, self.y)
        print("The best parameters are %s with a score of %0.2f"
              % (grid.best_params_, grid.best_score_))
        self.C = grid.best_params_["C"]
        param = {"classifier":{"LinearSVM":{"accuracy":grid.best_score_, "C":self.C}}}
        with open("param.json","w") as f:
            json.dump(param,f)

    def training(self, Model, C=1, gamma=1):
        cv = KFold(n_splits = 10, shuffle = True,random_state = 12345)
        start_time = dt.datetime.now()
        print('Start learning at {}'.format(str(start_time)))
        i = 0
    
        df_f1_micro = []
        df_f1_macro = []
        df_accuracy = []
        df_MCC = []

        for train_index, test_index in cv.split(self.X):
            i += 1
            if Model == "SVMLinear":
                classifier = svm.LinearSVC(C = C)
                classifier.fit(self.X[train_index], self.y[train_index])
            else:
                classifier = svm.SVC(kernel='rbf', C= C, gamma = gamma )
                classifier.fit(self.X[train_index], self.y[train_index])
            ypred = classifier.predict(self.X[test_index])
            kappa_score = cohen_kappa_score(self.y[test_index], ypred)
            confmat = confusion_matrix(self.y[test_index], ypred)
            f1micro = f1_score(self.y[test_index], ypred,average="micro")
            f1macro = f1_score(self.y[test_index], ypred,average="macro")
            accuracy = accuracy_score(self.y[test_index], ypred)
            MCC = matthews_corrcoef(self.y[test_index], ypred)
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
    
        self.f1_micro = pd.DataFrame(df_f1_micro,columns=["f1micro"])
        self.f1_macro = pd.DataFrame(df_f1_macro,columns=["f1macro"])
        self.accuracy = pd.DataFrame(df_accuracy,columns=["accuracy"])
        self.MCC = pd.DataFrame(df_MCC,columns = ["MCC"] ) 
        self.confmat = df_confmat
        
        end_time = dt.datetime.now() 
        print('Stop learning {}'.format(str(end_time)))
        elapsed_time= end_time - start_time
        print('Elapsed learning {}'.format(str(elapsed_time)))
        
        return self.confmat, self.f1_micro, self.f1_macro, self.accuracy, self.MCC


        
            