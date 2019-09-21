import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout 
from keras.utils.np_utils import to_categorical
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import matthews_corrcoef
import numpy as np
import pandas as pd
import json

class ANN():
    
    def __init__(self, X, y, categories):
        self.X = X
        self.y = y
        self.categories = categories
        self.Node1 = math.floor(math.sqrt(((len(self.categories)+2)*len(self.y)))) + math.floor(2*math.sqrt((len(y))/(len(self.categories)+2)))
        self.Node2 = math.floor(len(self.categories)*math.sqrt(len(self.y)/(len(self.categories)+2)))
        _,inp = self.X.shape
        self.inp = inp
        
    def Tune(self):
        
        def build_classifier( X = self.X, y = self.y, dropout_rate1=0.0, dropout_rate2=0.0):
        
            classifier = Sequential()
            classifier.add(Dense(units = self.Node1, kernel_initializer = "uniform", activation = "relu", input_dim = self.inp ))
            classifier.add(Dropout(dropout_rate1))
            classifier.add(Dense(units = self.Node2, kernel_initializer = "uniform", activation = 'relu'))
            classifier.add(Dropout(dropout_rate2))
            classifier.add(Dense(units = len(self.categories), kernel_initializer = "uniform", activation = 'softmax'))
            if self.categories.shape[0] > 2:
                classifier.compile(optimizer = "sgd", loss = 'categorical_crossentropy', metrics = ['accuracy'])
            else:
                classifier.compile(optimizer = "sgd", loss = 'binary_crossentropy', metrics = ['accuracy'])
            return classifier
        
        cv = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=12345)
        classifier = KerasClassifier(build_fn = build_classifier)

        if len(self.X) > 30000:
            parameters = {'batch_size': [32],
                          'epochs': [64],
                          'dropout_rate1' : [0, 0.1, 0.2],
                          'dropout_rate2' : [0, 0.1, 0.2]
                          }
        else:
            parameters = {'batch_size': [32],
                          'epochs': [32, 64],
                          'dropout_rate1' : [0, 0.1, 0.2],
                          'dropout_rate2' : [0, 0.1, 0.2]
                          }
            
        grid_search = GridSearchCV(estimator = classifier,
                                   param_grid = parameters,
                                   scoring = 'average_precision',
                                   cv = cv,
                                   verbose = 1000)
        if self.categories["index"].min() == 0 :
            grid_search = grid_search.fit(self.X, to_categorical(self.y))
        else:
            grid_search = grid_search.fit(self.X, to_categorical(to_categorical(self.y)[:,1:]))
        best_parameters = grid_search.best_params_
        best_accuracy = grid_search.best_score_
        print("best_parameters\n"); print(best_parameters)
        print("best_accuracy\n"); print(best_accuracy)
        self.batch = best_parameters["batch_size"]
        self.epochs = best_parameters["epochs"]
        self.dropout1 = best_parameters["dropout_rate1"]
        self.dropout2 = best_parameters["dropout_rate2"]
        self.accuracy = best_accuracy
        param = {"accuracy":self.accuracy,
                    "batch_size":self.batch,
                    "epochs":self.epochs,
                    "dropout1":self.dropout1,
                    "dropout2":self.dropout2}
        with open("param.json","r") as f:
            data = json.load(f)
        data["classifier"]["ANN"] = param
        with open("param.json","w") as f:
            json.dump(data,f)
    
    def Train(self, batch, dropout1, dropout2, epochs):
        
        cv = KFold(n_splits = 10, shuffle = True,random_state = 12345)
        i = 0
        df_f1_micro = []
        df_f1_macro = []
        df_accuracy = []
        df_MCC = []
        for train_index,test_index in cv.split(self.X):
            i +=1
            model = Sequential()
            model.add(Dense(units = self.Node1, kernel_initializer = 'uniform', activation = 'relu', input_dim = self.inp))
            model.add(Dropout(dropout1))
            model.add(Dense(units = self.Node2, kernel_initializer = 'uniform', activation = 'relu'))
            model.add(Dropout(dropout2))
            model.add(Dense(units = len(self.categories), kernel_initializer = 'uniform', activation = 'softmax'))
            if self.categories.shape[0] > 2:
                model.compile(optimizer = "sgd", loss = 'categorical_crossentropy', metrics = ['accuracy'])
            else:
                model.compile(optimizer = "sgd", loss = 'binary_crossentropy', metrics = ['accuracy'])
            
            if self.categories["index"].min() == 0 :
                model.fit(self.X[train_index],to_categorical(self.y[train_index]),epochs=epochs, batch_size= batch, verbose = 0)
            else:
                model.fit(self.X[train_index],to_categorical(self.y[train_index])[:,1:],epochs=epochs, batch_size=batch, verbose = 0)
            Y_pred = model.predict(self.X[test_index])
            y_pred = np.argmax(Y_pred, axis=1) 
            confmat = confusion_matrix(self.y[test_index],y_pred)
            f1micro = f1_score(self.y[test_index], y_pred,average="micro")
            f1macro = f1_score(self.y[test_index], y_pred,average="macro")
            accuracy = accuracy_score(self.y[test_index], y_pred)
            MCC = matthews_corrcoef(self.y[test_index], y_pred)
            if i == 1:
                
                df_confmat = pd.DataFrame(confmat)
                
                
            elif i > 1:
                
                temp = pd.DataFrame(confmat)
                df_confmat = df_confmat.append(temp)
            
            
            df_f1_micro.append(f1micro)
            df_f1_macro.append(f1macro)
            df_accuracy.append(accuracy)
            df_MCC.append(MCC)
            
        self.confmat = df_confmat
        self.f1_micro = pd.DataFrame(df_f1_micro,columns=["f1micro"])
        self.f1_macro = pd.DataFrame(df_f1_macro,columns=["f1macro"])
        self.accuracy = pd.DataFrame(df_accuracy,columns=["accuracy"])
        self.MCC = pd.DataFrame(df_MCC, columns = ["MCC"])
        return self.confmat, self.f1_micro, self.f1_macro, self.accuracy, self.MCC
        