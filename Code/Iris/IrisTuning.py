# Importing the libraries
import math
import pandas as pd
from sklearn import svm
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout 
from keras.utils.np_utils import to_categorical
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.preprocessing import StandardScaler


# Importing the dataset

headers = [1,2,3,4,"Species"]

dataset = pd.read_csv("iris.csv",delimiter=",",header = None,names = headers)
dataset["Species"] = dataset["Species"].astype("category")
dataset["SpeciesNew"] = dataset["Species"].cat.codes

X = dataset.iloc[:, 0:-2].values 
y = dataset.iloc[:, -1].values

Label = pd.value_counts(y).to_frame().reset_index()
print(Label)

sc = StandardScaler()
Xsc = sc.fit_transform(X)


def SVMTUNINGRBF():
	C_range = [0.1,1,10,100]
	gamma_range = [0.0000001, 0.00001, 0.01, 0.1, 1]
	parameters = { 'C' : C_range, "gamma" : gamma_range}
	cv = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=12345)
	grid = GridSearchCV(svm.SVC(kernel = "rbf"), param_grid=parameters, cv=cv, verbose = 1000)
	grid.fit(X, y)
	print("The best parameters are %s with a score of %0.2f"
		% (grid.best_params_, grid.best_score_))






def SVMTUNINGLinear():
	C_range = [0.1,1,10,100]
	parameters = {"C" : C_range}
	cv = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=12345)
	grid = GridSearchCV(svm.LinearSVC(), param_grid=parameters, cv=cv, verbose = 1)
	grid.fit(X, y)
	print("The best parameters are %s with a score of %0.2f"
		% (grid.best_params_, grid.best_score_))



####################################### Tuning ANN
#######################################
#######################################

# Tuning the ANN

Node1 = math.floor(math.sqrt(((len(Label)+2)*len(y)))) + math.floor(2*math.sqrt((len(y))/(len(Label)+2)))
Node2 = math.floor(len(Label)*math.sqrt(len(y)/(len(Label)+2)))
_,inp = X.shape

def build_classifier(dropout_rate1=0.0,dropout_rate2=0.0):
    classifier = Sequential()
    classifier.add(Dense(units = Node1, kernel_initializer = "uniform", activation = "relu", input_dim = inp ))
    classifier.add(Dropout(dropout_rate1))
    classifier.add(Dense(units = Node2, kernel_initializer = "uniform", activation = 'relu'))
    classifier.add(Dropout(dropout_rate2))
    classifier.add(Dense(units = len(Label), kernel_initializer = "uniform", activation = 'softmax'))
    classifier.compile(optimizer = "sgd", loss = 'categorical_crossentropy', metrics = ['accuracy'])
    return classifier


def TuningANN():
    cv = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=12345)
    classifier = KerasClassifier(build_fn = build_classifier)
    parameters = {'batch_size': [4],
                  'epochs': [32,64],
                  "dropout_rate1" : [0, 0.1, 0.2, 0.3, 0.4],
                  "dropout_rate2" : [0, 0.1, 0.2, 0.3, 0.4]
                  }
    grid_search = GridSearchCV(estimator = classifier,
                               param_grid = parameters,
                               scoring = 'average_precision',
                               cv = cv,
                               verbose = 1000)
    grid_search = grid_search.fit(X, to_categorical(y))
    best_parameters = grid_search.best_params_
    best_accuracy = grid_search.best_score_
    print("best_parameters\n"); print(best_parameters)
    print("best_accuracy\n"); print(best_accuracy)

TuningANN()
SVMTUNINGRBF()
SVMTUNINGLinear()

