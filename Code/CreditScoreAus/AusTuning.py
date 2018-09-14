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
# Importing the dataset

headers = ["A1", "A2" , "A3", "A4", "A5", "A6","A7","A8","A9","A10","A11","A12","A13","A14","A15"]
categorical = ["A1","A4","A5","A6","A8","A9","A11","A12"]

dataset = pd.read_csv("australian.dat",sep="\s+", header = None,skiprows=1, names = headers)

for i in categorical:
    dataset[i]=dataset[i].astype("category")

y = dataset.iloc[:, -1].values

for i in categorical:
    dataset = pd.concat([dataset,pd.get_dummies(dataset[i],prefix = i,drop_first= True)],axis = 1)
    dataset.drop([i],axis = 1, inplace = True)

dataset.drop(["A15"],axis = 1, inplace = True)
X = dataset.iloc[:, :].values 

Label = pd.value_counts(y).to_frame().reset_index()
print(Label)


def SVMTUNINGRBF():
	C_range = [0.1, 1, 10, 100]
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
	grid = GridSearchCV(svm.LinearSVC(), param_grid=parameters, cv=cv, verbose = 1000)
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
    classifier.add(Dense(units = Node1, kernel_initializer = "uniform", activation = "relu", input_dim = inp))
    classifier.add(Dropout(dropout_rate1))
    classifier.add(Dense(units = Node2, kernel_initializer = "uniform", activation = 'relu'))
    classifier.add(Dropout(dropout_rate2))
    classifier.add(Dense(units = len(Label), kernel_initializer = "uniform", activation = "sigmoid"))
    classifier.compile(optimizer = "sgd", loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier


def TuningANN():
    cv = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=12345)
    classifier = KerasClassifier(build_fn = build_classifier)
    parameters = {'batch_size': [32],
                  'epochs': [32,64],
                  "dropout_rate1" : [0, 0.1, 0.2, 0.3, 0.4],
                  "dropout_rate2" : [0, 0.1, 0.2, 0.3, 0.4],
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