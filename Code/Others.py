import os
from sklearn.model_selection import KFold
from sklearn.metrics.classification import cohen_kappa_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import matthews_corrcoef
from sklearn.tree import export_graphviz
from PIL import Image
import datetime as dt
import pandas as pd



def MakeTree(number,classifier,X):
	
	export_graphviz(classifier.estimators_[number],
                feature_names=pd.DataFrame(X).columns,
                filled=True,
                rounded=True,
                out_file = "Tree.dot")


	os.system('dot -Tpng Tree.dot -o Tree.png') # need to dl graphviz and add to environment path
	im = Image.open("Tree.png")
	im.rotate(360).show()

def Training(X, y, classifier, RF = False):
    
    cv = KFold(n_splits = 10, shuffle = True,random_state = 12345)
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
        MakeTree(0,classifier,X)
    else:
        pass
    
    return df_confmat, df_f1_micro, df_f1_macro, df_accuracy, df_MCC