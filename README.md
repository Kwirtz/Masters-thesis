# Masters-thesis
This repository contains all the files used in my master's thesis on comparison of machine learning algorithms performance.

Each folder in "Code" contains two python files. The tuning files is used to decide hyperparameters from SVM and ANN and do not require running.
The All file (e.g: IrisALL) should be runned first to obtain 10 (cross-validation) performance score for each algorithm on the dataset
To run the "all" files, the datasets have been set in the respective folder. Credit goes to the UCI repository, "data.world" and Kuhn and Johnson (Applied predictive modeling) for the datasets.  
An exhaustive list on where to find them can be found on the bottom of the document.

Once every file has been run you should have multiple csv files in each folders. This allows to run the Statistical test file for each performance measure. 

If you are interested in reading the full paper you need to run the .tex file

Adult:
https://archive.ics.uci.edu/ml/datasets/adult

bank :
https://archive.ics.uci.edu/ml/datasets/bank+marketing

basket:
https://data.world/exercises/logistic-regression-exercise-1

Breast:
https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)

Contraceptive:
https://archive.ics.uci.edu/ml/datasets/Contraceptive+Method+Choice

CreditScoreAus:
http://archive.ics.uci.edu/ml/datasets/statlog+(australian+credit+approval)

CreditScoreGer:
https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data)

Iris:
https://archive.ics.uci.edu/ml/datasets/iris

Sonar:
http://archive.ics.uci.edu/ml/datasets/connectionist+bench+(sonar,+mines+vs.+rocks)

Wine:
https://archive.ics.uci.edu/ml/datasets/wine

Alzheimer and Cell can be found in the package "Applied Predictive Modeling" in R which is accessed with the rpy2 library in python. 
