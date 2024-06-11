from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
import pandas as pd
from sklearn.model_selection import train_test_split 
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.preprocessing import LabelEncoder 
from sklearn.linear_model import LogisticRegression 
import matplotlib.pyplot as plt
from sklearn.metrics import log_loss

sonar=pd.read_csv(r"D:/ML/Day1/Cases/Sonar/Sonar.csv") 
le=LabelEncoder()
y=le.fit_transform(sonar['Class'])
X=sonar.drop('Class',axis=1)
print(le.classes_)


gaussian= GaussianNB()
kfold = StratifiedKFold(n_splits=5,shuffle=True,random_state=24)
results=cross_val_score(gaussian,X,y, cv= kfold, scoring='roc_auc')
print("results =",results)
