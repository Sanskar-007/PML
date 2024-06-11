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
y=sonar['Class']
X=sonar.drop('Class',axis=1)


X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size = 0.3, 
                                                    random_state=24,
                                                    stratify=y)






