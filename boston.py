import pandas as pd
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score 
from sklearn.linear_model import Ridge, LinearRegression,ElasticNet,Lasso
from sklearn.metrics import log_loss,accuracy_score
import numpy as np 
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

boston = pd.read_csv("D:/DATA_SET/Datasets/Boston.csv")
y = boston['medv']
X= boston.drop('medv',axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=24)

lasso = Lasso()
poly= PolynomialFeatures(degree=1).set_output(transform='pandas')

pipe = Pipeline([('POLY', poly), ('LR', lasso)])
print(pipe.get_params())
params = {'POLY__degree': [1,2,3],
          'LR__alpha': np.linspace(0.001, 5, 10)}

kfold = KFold(n_splits=5, shuffle=True, random_state=24)

gcv_lass = GridSearchCV(pipe, param_grid=params, cv=kfold)
gcv_lass.fit(X,y)
print(gcv_lass.best_score_)
print(gcv_lass.best_params_)

#best_model = gcv_lass.best_estimator_
#print(best_model.named_steps.LR.coef)
#print(best_model.named_steps.LR.intercept_)

#################Inferencing################
##### in case refit=false
#poly=PolynomialFeatures(degree=3)
#ridge=Ridge()

#best_model = gcv_ridge.

