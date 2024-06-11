import pandas as pd
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score 
from sklearn.linear_model import Ridge, LinearRegression,ElasticNet,Lasso
from sklearn.linear_model import Lasso
import numpy as np 

medical = pd.read_csv(r"D:/DATA_SET/Medical Cost Personal/insurance.csv")
dum_med=pd.get_dummies(medical,drop_first=True)
X = dum_med.drop('charges', axis=1)
y = dum_med['charges']

lr = LinearRegression()
ridge=Ridge()
lasso=Lasso()
elastic=ElasticNet()

kfold = KFold(n_splits=5, shuffle=True, 
              random_state=24)
results=cross_val_score(lr,X,y, cv=kfold)
print(results.mean())

params = {'alpha':np.linspace(0.001, 100,50)}
gcv = GridSearchCV(lasso, param_grid=params,
                   cv=kfold)
gcv.fit(X, y)
print(gcv.best_params_)
print(gcv.best_score_)

##lasso
lasso=Lasso()
params = {'alpha':np.linspace(0.001, 100,4)}
gcv = GridSearchCV(lasso, param_grid=params,
                   cv=kfold)
gcv.fit(X, y)
#pd.csv=pd.DataFrame(gcv,cv_results_)
print(gcv.best_params_)
print(gcv.best_score_)

#####ridge

ridge = Ridge(alpha=0.02)
ridge.fit(X_train, y_train)
y_pred = ridge.predict(X_test)
print(r2_score(y_test, y_pred))


###elastic net

elastic = ElasticNet()
print(elastic.get_params())
params={'alpha':np.linspace(0.001, 50,5),
        'L1_ratio':np.linspace(0.001,1,10)}
gcv=GridSearchCV(elastic, param_grid=params,
                 cv=kfold, scoring='r2')

gcv.fit(X,y)
pd_cv=pd.DataFrame(gcv.cv_results_)
print(gcv.best_params_)
print(gcv.best_score_)
