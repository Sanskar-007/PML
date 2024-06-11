import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

concrete=pd.read_csv(r"D:/ML/Day1/Cases/Concrete Strength/Concrete_Data.csv") 
X=concrete.drop('Strength',axis=1)
y=concrete['Strength']

X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size = 0.3, 
                                                    random_state=24)

lr=LinearRegression()
lr.fit(X_train,y_train)
y_pred = lr.predict(X_test)
print(r2_score(y_test, y_pred))

################### K fold ###############################
kfold = KFold(n_splits=5, shuffle=True, random_state=24)
results = cross_val_score(lr,X,y,cv=kfold) #scoring='r2' By Default r2 score
print(results.mean())
