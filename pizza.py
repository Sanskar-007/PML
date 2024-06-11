import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression





pizza= pd.read_csv('D:/DATA_SET/pizza.csv')
#print(pizza.head())


xi=pizza['Promote']
yi=pizza['Sales']

n=pizza.shape[0]

xbar=np.mean(xi)
ybar=np.mean(yi)

m_xi_yi=np.sum(xi*yi)/n
m_xi_2=np.sum(xi**2)/2

b1=(m_xi_yi-(xbar*ybar))/(m_xi_2-(xbar**2))
b0=ybar-b1*xbar

###########################################################



X=pizza[['Promote']]
y=pizza['Sales']

lr=LinearRegression()
lr.fit(X,y)
print(lr.intercept_, lr.coef_)

###########################################################

A=np.array([[2,1],[3,2]])

b=np.array([4,7])

print(np.linalg.solve(A,b))

###########################################################
































