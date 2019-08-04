#%%
#libraries

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
#%%
#reading the data file
df = pd.read_csv('dataSets\\studentscores.csv')

X = df.iloc[ : , : -1].values #indipendet variables
Y = df.iloc[ : , -1].values #dependent variables

df.head()

#%%
#checking for missing values

df.isnull().sum()

#%%

#spliting data into test and train sets

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

#%%

#fitting regression model to the training set

from sklearn.linear_model import LinearRegression
regression = LinearRegression()
regression = regression.fit(X_train, Y_train)


#%%

#predicting the result 

Y_predict = regression.predict(X_test)


#%%
#train result visualization
plt.scatter(X_train, Y_train, color = 'red')
plt.plot(X_train, regression.predict(X_train), color = 'blue')


#%%
#test result visualizatiom 
plt.scatter(X_test, Y_test, color ='blue')
plt.plot(X_test, regression.predict(X_test),color = 'red')

#%%
