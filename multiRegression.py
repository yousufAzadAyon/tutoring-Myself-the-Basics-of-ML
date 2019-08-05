#%%
import numpy as np 
import pandas as pd 

#%%
df = pd.read_csv('dataSets//50_Startups.csv')
df.head()

X = df.iloc[ : , : -1].values
Y = df.iloc[ : , -1].values

#%%

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

#creating dummies

cTransformer = ColumnTransformer([('hotEncoder' ,  OneHotEncoder(), [3])], remainder='passthrough')

X = np.array(cTransformer.fit_transform(X))

#to avoid dummy variable trap

X = X[ : , 1:] 

#%%

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size= 0.3, random_state=0)

#%%

from sklearn.linear_model import LinearRegression
regression = LinearRegression()
regression.fit(X_train, Y_train)
#%%
#Predicting the Test set results

predict = regression.predict(X_train)
