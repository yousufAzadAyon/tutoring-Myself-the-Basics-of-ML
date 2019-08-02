#%%
#libraries 
import pandas as pd 
import numpy as np 

#%%
#importing data and separating it
df = pd.read_csv('dataSets\\Data.csv')
df.head()

X = df.iloc[ : , : -1].values # X contains the Indipendent Variables
Y = df.iloc[ : , -1].values #  Y contains the Dependent Variables

#%%
#checking for missing values
df.isnull().sum()

#%%
#replacing the missing values with Mean or Median of the column

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan,strategy='mean')
imputer.fit( X[ : , 1:3])
X[ : , 1:3] = imputer.transform(X[ : , 1:3])
print(X)
#%%
#Encoding Catagorical Datas


from sklearn.preprocessing import OneHotEncoder , LabelEncoder
from sklearn.compose import ColumnTransformer

label_X = LabelEncoder()
X[ : , 0] = label_X.fit_transform(X[ : , 0])

label_Y = LabelEncoder()
Y = label_Y.fit_transform(Y)

# Creating Dummy Values

columnTransform = ColumnTransformer([('hotEncoder' , OneHotEncoder(),[0]) ],remainder='passthrough')

X = np.array(columnTransform.fit_transform(X))


#%%

