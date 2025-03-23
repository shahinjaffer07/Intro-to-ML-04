# SGD-Regressor-for-Multivariate-Linear-Regression

## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Load California housing data, select three features as X and two target variables as Y, then split into train and test sets.

2.Standardize X and Y using StandardScaler for consistent scaling across features.

3.Initialize SGDRegressor and wrap it with MultiOutputRegressor to handle multiple targets.

4.Train the model on the standardized training data.

5.Predict on the test data, inverse-transform predictions, compute mean squared error, and print results.

## Program:
```
/*
Program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor.
Developed by: SHAHIN J
RegisterNumber: 212223040190

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import SGDRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

dataset = fetch_california_housing()
df=pd.DataFrame(dataset.data,columns=dataset.feature_names)
df['HousingPrice']=dataset.target
print(df.head())

df.info()

X=df.drop(columns=['AveOccup','HousingPrice'])
X.info()

Y=df[['AveOccup','HousingPrice']]
Y.info()

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=42)
scaler_X=StandardScaler()
scaler_Y=StandardScaler()
X_train=scaler_X.fit_transform(X_train)
X_test=scaler_X.transform(X_test)
Y_train=scaler_Y.fit_transform(Y_train)
Y_test=scaler_Y.transform(Y_test)
print(X_train)
print(X_test)
print(Y_train)
print(Y_test)

sgd=SGDRegressor(max_iter=1000,tol=1e-3)
multi_output_sgd=MultiOutputRegressor(sgd)
multi_output_sgd.fit(X_train,Y_train)

Y_pred=multi_output_sgd.predict(X_test)
Y_pred=scaler_Y.inverse_transform(Y_pred)
Y_test=scaler_Y.inverse_transform(Y_test)
mse=mean_squared_error(Y_test,Y_pred)
print("Mean Squared Error:",mse)
print(Y_pred)

*/
```

## Output:
## Head
![image](https://github.com/user-attachments/assets/6f51ac40-2a8f-45b0-be7a-8e7336b983a7)
## Dataset info
![image](https://github.com/user-attachments/assets/1a478ee6-0625-48b8-9797-e080a78d51a6)
## Removing columns
![image](https://github.com/user-attachments/assets/c2b1dbbb-af00-4739-a78d-3bfb78e92297)
## Columns info
![image](https://github.com/user-attachments/assets/ac0c5108-ffde-4761-b820-1ecbbb17cc1d)
## X_train,X_test,Y_train,Y_test
![image](https://github.com/user-attachments/assets/b0cae13d-e9ad-4d4a-ad50-779b4301da5d)
## MultiOutputRegressor(sgd)
![image](https://github.com/user-attachments/assets/bce96a34-32e8-4bac-afec-ba88f92c86d3)
## Mean Squared error
![image](https://github.com/user-attachments/assets/2820506b-c9e6-4c13-aa6e-008003563844)

![image](https://github.com/user-attachments/assets/4606febe-396d-4c59-b43a-0807fb0c54e2)
## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
