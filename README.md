# BLENDED_LERNING
# Implementation-of-Multiple-Linear-Regression-Model-with-Cross-Validation-for-Predicting-Car-Prices

## AIM:
To write a program to predict the price of cars using a multiple linear regression model and evaluate the model performance using cross-validation.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
```
/*
Program to implement the multiple linear regression model for predicting car prices with cross-validation.
Developed by: 
RegisterNumber:
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt

#1. Load and prepare data
data=pd.read_csv('CarPrice_Assignment.csv')


data=data.drop(['car_ID','CarName'],axis=1)
data=pd.get_dummies(data, drop_first=True)

#2.Split Data
x=data.drop('price', axis=1)
y=data['price']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

#3. Create Model and Train it

model=LinearRegression()
model.fit(x_train,y_train)

#4. Evaluate with cross validation
print('Name:G Dhinesh kumar')
print("Reg No:212225240036")
print("\n=== Cross-Validation ===")
cv_scores=cross_val_score(model,x,y,cv=5)
print("Fold R2 scores",[f"{score:.4f}" for score in cv_scores])
print(f"Average R2: {cv_scores.mean():.4f}")


#5.Test set evaluation
y_pred=model.predict(x_test)
print(f"MSE: {mean_squared_error(y_test,y_pred):.2f}")
print(f"MAE: {mean_absolute_error(y_test,y_pred):.2f}")
print(f"R-square: {r2_score(y_test,y_pred):.4f}")

#6. Visualization
plt.figure(figsize=(8,6))
plt.scatter(y_test,y_pred,alpha=0.6)
plt.plot([y.min(),y.max()], [y.min(),y.max()],'r--')
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title("Actual vs predicted price")
plt.show()  
*/
```

## Output:
<img width="1244" height="211" alt="image" src="https://github.com/user-attachments/assets/841cdf7e-d5f1-4c4a-ba36-763119cd3eb3" />

<img width="1294" height="681" alt="image" src="https://github.com/user-attachments/assets/ce1a4ecf-13a4-4642-80b1-5471e9ac2512" />


## Result:
Thus, the program to implement the multiple linear regression model with cross-validation for predicting car prices is written and verified using Python programming.
