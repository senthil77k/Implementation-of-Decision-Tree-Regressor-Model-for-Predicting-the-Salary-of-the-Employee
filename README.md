# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the standard libraries.

2. Upload the dataset and check for any null values using .isnull() function.

3. Import LabelEncoder and encode the dataset.

4. Import DecisionTreeRegressor from sklearn and apply the model on the dataset.

5. Predict the values of arrays.

6. Import metrics from sklearn and calculate the MSE and R2 of the model on the dataset.

7. Predict the values of array.

8. Apply to new unknown values.

## Program:
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

Developed by:  c.senthil kumaran

RegisterNumber:  212223220103
```python
import pandas as pd
data=pd.read_csv("Salary.csv")
data.head()

data.info()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["Position"]=le.fit_transform(data["Position"])

x=data[["Position","Level"]]
x.head()

y=data[["Salary"]]
y.head()

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)
from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
y_pred

from sklearn import metrics
mse=metrics.mean_squared_error(y_test,y_pred)
mse

r2=metrics.r2_score(y_test,y_pred)
r2

dt.predict([[5,6]])
```
## Output:
## data

![370131767-297709b0-3e39-4f81-882e-6c7449842bb9](https://github.com/user-attachments/assets/b636e189-b16e-4ae2-ac0d-c6c9cc4f73aa)

## X_variables
![370131823-b1419ed9-aa24-4816-bc02-2d4c9679d649](https://github.com/user-attachments/assets/e7ab6a04-f9d5-40af-af6c-47ff3fb7b07a)

## y_variables
![370131843-d63e4ac0-8d44-45ca-9100-fc9a89aee0b2](https://github.com/user-attachments/assets/08eb7758-2d4f-4288-b157-ef8996e648a4)

## y_pred
![370131876-60dcc712-2d32-4621-a4a2-b76a9f312579](https://github.com/user-attachments/assets/9fc6c568-1544-4798-862d-a2cfb00a634c)

## MSE
![370131912-c1e6cca8-2fd1-4ee4-a6bc-20c70dac0bb1](https://github.com/user-attachments/assets/1361cd20-78d5-48f2-947a-c50147820555)

![370131936-a84e629b-b599-40fe-a08d-1b4c571cb199](https://github.com/user-attachments/assets/b523e814-f684-4a41-8aa3-ea993afd2317)

## predictions
![370131959-846051b4-f62f-45a8-95c3-274b75fe1016](https://github.com/user-attachments/assets/ceeb5082-199d-4188-a0f4-9cf095100303)

## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
