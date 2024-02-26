# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm:
1.Import the standard Libraries.
2.Set variables for assigning dataset values.
3.Import linear regression from sklearn.
4.Assign the points for representing in the graph.
5.Predict the regression for marks by using the representation of the graph.
6.Compare the graphs and hence we obtained the linear regression for the given datas.


## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Pradeep V
RegisterNumber:  212223240119
*/
```
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv("C:/Users/PRADEEP V/Downloads/student_scores.csv")
df.head()
```
```
df.tail()
```
```
#segregating data to variables
X=df.iloc[:,:-1].values
X
```
```
Y=df.iloc[:,1].values
Y
```
```
#splitting training and test data
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)
Y_pred=regressor.predict(X_test)
```
```
#displaying predicted values
Y_pred
```
```
Y_test
```
```
#graph plot for training data
plt.scatter(X_train,Y_train,color='orange')
plt.plot(X_train,regressor.predict(X_train),color='red')
plt.title("Hours VS Scores(Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
```
```
plt.scatter(X_test,Y_test,color='purple')
plt.plot(X_test,regressor.predict(X_test),color='yellow')
plt.title("Hours VS Scores(Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
```
```
mse=mean_squared_error(Y_test,Y_pred)
print('MSE =',mse)
```
```
mae=mean_absolute_error(Y_test,Y_pred)
print('MAE =',mae)
```
```
rmse=np.sqrt(mse)
print('RMSE =',rmse)
```

## Output:
![image](https://github.com/velupradeep/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/150329341/f09d76c2-8f60-411b-8a31-bc0ed7a6dcd0)
![image](https://github.com/velupradeep/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/150329341/9f128445-69ed-4f65-993f-32bd5923163d)
![image](https://github.com/velupradeep/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/150329341/06947775-d60c-40ad-9384-291c1e3199ef)
![image](https://github.com/velupradeep/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/150329341/d39e4e00-84fe-4fce-9476-e007db1ff69d)
![image](https://github.com/velupradeep/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/150329341/82524506-64c9-46ee-be01-413da5c77d08)
![image](https://github.com/velupradeep/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/150329341/5a1afca8-ca06-4cab-a934-b0be80831729)

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
