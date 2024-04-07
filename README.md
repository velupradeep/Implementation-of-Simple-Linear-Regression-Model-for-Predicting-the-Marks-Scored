                                                              #NAME:PRADEEP V
                                                              #REG NO:212223240119

# EX-02 Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

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
Developed by: PRADEEP V
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
df.tail()
#segregating data to variables
X=df.iloc[:,:-1].values
X
Y=df.iloc[:,1].values
Y
#splitting training and test data
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)
Y_pred=regressor.predict(X_test)

#displaying predicted values
Y_pred
Y_test
#graph plot for training data
plt.scatter(X_train,Y_train,color='orange')
plt.plot(X_train,regressor.predict(X_train),color='red')
plt.title("Hours VS Scores(Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
plt.scatter(X_test,Y_test,color='purple')
plt.plot(X_test,regressor.predict(X_test),color='yellow')
plt.title("Hours VS Scores(Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_squared_error(Y_test,Y_pred)
print('MSE =',mse)
mae=mean_absolute_error(Y_test,Y_pred)
print('MAE =',mae)
rmse=np.sqrt(mse)
print('RMSE =',rmse)

```

## Output:
![image](https://github.com/velupradeep/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/150329341/2725adc9-1b2a-45ca-9f4f-3a4d11bdbcf6)

![image](https://github.com/velupradeep/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/150329341/7adfd6ca-0830-484d-b36d-ccf4def3d519)

![image](https://github.com/velupradeep/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/150329341/3307d249-8432-4cec-bbbf-be9139f667bc)

![image](https://github.com/velupradeep/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/150329341/cbcadc5a-0622-4628-b885-467f1dea286d)

![image](https://github.com/velupradeep/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/150329341/874d6525-1e17-4b6e-81b6-64ac820c0f4b)

![image](https://github.com/velupradeep/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/150329341/238f1e62-fb05-4d35-9dfb-eb4f97625214)

![image](https://github.com/velupradeep/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/150329341/e7430c82-df22-4920-a6b2-7ad3d5e4dfbb)

![image](https://github.com/velupradeep/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/150329341/2badf3e9-b987-471b-a316-f8ea5722de7e)

![image](https://github.com/velupradeep/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/150329341/bee99dee-2285-423b-8cd7-59dccf6bd328)

![image](https://github.com/velupradeep/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/150329341/942765f0-3077-4fa4-bb20-20dac68dbb46)

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
