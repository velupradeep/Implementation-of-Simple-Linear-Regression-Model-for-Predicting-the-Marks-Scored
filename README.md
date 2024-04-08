

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



```








```

```



```

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

```


```

## Output:
![image](https://github.com/velupradeep/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/150329341/1cbd36d1-8030-4489-bdc7-2cefabd21477)
```



```




![image](https://github.com/velupradeep/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/150329341/fecc0c1f-e6d9-4b5e-9013-8f42208135d6)


![image](https://github.com/velupradeep/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/150329341/c4e2c805-8c7f-4f6d-920a-6f5c2decb0cc)


![image](https://github.com/velupradeep/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/150329341/84ebdd0a-27a5-49f6-838f-d873ec6bab78)


![image](https://github.com/velupradeep/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/150329341/0eadd030-6e53-4772-bfe3-c61aee0367cb)


![image](https://github.com/velupradeep/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/150329341/e5f9a69d-ef68-4168-829d-11d602445c31)


![image](https://github.com/velupradeep/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/150329341/ad9c4e63-3099-4708-acf3-f6dd253f4729)


![image](https://github.com/velupradeep/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/150329341/c37abebf-0495-47e8-9025-8ef4563b2cfc)


![image](https://github.com/velupradeep/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/150329341/d887d7b0-5a95-4b95-ab2e-8f7a87ee2390)
```





````

```





```





![image](https://github.com/velupradeep/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/150329341/d82a1425-0c61-41fb-8db0-6ed91f299249)


![image](https://github.com/velupradeep/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/150329341/dc15f54d-3323-4cc1-a1bc-778479c2a138)


![image](https://github.com/velupradeep/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/150329341/ae42f178-ea12-4e73-a56d-ede3f5493f26)


![image](https://github.com/velupradeep/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/150329341/1cd71f5d-6c29-4a10-b7ac-1a0fff0ce145)


![image](https://github.com/velupradeep/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/150329341/2506dfcd-392a-4509-9492-7933628ca4f9)




















## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
