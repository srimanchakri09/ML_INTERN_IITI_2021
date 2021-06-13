import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model,metrics
from sklearn.model_selection import train_test_split

num_errors=0
np.random.seed(0)

#intialising the values to create the dataset

z=0.5*np.random.randn(100)
x=np.random.randint(-1,1,(100,))
for i in range(len(x)):
    if x[i]==0:
        x[i]=1

#intialising the system vector
h=[0.5,1,0.5]

#convolving the input X values

x1=np.zeros(100)
for i in range(100):
    x1[i]=x[i-1]
x2=np.zeros(100)
for i in range(100):
    x2[i]=x[i-2]


X=np.stack((x,x1,x2),axis=1)
y=np.zeros(100)
temp=np.zeros(100)

#creating the y vector with the FIR filter equation

#for i in range(100):
#   y[i] = X[i][0] * 0.5 + X[i][1] * 1 + X[i][2] * 0.5 + z[i]   #for limited co-efficients

for j in range(len(h)):
    for i in range(len(y)):
        temp[i]= temp[i]+(X[i][j]*h[j])
        y[i] = temp[i] + z[i]

print(np.shape(X),np.shape(y),np.shape(z))
#print(y)
#print(X)

#APPLYING THE LINEAR REGRESSION ALGORITHM FOR THE DATA SET TO PREDICT THE CHANNEL CO-EFFICIENTS

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
model=linear_model.LinearRegression()
model.fit(x_train, y_train)

#PREDICTING THE Y VALUES  AND THEN ERROR
y_pred=model.predict(x_test)
error=y_test-y_pred
for i in range(len(error)):
    if error[i]!=0:
        num_errors=num_errors+1

print(num_errors)

accuracy = model.score(x_test,y_test)
print('accuracy of model: ',accuracy )
# The coefficients
print('Coefficients: \n', model.coef_)
# the intercept
print('intercept: \n',model.intercept_)
# The mean squared error
print('Mean squared error: %.2f'% metrics.mean_squared_error(y_test, y_pred))
