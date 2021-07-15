import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import pipeline,neural_network
from sklearn import preprocessing
import matplotlib.pyplot as plt
import tensorflow as tf
np.random.seed(42)
num_errors=0

#loading the csv file
df=pd.read_csv("myfile1multi.csv",header=None)
df=df.to_numpy()
#print(df)
x=df[1:,0:9].astype(float)
y=df[1:,9:].astype(float)

#normalising the data
scalar=preprocessing.MinMaxScaler()

#splitting the data into training and testing data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

x_train=scalar.fit_transform(x_train)
x_test=scalar.fit_transform(x_test) 
y_train=scalar.fit_transform(y_train)
y_test=scalar.fit_transform(y_test)

#getting the estimators required for the pipeline
estimators = []
estimators.append(('minmaxscaler', preprocessing.MinMaxScaler()))
estimators.append(('mlp',neural_network.MLPRegressor(hidden_layer_sizes=(81,45,27),learning_rate_init=0.01,early_stopping=True,activation='relu')))
pipe= pipeline.Pipeline(estimators)

#fitting the model
pipe.fit(x_train,y_train)
#predicting the values
y_pred=pipe.predict(x_test)
#getting accuracy of the model with test and train data
score_train=pipe.score(x_train,y_train)
score_test= pipe.score(x_test,y_test)
print('test data score: ',score_test*100)
print('training data score: ',score_train*100)

#making all the values positive
for i in range(len(y_pred)):
    y_pred[i]=abs(y_pred[i])

#finding mean absolute error and huber loss
mae_abs=metrics.mean_absolute_error(y_test,y_pred)
print("mean absolute error : ",mae_abs)
huber_loss=tf.losses.Huber(delta=1.0)
print("Huber loss : ",huber_loss(y_test,y_pred).numpy())

#comparing predicted and actual values
print('actual first value %f , predicted first value %f '%(y_test[0][0],y_pred[0][0]))
print('actual first value %f , predicted first value %f '%(y_test[0][1],y_pred[0][1]))
print('actual second value %f , predicted second value %f '%(y_test[1][0],y_pred[1][0]))


# output of the code :
#test data score:  99.22336755601668
#training data score:  99.3972575392786
#actual first value 0.003304 , predicted first value 0.001734
#actual second value 0.634361 , predicted second value 0.617901
#mean absolute error :  0.01173260997333336
#Huber loss :  0.095701024

