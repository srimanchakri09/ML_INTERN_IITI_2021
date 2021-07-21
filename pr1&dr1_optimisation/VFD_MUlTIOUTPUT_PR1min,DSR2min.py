import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import pipeline,neural_network
from sklearn import preprocessing
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import KFold,cross_val_score
np.random.seed(42)
num_errors=0

#loading the csv file
df=pd.read_csv("myfile1multi_10850samples.csv",header=None)
df=df.to_numpy()
#print(df)
x=df[1:,0:9].astype(float)
y=df[1:,9:].astype(float)

#splitting the data into training and testing data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

#normalising the data
scalar=preprocessing.MinMaxScaler()

x=scalar.fit_transform(x)
y=scalar.fit_transform(y)
x_train=scalar.fit_transform(x_train)
x_test=scalar.fit_transform(x_test)
y_train=scalar.fit_transform(y_train)
y_test=scalar.fit_transform(y_test)

#getting the estimators required for the pipeline
estimators = []
estimators.append(('minmaxscaler', preprocessing.MinMaxScaler()))
estimators.append(('mlp',neural_network.MLPRegressor(hidden_layer_sizes=(128,64,32),learning_rate_init=0.001,early_stopping=True,activation='relu')))
pipe= pipeline.Pipeline(estimators)
pipemodel=pipe.fit(x_train,y_train)

#accuracy by using KFold algorithm
fold=KFold(n_splits=10,random_state=42,shuffle=True)
results = cross_val_score(pipemodel, x, y, cv=fold,n_jobs=-1)
print("Cross_val_score/accuracy : %f " % (results.mean()*100))

#predicting the values
y_pred=pipemodel.predict(x_test)
#getting accuracy of the model with test and train data
score_train=pipemodel.score(x_train,y_train)
score_test= pipemodel.score(x_test,y_test)
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

#comparing first two predicted and actual values
print('actual first pr1min %f , predicted first pr1min %f '%(y_test[0][0],y_pred[0][0]))
print('actual first dsr2min %f , predicted first dsr2min %f '%(y_test[0][1],y_pred[0][1]))
print('actual second pr1min %f , predicted second pr1min %f '%(y_test[1][0],y_pred[1][0]))
print('actual second dsr2min %f , predicted second dsr2min %f '%(y_test[1][1],y_pred[1][1]))


# output of the code :
#test data score:  97.62727323004303
#training data score:  97.7626979287762
#Cross_val_score/accuracy : 97.285030
#mean absolute error :  0.026987316848448014
#Huber loss :  0.0013892036
#actual first pr1min 0.272026 , predicted first pr1min 0.269571
#actual first dsr2min 0.156841 , predicted first dsr2min 0.122385
#actual second pr1min 1.000000 , predicted second pr1min 1.027115
#actual second dsr2min 0.044494 , predicted second dsr2min 0.037523


