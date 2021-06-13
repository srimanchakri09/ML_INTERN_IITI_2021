import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model,metrics
from sklearn.model_selection import train_test_split
import seaborn as sns
from scipy.special import expit

num_errors=0

np.random.seed(0)
x=np.random.randint(0,2,(1000,1))
z=np.random.randn(1000,1)
z=0.5*z
y=x+z

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=1)
model=linear_model.LogisticRegression()
model.fit(y_train.reshape(-1,1),x_train.reshape(-1,1))

intercept=model.intercept_
coef=model.coef_
print(intercept,coef)

x_pred=model.predict(y_test)

error=np.zeros(len(x_pred))
for i in range(len(x_pred)):
    error[i]=x_pred[i]-x_test[i]
    if error[i]!=0:
        num_errors=num_errors+1
        print(x_pred[i],x_test[i])
print(num_errors)

probability_pred=model.predict_proba(y_test)

acc=model.score(y_test,x_test)
print('accuraccy_model_score',acc)

confusion_matrix=metrics.confusion_matrix(x_test,x_pred)
print(confusion_matrix)

plt.scatter(y_test,x_test,marker='*',color='red')
plt.ylabel("X_values")
plt.xlabel("Y_values")

#SIGMOID CURVE PLOTTING
temp = np.linspace(-1,2,1000)
loss = expit(temp* coef + intercept).ravel()
plt.yticks([0, 0.2, 0.4, 0.5, 0.6, 0.8, 1])
plt.plot(temp, loss, color='green', linewidth=3)
plt.grid()

# Generating Heat map
plt.figure(figsize=(2,2))
sns.heatmap(confusion_matrix, annot=True, fmt=".3f", linewidths=.5, square = True);
plt.ylabel('Actual label');
plt.xlabel('Predicted label');
all_sample_title = 'Accuracy Score: {0}'.format(acc)
plt.title(all_sample_title, size = 15);


plt.show()
