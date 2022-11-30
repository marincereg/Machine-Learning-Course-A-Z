
#%matplotlib qt

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time

# import and extract data
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

#split the data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

# make LinearRegression 
from sklearn.linear_model import LinearRegression
regressor = LinearRegression() # create instance
regressor.fit(X_train, y_train) # train regressor
y_pred = regressor.predict(X_train) # make prediciton / linear regression

#plot 
plt.scatter(X_train, y_train, color = 'Blue')
plt.plot(X_train, y_pred, color = 'Green')
plt.scatter(X_test, y_test, color = 'red')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')


def GetLinearRegression_LS (x,y):
    A = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A, y, rcond=None)[0]
    return m,c

def GetValuesFromArray (Arr):
    #SkLearn returns array with [xx,1] and this is making trouble in linear reggression we just need values
    ar2 = []
    for i in Arr:
        ar2.append(i[0])
    return ar2

# Get quality of prediciton
Errors = []
for i in range (0,len(X_test)):
    # From predictions point create equation y = mx + c using LeastSqare method (10 points in linear position)
    x_arr = GetValuesFromArray(X_train)     
    m,c = GetLinearRegression_LS(x_arr,y_pred)
    
    # Get y value from the prediction 
    Y_onTest = X_test[i]*m + c
    # Get the errors percentages
    Errors.append( (y_test[i]-Y_onTest[0])/y_test[i])
    #print (y_test[i] , Y_onTest[0])
# make Average Error
AvgError = np.sum(Errors)/len(Errors) 
print("Prediction to real error is " + str(AvgError*100) + " %")

"""
not using pure y_pred is because ypred is array of few elements and not the equation itself
with that on y_test which is 10 random points, comparison will not be sucessfull
"""





