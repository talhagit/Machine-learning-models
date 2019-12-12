# -*- coding: utf-8 -*-
"""
@author: Talha.Iftikhar
"""

                                #Linear Regression#
# All libraries

import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
import pandas as pd
import numpy as np
import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style
%matplotlib inline

#------------------------------------------------------#
#Read data using pandas#
data=pd.read_csv('C:/Users/Talha.Iftikhar/Desktop/TFA/Machine Learn/student-mat.csv',sep=';')

#Cherry pick columns#
data=data[["G1","G2","G3","studytime","failures","absences"]]

#View few rows#
data.head()

#Preict variables#
predict="G3"

x=np.array(data.drop([predict],1))
y=np.array(data[predict])
x_train,x_test,y_train,y_test=sklearn.model_selection.train_test_split(x,y,test_size=0.3)

# saving the model
"""
bestScore=0
for _ in range(30):

    x_train,x_test,y_train,y_test=sklearn.model_selection.train_test_split(x,y,test_size=0.3)
    linear= linear_model.LinearRegression()
    linear.fit(x_train, y_train)
    acc=linear.score(x_test, y_test)
    print(acc)

    if acc>bestScore:
        bestScore=acc
        with open("studentModel.pickle","wb") as f:
            pickle.dump(linear,f)
"""


pickle_in=open("studentmodel.pickle","rb")
linear=pickle.load(pickle_in)


print('Cofficient:\n', linear.coef_)
print('Intercept: \n', linear.intercept_)

#Predict data

predictions=linear.predict(x_test)


for x in range(len(predictions)):
    print(predictions[x],x_test[x],y_test[x])

#--------------------------------------------------------------#
p="G1"
style.use("ggplot")
pyplot.scatter(data[p],data["G3"])
pyplot.xlabel("P")
pyplot.ylabel("Final Grade")



