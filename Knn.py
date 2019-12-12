# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 18:07:51 2019

@author: Talha.Iftikhar
"""


import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn import linear_model,preprocessing
import matplotlib.pyplot as pyplot


data=pd.read_csv('C:/Users/Talha.Iftikhar/Desktop/TFA/Machine Learn/car.data.csv')

print(data.head())


le=preprocessing.LabelEncoder()

buying=le.fit_transform(list(data["buying"]))
maint=le.fit_transform(list(data["maint"]))
doors=le.fit_transform(list(data["doors"]))
persons=le.fit_transform(list(data["persons"]))
lug_boot=le.fit_transform(list(data["lug_boot"]))
safety=le.fit_transform(list(data["safety"]))
cls=le.fit_transform(list(data["class"]))

predict="class"

x=list(zip(buying,maint,doors,persons,lug_boot,safety))
y=list(cls)

x_train,x_test,y_train,y_test=sklearn.model_selection.train_test_split(x,y,test_size=0.3)

model=KNeighborsClassifier(n_neighbors=7)
model.fit(x_train,y_train)

acc=model.score(x_test,y_test)
print(acc)

predicted=model.predict(x_test)

names=["unacc","acc","good","Top"]

for x in range(len(x_test)):
    print("Predicted ",names[predicted[x]],x_test[x],"Actual: ",names[y_test[x]])
