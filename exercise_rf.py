# -*- coding: utf-8 -*-
"""
Created on Sun May 19 00:14:50 2019

@author: rojay
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing as pr
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix 
from sklearn.model_selection import train_test_split

#Import the raw data
raw_data = pd.read_csv("pml-training.csv")

#Find the columns that contain NA values (the columns either contain all NA values or not)
a=pd.isna(raw_data.loc[0,:])

#Use true false indexing to remove the NA columns
df=raw_data.loc[:,~a]

df=df.iloc[:,0:59]

#remove the first 7 columns, which don't contain exercise data
df=df.iloc[:,7:]

x_train=df

#Use one hot encoding to convert the final classifications to numerical values
y_train=pr.LabelBinarizer()
y_train.fit(raw_data['classe'])
y_train=y_train.transform(raw_data['classe'])

#Create parameter for grid serach for random forest (the number of variables selected at each split)
parameters={"max_features":[5,10,25,40]}
#Using random forest classifier in sci-kit learn
clf = RandomForestClassifier(n_estimators=100, random_state=0)
clf.cv = GridSearchCV(clf, parameters, cv=5)
#train random forest on training set using grid search
clf.cv.fit(x_train,y_train)

#Get and pre-process test set
test_data=pd.read_csv('pml-testing.csv')
dft=test_data.loc[:,list(~a)]
dft=dft.iloc[:,0:59]
dft=dft.iloc[:,7:]
x_test=dft

#Make final prediction
clf.cv.predict(x_test)
y_train=pr.LabelBinarizer()
y_train.fit(raw_data['classe'])
#Get results, and convert from numeric results back to classes
y_train.inverse_transform(clf.cv.predict(x_test))