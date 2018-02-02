# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 20:18:03 2018

@author: Lenovo
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing dataset
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
train = train_df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis = 1)
test = test_df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis = 1)

dataset = [train, test]

for data in dataset:
    age_avg = data['Age'].mean()
    age_std = data['Age'].std()
    age_null_count = data['Age'].isnull().sum()
    age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)
    data['Age'][np.isnan(data['Age'])] = age_null_random_list
    data['Age'] = data['Age'].astype(int)

for data in dataset:
    data['Embarked'] = data['Embarked'].fillna('S')
    
for data in dataset:
    data['Fare'] = data['Fare'].fillna(train['Fare'].mean())

train.isnull().sum()
test.isnull().sum()

for data in dataset:
    data['Sex'] = data['Sex'].map({'male' :0, 'female': 1}).astype(int)
    
X = train.iloc[:, 1:].values
Y = train.iloc[:, 0].values

#encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
#turn category into ordinal number
labelencoder_X_1 = LabelEncoder()
X[:, 6] = labelencoder_X_1.fit_transform(X[:, 6])
onehotencoder = OneHotEncoder(categorical_features = [6])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]

X_pred = test.iloc[:, :].values 
   
labelencoder_X_2 = LabelEncoder()
X_pred[:, 6] = labelencoder_X_2.fit_transform(X_pred[:, 6])
onehotencoder = OneHotEncoder(categorical_features = [6])
X_pred = onehotencoder.fit_transform(X_pred).toarray()
X_pred = X_pred[:, 1:]

#splitting dataset into train and test
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

#feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)

import keras
from keras.models import Sequential
from keras.layers import Dense

#initiliasing ANN
classifier = Sequential()

#adding input layer and the first hidden layer
classifier.add(Dense(5, kernel_initializer='uniform',  activation = 'relu', input_shape = (8,)))

#adding second hidden layer
classifier.add(Dense(5, kernel_initializer='uniform',  activation = 'relu'))

#adding output layer
classifier.add(Dense(1, kernel_initializer='uniform',  activation = 'sigmoid'))

#compiling ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'] )

#fitting the ANN to the training set
classifier.fit(X_train, Y_train, batch_size = 10, epochs = 100)

#predicting the test set results
y_pred = classifier.predict(X_test)
#will return True if above threshold and otherwise
y_pred = (y_pred > 0.5)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, y_pred)

y_pred_2 = classifier.predict(X_pred)
y_pred_2 = (y_pred_2 > 0.5)
y_pred_2 = y_pred_2.astype(int)

submission = pd.read_csv('gender_submission.csv')

submission['Survived'] = y_pred_2

from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(1000, max_depth=2, random_state=0)
clf.fit(X_train, Y_train)
y_pred = clf.predict(X_test)

y_true = Y_test

from sklearn.metrics import accuracy_score
print(accuracy_score(y_true, y_pred))

y_pred_2 = clf.predict(X_pred)




