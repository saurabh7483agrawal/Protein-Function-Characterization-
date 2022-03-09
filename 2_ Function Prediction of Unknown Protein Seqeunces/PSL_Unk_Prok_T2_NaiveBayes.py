# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 19:46:43 2020

@author: Indian
"""
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from collections import Counter
from imblearn.over_sampling import SMOTE 

data=pd.read_csv("Gram_Negative_Final_5.csv")
X=data.drop(['FN', 'Class'],axis=1)
y=data['Class']

lda = LinearDiscriminantAnalysis(n_components=7)
Z = lda.fit(X, y).transform(X)


x_train,x_test,y_train,y_test=train_test_split(Z,y,random_state=0)
gnb = GaussianNB()
model=gnb.fit(x_train,y_train)
y_pred = gnb.fit(Z, y).predict(Z)
gnb.score(x_test,y_test)
print("Number of mislabled points out of a total %d points : %d" %(X.shape[0], (y != y_pred).sum()))
feature_names=np.array(X.columns)
target_names=['1', '2', '3', '4', '5', '6', '7', '8']
#print(target_names)
#print(feature_names)

sm = SMOTE(random_state=50)
Z_res, Y_res = sm.fit_resample(Z, y)
print('Resampled dataset shape %s' % Counter(Y_res))


validation_size = 0.10
Z_train, Z_validation, y_train, y_validation = train_test_split(Z, y, test_size=validation_size)                    ## shuffle and split training and test sets
scoring = 'accuracy'
gnb.score(x_test,y_test)
predictions = gnb.predict(Z_validation)
predictions1 = gnb.predict_proba(Z_validation)
print('Accuracy::', accuracy_score(y_validation, predictions))
print('Confusion Matrix::', confusion_matrix(y_validation, predictions))
print('Classification Report::', classification_report(y_validation, predictions))

test = pd.read_csv("Prokar_Test_2_Ind_5_105.csv")

ZN = lda.transform(test)

pred = gnb.predict(ZN)

pred1 = gnb.predict_proba(ZN)