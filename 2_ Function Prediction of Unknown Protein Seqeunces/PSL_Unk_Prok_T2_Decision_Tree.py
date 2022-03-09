# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 19:26:42 2020

@author: Indian
"""
import pandas as pd 
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score 
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from collections import Counter
from imblearn.over_sampling import SMOTE 


data=pd.read_csv("Gram_Negative_Final_5.csv")
X=data.drop(['FN','Class'],axis=1)
y=data['Class']

lda = LinearDiscriminantAnalysis(n_components=7)
Z = lda.fit(X, y).transform(X)


#pca = PCA(n_components=10)
#Z = pca.fit_transform(X)

x_train,x_test,y_train,y_test=train_test_split(Z,y,random_state=0)
dtc=tree.DecisionTreeClassifier()
model=dtc.fit(x_train,y_train)
y_pred = dtc.fit(Z, y).predict(Z)
dtc.score(x_test,y_test)
feature_names=np.array(X.columns)
target_names=['1','2','3','4','5','6','7','8']

sm = SMOTE(random_state=42)
Z_res, Y_res = sm.fit_resample(Z, y)
print('Resampled dataset shape %s' % Counter(Y_res))


validation_size = 0.1
Z_train, Z_validation, y_train, y_validation = train_test_split(Z, y, test_size=validation_size)                    ## shuffle and split training and test sets
scoring = 'accuracy'
dtc.score(x_test,y_test)
predictions = dtc.predict(Z_validation)
print(accuracy_score(y_validation, predictions))
print(confusion_matrix(y_validation, predictions))
print(classification_report(y_validation, predictions))
print("PRECISION SCORE::",precision_score(y_validation,predictions, average="macro"))
print("RECALL SCORE::",recall_score(y_validation,predictions, average="macro")) 
print("F1_SCORE::",f1_score(y_validation,predictions, average="weighted"))
print("Accuracy::",accuracy_score(y_validation,predictions))

test = pd.read_csv("Prokar_Test_2_Ind_5_105.csv")

ZN = lda.transform(test)

pred = dtc.predict(ZN)

pred1 = dtc.predict_proba(ZN)