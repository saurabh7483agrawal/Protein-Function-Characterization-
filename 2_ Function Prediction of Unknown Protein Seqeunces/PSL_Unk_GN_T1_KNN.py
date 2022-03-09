# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 12:17:06 2020

@author: Indian
"""
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from collections import Counter
from imblearn.over_sampling import SMOTE 



# dataload
data = pd.read_csv("Gram_Negative_Final_5.csv") 
X=data.iloc[:,1:175]
Y=data.iloc[:,175]
plt.figure()

lda = LinearDiscriminantAnalysis(n_components=7)
Z = lda.fit(X, Y).transform(X)

sm = SMOTE(random_state=42)
Z_res, Y_res = sm.fit_resample(Z, Y)
print('Resampled dataset shape %s' % Counter(Y_res))


validation_size = 0.10
seed = 7
Z_res_train, Z_res_validation, Y_res_train, Y_res_validation = train_test_split(Z_res, Y_res, test_size=validation_size, random_state=seed)                    ## shuffle and split training and test sets
scoring = 'accuracy'

knn=KNeighborsClassifier()
knn.fit(Z_res_train, Y_res_train)
predictions = knn.predict(Z_res_validation)
print(accuracy_score(Y_res_validation, predictions))
print(confusion_matrix(Y_res_validation, predictions))
print(classification_report(Y_res_validation, predictions))

test = pd.read_csv("GN_Test_1_Ind_5_156.csv")

ZN = lda.transform(test)

pred = knn.predict(ZN)

pred1 = knn.predict_proba(ZN)