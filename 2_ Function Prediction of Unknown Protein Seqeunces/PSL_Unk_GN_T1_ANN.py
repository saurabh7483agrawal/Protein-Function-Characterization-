# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 11:53:34 2020

@author: Indian
"""
import pandas as pd 
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from collections import Counter
from imblearn.over_sampling import SMOTE 
from sklearn.neural_network import MLPClassifier


# dataload
data = pd.read_csv("Gram_Negative_Final_5.csv") 
X=data.iloc[:,1:175]
Y=data.iloc[:,175]


lda = LinearDiscriminantAnalysis(n_components=7)
Z = lda.fit(X, Y).transform(X)
print('Original dataset shape %s' % Counter(Y))

sm = SMOTE(random_state=48)
Z_res, Y_res = sm.fit_resample(Z, Y)
print('Resampled dataset shape %s' % Counter(Y_res))


validation_size = 0.10
seed = 2
Z_res_train, Z_res_validation, Y_res_train, Y_res_validation = train_test_split(Z_res, Y_res, test_size=validation_size, random_state=seed)                    ## shuffle and split training and test sets
scoring = 'accuracy'

clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                    hidden_layer_sizes=(50,), random_state=1)
clf.fit(Z_res_train, Y_res_train)

#clf.predict(Z_res_train)
#clf.predict([[0., 0.]])



predictions = clf.predict(Z_res_validation)
predictions1 = clf.predict_proba(Z_res_validation)
y_score = clf.fit(Z_res_train, Y_res_train)

#.decision_function(Z_res_validation)
print(accuracy_score(Y_res_validation, predictions))
print(confusion_matrix(Y_res_validation, predictions))
print(classification_report(Y_res_validation, predictions))

test = pd.read_csv("GN_Test_1_Ind_5_156.csv")

ZN = lda.transform(test)

pred = clf.predict(ZN)

pred1 = clf.predict_proba(ZN)