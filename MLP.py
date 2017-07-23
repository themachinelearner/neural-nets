# -*- coding: utf-8 -*-
"""
Created on Sun Jul 23 07:46:23 2017

@author: wired_000
"""

from sklearn.neural_network import MLPClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

X = [[0., 0.], [1., 1.],[2., 2.]]
Y = [0, 1, 2]
clf = MLPClassifier(solver='lbfgs', hidden_layer_sizes=(40,40,40))

clf.fit(X, Y)
print (clf.predict([[1.8,1.6]]))
print ([coef.shape for coef in clf.coefs_])
print ("Iris set")

iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=.4)

clf.fit(X_train, y_train)
print (clf.score(X_test, y_test))