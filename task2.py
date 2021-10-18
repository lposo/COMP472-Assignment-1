# Leslie Poso (400578877)
# Mini Project 1 - Task 1

import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

# Part 2

dataset = pd.read_csv('drug200.csv')
print(dataset)

# Part 3

categories = categories=['DrugA', 'DrugB', 'DrugC', 'DrugX', 'DrugY']
instances = [23, 16, 16	, 64, 91]
	
#plt.bar(categories, instances)
#plt.savefig('drug-distribution.pdf', format='pdf')
#plt.show()

# Part 4

dataset = pd.get_dummies(dataset, columns=['Sex'])
dataset["BP"].replace({'LOW': 0, 'NORMAL': 1, 'HIGH': 2}, inplace = True)
dataset["Cholesterol"].replace({'NORMAL': 1, 'HIGH': 2}, inplace = True)

print(dataset)

# Part 5
X_train, X_test, y_train, y_test = train_test_split(dataset[["Age", "BP", "Cholesterol", "Na_to_K", "Sex_F", "Sex_M"]], dataset["Drug"])

# Part 6

# GaussianNB:

print("Gaussian NB")

clf = GaussianNB()

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# Confusion Matrix:
print(confusion_matrix(y_test, y_pred))

# Classification Report:

print(classification_report(y_test, y_pred, target_names=categories))

# Accuracy Score, F1 Score:

print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

print("Macro F1:", metrics.f1_score(y_test, y_pred, average='macro'))

print("Weighted F1:", metrics.f1_score(y_test, y_pred, average='weighted'))
