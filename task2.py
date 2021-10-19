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
	
# Base-DT

print("Base-DT")

clf = DecisionTreeClassifier()
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

#Top-DT, max_depth = 10, min samples split = 3:

print("Top-DT, max_depth = 10, min samples split = 3")

clf = GridSearchCV(DecisionTreeClassifier(), param_grid=[{'criterion':['gini'], 'max_depth': [10], 'min_samples_split': [3]}])
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

# Top-DT max_depth = 8, min samples split = 5:

print("Top-DT, max_depth = 8, min samples split = 5")

clf = GridSearchCV(DecisionTreeClassifier(), param_grid=[{'criterion':['gini'], 'max_depth': [8], 'min_samples_split': [5]}])
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

# Top-DT max_depth = 15, min samples split = 2:

print("Top-DT, max_depth = 15, min samples split = 2")

clf = GridSearchCV(DecisionTreeClassifier(), param_grid=[{'criterion':['gini'], 'max_depth': [15], 'min_samples_split': [2]}])
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

# Perceptron

print("Perceptron")

clf = Perceptron()
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

# Base-MLP

print("Base-MLP")

clf = MLPClassifier(hidden_layer_sizes=(100,), activation='logistic', solver='sgd')
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

# Top-MLP

print("Top-MLP - Hidden Layer Sizes: 30,50 - Activation: Tanh - Solver: Adam")

clf = GridSearchCV(MLPClassifier(), param_grid=[{'hidden_layer_sizes':[30,50], 'activation': ['tanh'], 'solver': ['adam']}])
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