# Leslie Poso (400578877)
# Mini Project 1 - Task 1

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# Part 2

categories = categories=['business', 'entertainment', 'politics', 'sport', 'tech']
instances = [510, 386, 417, 511, 401]
	
plt.bar(categories, instances)
plt.savefig('BBC-Distribution.pdf', format='pdf')
plt.show()

# Part 3

corpus = load_files('BBC', categories=categories, encoding='latin1')

# Part 4

vectorizer = CountVectorizer()
corpus_counts = vectorizer.fit_transform(corpus.data)
features = vectorizer.get_feature_names_out()
# Part 5

X_train, X_test, y_train, y_test = train_test_split(corpus_counts, corpus.target, test_size = 0.2, random_state = None)

# Part 6

clf = MultinomialNB()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

# Part 7

# Confusion Matrix:
print(confusion_matrix(y_test, y_pred))

# Classification Report:

print(classification_report(y_test, y_pred, target_names=categories))








