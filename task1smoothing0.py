# Leslie Poso (400578877)
# Mini Project 1 - Task 1

import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

# Part 2

categories = categories=['business', 'entertainment', 'politics', 'sport', 'tech']
instances = [510, 386, 417, 511, 401]
	
#plt.bar(categories, instances)
#plt.savefig('BBC-Distribution.pdf', format='pdf')
#plt.show()

# Part 3

corpus = load_files('BBC', categories=categories, encoding='latin1')

# Part 4

vectorizer = CountVectorizer()
corpus_counts = vectorizer.fit_transform(corpus.data)
features = vectorizer.get_feature_names_out()
# Part 5

X_train, X_test, y_train, y_test = train_test_split(corpus_counts, corpus.target, test_size = 0.2, random_state = None)

# Part 6

clf = MultinomialNB(alpha = 0.9)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# Part 7

# Confusion Matrix:
print(confusion_matrix(y_test, y_pred))

# Classification Report:

print(classification_report(y_test, y_pred, target_names=categories))

# Accuracy Score, F1 Score:

print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

print("Macro F1:", metrics.f1_score(y_test, y_pred, average='macro'))

print("Weighted F1:", metrics.f1_score(y_test, y_pred, average='weighted'))

# Priors:

print(clf.class_log_prior_)

# Size of Vocabulary):

print(clf.n_features_in_)

# Word-Tokens Per Class:
for i in range(5):
	count = 0
	for j in clf.feature_count_[i]:
		count += int(clf.feature_count_[i,int(j)])
	print("Total Count for %s is %s" % (i, count))

# Total Word Tokens for Corpus: 
print("Total Word Count:", sum(vectorizer.vocabulary_.values()))

# Words with Frequency of 0 in Class: 

for i in range(5):
	count = 0
	for j in clf.feature_count_[i]:
		if (j == 0):
			count = count + 1
	percent = count / clf.n_features_in_
	print("Total Words With Zero Frequency for %s is %s" % (i, count))
	print("Percentage of Words With Zero Frequency for %s is %s" % (i, percent))

# Words with Frequency of 1 in Corpus: 
count = 0
for i in range(5):
	for j in clf.feature_count_[i]:
		if (j == 1):
			count = count + 1
percent = count / clf.n_features_in_

# Favorite Words: (Get Indices)

print("Total Words With One Frequency:", count)
print("Percentage of Words With One Frequency:", percent)

index_of_the = np.where(features == "the")
index_of_start = np.where(features == "start")
print(index_of_the) #26462
print(index_of_start) #28139

count = 0
for i in range(5):
	count = count + clf.feature_count_[i, 26462]
print("Total Frequency of THe:", count)
log = math.log(count / sum(vectorizer.vocabulary_.values()), 10)
print("Log Probability of THe:", log)

count = 0
for i in range(5):
	count = count + clf.feature_count_[i, 28139]
print("Total Frequency of Start:", count)
log = math.log(count / sum(vectorizer.vocabulary_.values()), 10)
print("Log Probability of Start:", log)