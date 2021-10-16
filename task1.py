# Leslie Poso (400578877)
# Mini Project 1 - Task 1

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics

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
tf_transformer = TfidfTransformer(use_idf=False).fit(corpus_counts)
corpus_counts_tf = tf_transformer.transform(corpus_counts)

# Part 5

X_train, X_test, y_train, y_test = train_test_split(corpus_counts, corpus.target, test_size = 0.2, random_state = None)

# Part 6

clf = MultinomialNB()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

# Part 7

#a) a clear separator (a sequence of hyphens or stars) and string clearly describing the model (e.g. “MultinomialNB default values, try 1”)








