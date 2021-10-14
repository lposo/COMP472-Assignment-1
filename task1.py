# Leslie Poso (400578877)
# Mini Project 1 - Task 1

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_files

# Part 2

categories = categories=['business', 'entertainment', 'politics', 'sport', 'tech']
instances = [510, 386, 417, 511, 401]
	
plt.bar(categories, instances)
plt.show()
plt.savefig()

# Part 3

corpus = load_files('BBC', categories=categories, encoding='latin1')

# Part 4

vectorizer.fit_transform(corpus)