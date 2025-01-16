#!/usr/bin/python3

"""
This is the code to accompany the Lesson 1 (Naive Bayes) mini-project.

Use a Naive Bayes Classifier to identify emails by their authors.

authors and labels:
- Sara has label 0
- Chris has label 1
"""

from time import time
from email_preprocess import preprocess
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score



# Preprocess the data
try:
    features_train, features_test, labels_train, labels_test = preprocess()
except Exception as e:
    print(f"Error during preprocessing: {e}")
    exit(1)

# Train a Naive Bayes classifier
clf = GaussianNB()

# Measure training time
t0 = time()
clf.fit(features_train, labels_train)
print("Training Time:", round(time() - t0, 3), "s")

# Measure prediction time
t0 = time()
predictions = clf.predict(features_test)
print("Predicting Time:", round(time() - t0, 3), "s")

# Calculate accuracy
accuracy = accuracy_score(labels_test, predictions)
print("Accuracy:", accuracy)
