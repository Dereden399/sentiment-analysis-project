import pandas as pd
import numpy as np

# from transformers import BertModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm

# Load data
train_df = pd.read_json('data/train.json')
print(train_df.shape)
print(train_df.head())

vectorizer = TfidfVectorizer()
vectorizer = vectorizer.fit(train_df['reviews'])
X_train = vectorizer.transform(train_df['reviews'])
print(X_train.shape)

test_df = pd.read_json('data/test.json')
print(test_df.shape)
X_test = vectorizer.transform(test_df['reviews'])
print(X_test.shape)