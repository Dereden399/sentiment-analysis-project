import pandas as pd
import numpy as np
import torch

from transformers import BertModel, BertTokenizer, DistilBertTokenizer, DistilBertModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn import svm, tree
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

# Load data
train_df = pd.read_json('data/train.json')

# Split data
train_df, val_df = train_test_split(train_df, test_size=0.2, shuffle=True)


# Vectorize data using TF-IDF
vectorizer = TfidfVectorizer()
vectorizer = vectorizer.fit(train_df['reviews'])
X_train = vectorizer.transform(train_df['reviews'])
print("Train data shape:", X_train.shape)

X_val = vectorizer.transform(val_df['reviews'])
print("Validation data shape: ", X_val.shape)

test_df = pd.read_json('data/test.json')
X_test = vectorizer.transform(test_df['reviews'])
print("Test data shape: ", X_test.shape)


'''# Vectorize data using BERT

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertModel.from_pretrained('distilbert-base-uncased')
model.eval()
def get_bert_embeddings(text):
  # Tokenize and encode input text
  inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
  
  # Get BERT embeddings (we use the output of the last hidden state)
  with torch.no_grad():
    outputs = model(**inputs)
  
  # We can use the [CLS] token embedding for classification tasks (first token output)
  cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
  return cls_embedding

print("Start vectorizing data using BERT...")

X_train = np.array([get_bert_embeddings(text) for text in train_df['reviews']])
print("Train data shape:", X_train.shape)

X_val = np.array([get_bert_embeddings(text) for text in val_df['reviews']])
print("Validation data shape: ", X_val.shape)

test_df = pd.read_json('data/test.json')
X_test = np.array([get_bert_embeddings(text) for text in test_df['reviews']])
print("Test data shape: ", X_test.shape)

print("Data vectorization completed!")
'''

'''# Train model using SVM
clf = svm.SVC(C=1.0, kernel='linear', class_weight='balanced')
clf.fit(X_train, train_df['sentiments'])

# Predict
train_preds = clf.predict(X_val)'''


'''# Train model using Decision Tree
clf = tree.DecisionTreeClassifier()
clf.fit(X_train, train_df['sentiments'])

# Predict
train_preds = clf.predict(X_val)'''

'''# Train model using Logistic Regression
clf = LogisticRegression()
clf.fit(X_train, train_df['sentiments'])

# Predict
train_preds = clf.predict(X_val)'''

# Train model using MLP
clf = MLPClassifier(hidden_layer_sizes=(100,100), max_iter=1000)
clf.fit(X_train, train_df['sentiments'])

# Predict
train_preds = clf.predict(X_val)


# Evaluate
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(val_df['sentiments'], train_preds)
print("Total accuracy: ", accuracy)

# Show confusion matrix
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(val_df['sentiments'], train_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
disp.plot()
disp.ax_.set_title(f"Accuracy: {accuracy*100:.2f}")
plt.show()
