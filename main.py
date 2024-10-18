import pandas as pd
import numpy as np
import torch

from transformers import BertModel, BertTokenizer, DistilBertTokenizer, DistilBertModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn import svm, tree
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

LOAD_FROM_FILES = True
# Load data
train_df = pd.read_json('data/train.json')

if LOAD_FROM_FILES:
  X_df = np.load('data/train.npy', allow_pickle=True)
  print("All data shape:", X_df.shape)

  X_test = np.load('data/test.npy', allow_pickle=True)
else:
  '''# Vectorize data using TF-IDF
  vectorizer = TfidfVectorizer()
  vectorizer = vectorizer.fit(train_df['reviews'])
  X_df = vectorizer.transform(train_df['reviews'])
  print("All data shape:", X_df.shape)
  # Save to file
  np.save('data/train.npy', X_df.toarray())

  test_df = pd.read_json('data/test.json')
  X_test = vectorizer.transform(test_df['reviews'])
  # Save to file
  np.save('data/test.npy', X_test.toarray())'''
  # Vectorize data using BERT

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

  X_df = np.array([get_bert_embeddings(text) for text in train_df['reviews']])
  print("Train vectorized")

  # Save to file
  np.save('data/train.npy', X_df)

  test_df = pd.read_json('data/test.json')
  X_test = np.array([get_bert_embeddings(text) for text in test_df['reviews']])
  print("Test vectorized")
  # Save to file
  np.save('data/test.npy', X_test)

  print("Data vectorization completed!")

# Split data
X_train, X_val, y_train, y_val = train_test_split(X_df, train_df['sentiments'], test_size=0.3)
print("Train data shape:", X_train.shape)
print("Validation data shape: ", X_val.shape)
print("Test data shape: ", X_test.shape)


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
clf.fit(X_train, y_train)

# Predict
train_preds = clf.predict(X_val)


# Evaluate
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_val, train_preds)
print("Total accuracy: ", accuracy)

# Show confusion matrix
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_val, train_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
disp.plot()
disp.ax_.set_title(f"Accuracy: {accuracy*100:.2f}")
plt.show()
