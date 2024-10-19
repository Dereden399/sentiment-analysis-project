import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import re
import nltk
from nltk.corpus import stopwords

from classes import TfidfTransformer, BertTransformer, Transformer, SVMModel, Model, MLPModel, LogisticRegressionModel, DecisionTreeModel

LOAD_PICKLED = True
TRANSFORMER_TYPE = "Tfidf"
MODEL_TYPE = "MLP"
VALIDATION_SIZE = 0.15
RANDOM_STATE = 100
ITERATIONS = 3

transformerMappings = {
  "Tfidf": TfidfTransformer,
  "Bert": BertTransformer
}
modelMappings = {
  "SVM": SVMModel,
  "DecisionTree": DecisionTreeModel,
  "LogisticRegression": LogisticRegressionModel,
  "MLP": MLPModel
}

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
def preprocess_text(text):
  # Remove non-alphabetic characters and convert to lowercase
  text = re.sub(r'[^a-zA-Z\s]', '', text)
  text = text.lower()
  
  # Tokenize and remove stopwords
  tokens = text.split()
  tokens = [word for word in tokens if word not in stop_words]
  return ' '.join(tokens)

# Load data
train_df = pd.read_json('data/train.json')
test_df = pd.read_json('data/test.json')

# Preprocess data
train_df['reviews'] = train_df['reviews'].apply(preprocess_text)

if TRANSFORMER_TYPE not in transformerMappings:
  raise ValueError(f"Invalid transformer type: {TRANSFORMER_TYPE}")
transformer: Transformer = transformerMappings[TRANSFORMER_TYPE](train_df['reviews'])
if MODEL_TYPE not in modelMappings:
  raise ValueError(f"Invalid model type: {MODEL_TYPE}")
model: Model = modelMappings[MODEL_TYPE](RANDOM_STATE)

if LOAD_PICKLED:
  data = np.load('data/pickled.npz', allow_pickle=True)
  X_df = data['train']
  X_test = data['test']
else:
  X_df = transformer.transform(train_df['reviews'])
  X_test = transformer.transform(test_df['reviews'])
  
  np.savez_compressed('data/pickled', train=X_df.toarray(), test=X_test.toarray())

# Split data
X_train, X_val, y_train, y_val = train_test_split(X_df, train_df['sentiments'], test_size=VALIDATION_SIZE, random_state=RANDOM_STATE)
print("All data shape:", X_df.shape)
print("Train data shape:", X_train.shape)
print("Validation data shape: ", X_val.shape)
print("Test data shape: ", X_test.shape)


# Train model
model.fit(X_train, y_train)


# Predict
avgAccuracy = 0
for i in range(ITERATIONS):
  train_preds = model.predict(X_val)
  avgAccuracy += accuracy_score(y_val, train_preds)
  print(f"Iteration {i+1} complete")
avgAccuracy /= ITERATIONS

# Evaluate
accuracy = accuracy_score(y_val, train_preds)
print("Total accuracy: ", accuracy)

# Show confusion matrix
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_val, train_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.model.classes_)
disp.plot()
disp.ax_.set_title(f"Last iteration of {model.name()}\nAverage accuracy: {avgAccuracy*100:.2f}%")
plt.show()
