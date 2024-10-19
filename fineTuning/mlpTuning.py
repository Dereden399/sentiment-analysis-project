import numpy as np
import pandas as pd

from classes import MLPModel
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from joblib import Parallel, delayed

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

firstLayerRange = range(90, 101)
secondLayerRange = range(90, 101)

# Load pickled vectors
data = np.load('data/pickled.npz', allow_pickle=True)
X_df = data['train']
train_df = pd.read_json('data/train.json')
y_df = train_df['sentiments']


X_train, X_val, y_train, y_val = train_test_split(X_df, y_df, test_size=0.3, random_state=404)

# Using a subset for faster prototyping
X_train, _, y_train, _ = train_test_split(X_train, y_train, test_size=0.6, random_state=404)

bestAccuracy = 0
bestFirstLayer = 0
bestSecondLayer = 0

firstLayerSizes = []
secondLayerSizes = []
accuracies = []

def train_model(firstLayer, secondLayer, X_train, y_train, X_val, y_val):
  model = MLPClassifier(hidden_layer_sizes=(firstLayer, secondLayer), max_iter=300, solver='adam')
  model.fit(X_train, y_train)
  accuracy = accuracy_score(y_val, model.predict(X_val))
  print(f"First layer: {firstLayer}, Second layer: {secondLayer}, Accuracy: {accuracy*100:.2f}%")
  return (firstLayer, secondLayer, accuracy)

results = Parallel(n_jobs=-1)(delayed(train_model)(firstLayer, secondLayer, X_train, y_train, X_val, y_val)
                              for firstLayer in firstLayerRange
                              for secondLayer in secondLayerRange)

for firstLayer, secondLayer, accuracy in results:
  firstLayerSizes.append(firstLayer)
  secondLayerSizes.append(secondLayer)
  accuracies.append(accuracy)
  if accuracy > bestAccuracy:
    bestAccuracy = accuracy
    bestFirstLayer = firstLayer
    bestSecondLayer = secondLayer
  
print(f"Best accuracy: {bestAccuracy*100:.2f}%")
print(f"Best first layer size: {bestFirstLayer}")
print(f"Best second layer size: {bestSecondLayer}")

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Convert lists to NumPy arrays
x = np.array(firstLayerSizes)
y = np.array(secondLayerSizes)
z = np.array(accuracies)

# Create a scatter plot
ax.scatter(x, y, z, c=z, cmap='viridis', marker='o')

# Set labels
ax.set_xlabel('First Layer Size')
ax.set_ylabel('Second Layer Size')
ax.set_zlabel('Accuracy (%)')
ax.set_title('Accuracy for Different Hidden Layer Configurations')

# Show the plot
plt.show()