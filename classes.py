from abc import ABC, abstractmethod
from typing import Iterable
from numpy import ndarray as NDArray
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm, tree
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
import torch
from transformers import DistilBertTokenizer, DistilBertModel
import numpy as np

class Transformer(ABC):
  def __init__(self, dataToFit: Iterable) -> None:
    pass

  @abstractmethod
  def transform(self, dataToTransform: Iterable) -> NDArray:
    pass

class TfidfTransformer(Transformer):
  def __init__(self, dataToFit: Iterable) -> None:
    self.vectorizer = TfidfVectorizer()
    self.vectorizer = self.vectorizer.fit(dataToFit)
    print("Created TfidfVectorizer")

  def transform(self, dataToTransform: Iterable) -> NDArray:
    return self.vectorizer.transform(dataToTransform)

class BertTransformer(Transformer):
  def __init__(self, dataToFit: Iterable) -> None:
    self.model = DistilBertModel.from_pretrained('distilbert-base-uncased')
    self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    self.model.eval()
    print("Created DistilBertTransformer")
  
  def transform(self, dataToTransform: Iterable) -> NDArray:
    return np.array([self.getEmbedding(text) for text in dataToTransform])

  def getEmbedding(self, text: str) -> NDArray:
    inputs = self.tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    with torch.no_grad():
      outputs = self.model(**inputs)
    return outputs.last_hidden_state[:, 0, :].squeeze().numpy()

class Model(ABC):
  def __init__(self, randomState: int) -> None:
    pass

  @property
  @abstractmethod
  def name(self) -> str:
    pass

  @abstractmethod
  def fit(self, X: NDArray, y: NDArray) -> None:
    pass

  @abstractmethod
  def predict(self, X: NDArray) -> NDArray:
    pass

class SVMModel(Model):
  def __init__(self, randomState: int) -> None:
    self.model = svm.SVC(C=1.0, kernel='linear', class_weight='balanced', random_state=randomState)
    print("Created SVMModel")

  def fit(self, X: NDArray, y: NDArray) -> None:
    self.model.fit(X, y)

  def predict(self, X: NDArray) -> NDArray:
    return self.model.predict(X)
  
  def name(self) -> str:
    return "SVM Model"
  
class DecisionTreeModel(Model):
  def __init__(self, randomState: int) -> None:
    self.model = tree.DecisionTreeClassifier(random_state=randomState)
    print("Created DecisionTreeModel")

  def fit(self, X: NDArray, y: NDArray) -> None:
    self.model.fit(X, y)

  def predict(self, X: NDArray) -> NDArray:
    return self.model.predict(X)
  
  def name(self) -> str:
    return "Decision Tree Model"

class LogisticRegressionModel(Model):
  def __init__(self, randomState: int) -> None:
    self.model = LogisticRegression(random_state=randomState)
    print("Created LogisticRegressionModel")

  def fit(self, X: NDArray, y: NDArray) -> None:
    self.model.fit(X, y)

  def predict(self, X: NDArray) -> NDArray:
    return self.model.predict(X)
  
  def name(self) -> str:
    return "Logistic Regression Model"
  
class MLPModel(Model):
  def __init__(self, randomState: int) -> None:
    self.model = MLPClassifier(hidden_layer_sizes=(99,94), max_iter=500, random_state=randomState)
    print("Created MLPModel")

  def fit(self, X: NDArray, y: NDArray) -> None:
    self.model.fit(X, y)

  def predict(self, X: NDArray) -> NDArray:
    return self.model.predict(X)
  
  def name(self) -> str:
    return "MLP Model"