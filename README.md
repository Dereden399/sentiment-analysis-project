# Sentiment Analysis Project

![Scikit-learn](https://img.shields.io/badge/scikit--learn-1.5.2-orange)

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation and Running](#installation-and-running)
- [Important files](#important-files)

## Overview

This project is a sentiment analysis tool developed for the Nanyang Technological University's IE4483 Artificial Intelligence and Data Mining course. The objective is to classify reviews into positive and negative sentiments using various machine learning techniques. The project explores different feature extraction methods and model configurations to optimize classification performance.

## Features

- **Feature Extraction:**
  - **TF-IDF Vectorization:** Converts text data into numerical features based on term frequency and inverse document frequency.
  - **DistilBERT Embeddings:** Utilizes the DistilBERT tokenizer from Hugging Face for contextualized word embeddings.
  
- **Model Selection:**
  - **Multilayer Perceptron (MLP) Classifier:** Achieves an average accuracy of 91.54% with a carefully tuned architecture.

- **Performance Analysis:**
  - Comparative analysis of TF-IDF and DistilBERT features.
  - Detailed evaluation of model performance including confusion matrices.

## Installation and Running

1. **Clone the Repository**
2. **Create the Virtual Environment (Optional)**
   ```bash
     python3 -m venv venv
     source venv/bin/activate
   ```
3. **Install Dependencies**
   ```bash
     pip install -r requirements.txt
   ```
4. **Running**
   ```bash
      python main.py
   ```

## Important files
- **`data/`**
  - **`submission.csv`**: Submission file. The model's output for test.json
  - **`test.json`**: Contains the testing dataset in JSON format
  - **`train.json`**: Contains the training dataset in JSON format
- **`fineTuning/`**
  - **`mlpTuning.py`**: A Python script to fine-tune the hyperparameters of the MLP model.
- **`main.py`**: Entry file. Contains project settings
