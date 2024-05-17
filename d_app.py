import matplotlib.pyplot as plt
import seaborn as sns

import sqlite3


import sqlite3
import pandas as pd



data = pd.read_csv("titanic3.csv")
conn.close()

print(data['survived'].value_counts())

#Separating the data
a = data[data['survived'] == 0]
b = data[data['survived'] == 1]

A = a.sample(n = 501)

#Concatenating 2 dataframes(legit_sample , fraud)
data = pd.concat([A, b], axis = 0)
data.head()

from sklearn.model_selection import train_test_split
train, test = train_test_split(data, test_size=0.2, random_state=42)

train['sex'] = train['sex'].replace({'male': 0, 'female': 1})

train.drop(['name', 'ticket'], axis=1, inplace=True)

# Identify input and target columns
x = train.drop(['survived'], axis=1)
y = train['survived']

# separate numeric and categorical columns
import numpy as np
numeric = x.select_dtypes(include=np.number).columns.tolist()
categorical = x.select_dtypes('object').columns.tolist()

import pandas as pd
from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
encoder.fit(x[categorical])
encoded_cols = encoder.get_feature_names_out(categorical)
encoded_df = pd.DataFrame(encoder.transform(x[categorical]), columns=encoded_cols, index=x.index)
x = pd.concat([x, encoded_df], axis=1)
x.drop(columns=categorical, inplace=True)

from sklearn import preprocessing
X = preprocessing.normalize(X, norm='l1')

# Splitting the dataset into train and test
from sklearn.model_selection import train_test_split, cross_val_score
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
x_train = scaler.fit_transform(X_train)
x_test = scaler.transform(X_test)

from sklearn.metrics import classification_report, confusion_matrix

# Create the pipeline
results = {}
for name, model in models:
    pipeline = Pipeline([
        ('classifier', model)  # Each model is a step in the pipeline
    ])

    # Fit the model
    pipeline.fit(x_train, y_train)

    # Evaluate the model
    y_pred = pipeline.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)

    results[name] = {
        'accuracy': accuracy,
        'classification_report': classification_report(y_test, y_pred),
        'confusion_matrix': confusion_matrix(y_test, y_pred)
    }

model_accuracies = []

for name, result in results.items():
    model_accuracies.append((name, result['accuracy']))

model_accuracies.sort(key=lambda x: x[1], reverse=True)

top_models = model_accuracies[:3]
for i, (name, accuracy) in enumerate(top_models, start=1):
    print(f"Top {i} Model: {name} ; Accuracy: {accuracy:.2f}")
