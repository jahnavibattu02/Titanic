
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
import streamlit as st

# Load the data
data = pd.read_csv("titanic3.csv")
print(data['survived'].value_counts())

# Separating the data
not_survived = data[data['survived'] == 0]
survived = data[data['survived'] == 1]

sampled_not_survived = not_survived.sample(n=501)
balanced_data = pd.concat([survived, sampled_not_survived])

# Train-test split
train, test = train_test_split(balanced_data, test_size=0.2, random_state=42)

# Data preprocessing
categories = ['cabin', 'embarked', 'boat', 'body', 'home.dest']  # Include 'home.dest'
train.fillna({'home.dest': 'U', 'fare': train['fare'].median(), 'parch': train['parch'].mode()[0], 'sibsp': train['sibsp'].mode()[0]}, inplace=True)  # Handle missing values

# Ensure the columns to drop exist before trying to drop them
cols_to_drop = ['name', 'ticket'] + [col for col in categories if col in train.columns]
train.drop(cols_to_drop, axis=1, inplace=True)

# Encoding categorical data
categorical_cols = train.select_dtypes(include=['object']).columns
train[categorical_cols] = train[categorical_cols].fillna('missing')

encoder = OneHotEncoder(handle_unknown='ignore')
encoded_features = encoder.fit_transform(train[categorical_cols]).toarray()
encoded_cols = encoder.get_feature_names_out(categorical_cols)

# Creating a DataFrame with encoded features
encoded_df = pd.DataFrame(encoded_features, columns=encoded_cols, index=train.index)
train_final = pd.concat([train.drop(categorical_cols, axis=1), encoded_df], axis=1).dropna()

# Model training
X_train = train_final.drop('survived', axis=1)
y_train = train_final['survived']
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)

# RandomForest Classifier
random_forest_model = RandomForestClassifier(n_estimators=100, random_state=42)
random_forest_model.fit(X_train_scaled, y_train)
accuracy = random_forest_model.score(X_train_scaled, y_train)
print(f'Training Accuracy: {accuracy:.2f}')

# Streamlit app to display predictions
def main():
    st.title('Titanic Survival Prediction')
    pclass = st.selectbox('Passenger Class', options=[1, 2, 3])
    sex = st.selectbox('Sex', options=['male', 'female'])
    age = st.number_input('Age', min_value=1, max_value=100, value=28)
    fare = st.number_input('Fare', value=35.0)
    parch = st.number_input('Parch', value=0)
    sibsp = st.number_input('Siblings/Spouses Aboard', value=0)

    if st.button('Predict'):
        input_data = pd.DataFrame({
            'pclass': [pclass],
            'sex': [sex],
            'age': [age],
            'fare': [fare],
            'parch': [parch],
            'sibsp': [sibsp]
        })
        input_data['sex'] = input_data['sex'].map({'male': 0, 'female': 1})
        input_encoded = encoder.transform(input_data[['sex']])
        input_encoded_df = pd.DataFrame(input_encoded.toarray(), columns=encoder.get_feature_names(['sex']))
        input_final = pd.concat([input_data.drop(['sex'], axis=1), input_encoded_df], axis=1)
        input_scaled = scaler.transform(input_final)
        
        prediction = random_forest_model.predict(input_scaled)[0]
        probability = random_forest_model.predict_proba(input_scaled)[0][1]
        
        st.write(f'Predicted Survival: {"Yes" if prediction == 1 else "No"}')
        st.write(f'Survival Probability: {probability:.2%}')

if __name__ == '__main__':
    main()
