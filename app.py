import pandas as pd
import numpy as np
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
balanced_data = pd.concat([survived, sampled_not_survived], ignore_index=True)  # Reset index

# Train-test split
train, test = train_test_split(balanced_data, test_size=0.2, random_state=42)

columns_to_drop = ['name', 'ticket', 'body', 'boat','home.dest']
train = train.drop(columns=[col for col in columns_to_drop if col in train.columns])

# Data preprocessing
train['sex'] = train['sex'].replace({'male': 0, 'female': 1})
x = train.drop(['survived'], axis=1)
y = train['survived']

numeric = x.select_dtypes(include=np.number).columns.tolist()
categorical = x.select_dtypes('object').columns.tolist()

x['age'].fillna(x['age'].mean(), inplace=True)
x['fare'].fillna(x['fare'].mean(), inplace=True)
x.fillna({'cabin':'U', 'embarked': 'x'}, inplace=True)
print(x[numeric].isna().sum())
print(x[categorical].isna().sum())

encoder = OneHotEncoder(handle_unknown='ignore')
encoder.fit(x[categorical])
encoded_cols = encoder.get_feature_names_out(categorical)
encoded_df = pd.DataFrame(encoder.transform(x[categorical]).toarray(), columns=encoded_cols, index=x.index)

# Concatenating the DataFrames
train_final = pd.concat([x.drop(categorical, axis=1), encoded_df], axis=1)

# Model training
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(train_final)

random_forest_model = RandomForestClassifier(n_estimators=100, random_state=42)
random_forest_model.fit(X_train_scaled, y)

accuracy = random_forest_model.score(X_train_scaled, y)
print(f'Training Accuracy: {accuracy:.2f}')

# Streamlit app to display predictions
def main():
    st.title('Titanic Survival Prediction')

    st.write("GROUP-15: Jahnavi Pravaleeka Battu, Devi Deepshikha Jami, Rithwik Gilla, Sri naga chanran Gampala")
    
    st.header('Introduction')
    st.write('This application predicts the survival likelihood of passengers aboard the Titanic based on user-provided data. Please enter the details below.')
    
    
    # Collecting input
    pclass = st.selectbox('Passenger Class', options=[1, 2, 3])
    sex = st.selectbox('Sex', options=['male', 'female'])
    age = st.number_input('Age', min_value=1, max_value=100, value=28)
    fare = st.number_input('Fare', value=35.0)
    parch = st.number_input('Parch', value=0)
    sibsp = st.number_input('Siblings/Spouses Aboard', value=0)
    cabin = st.text_input('Cabin', value='U')  # Assuming default 'U' for unknown
    embarked = st.selectbox('Embarked', options=['C', 'Q', 'S', 'x'])  # Assuming 'x' as unknown

    if st.button('Predict'):
        # Create input DataFrame with the same structure as the training data
        input_data = pd.DataFrame({
            'pclass': [pclass],
            'sex': [sex],
            'age': [age],
            'fare': [fare],
            'parch': [parch],
            'sibsp': [sibsp],
            'cabin': [cabin],
            'embarked': [embarked]
        })

        # Preprocessing
        input_data['sex'] = input_data['sex'].map({'male': 0, 'female': 1})
        input_data.fillna({'age': input_data['age'].mean(), 'fare': input_data['fare'].mean(), 'cabin': 'U', 'embarked': 'x'}, inplace=True)
        
        # Encode categorical variables
        input_data_encoded = encoder.transform(input_data[['cabin', 'embarked']]).toarray()
        encoded_cols = encoder.get_feature_names_out(['cabin', 'embarked'])
        input_data_encoded_df = pd.DataFrame(input_data_encoded, columns=encoded_cols, index=input_data.index)
        
        # Combine numerical and encoded categorical data
        input_data_numeric = input_data.drop(['cabin', 'embarked'], axis=1)
        input_final = pd.concat([input_data_numeric, input_data_encoded_df], axis=1)
        
        # Ensure the column order matches the training data
        input_final = input_final[train_final.columns]

        # Scaling
        input_final_scaled = scaler.transform(input_final)

        # Prediction
        prediction = random_forest_model.predict(input_final_scaled)[0]
        probability = random_forest_model.predict_proba(input_final_scaled)[0][1]
        
        # Display results
        st.write(f'Predicted Survival: {"Yes" if prediction == 1 else "No"}')

if __name__ == '__main__':
    main()
