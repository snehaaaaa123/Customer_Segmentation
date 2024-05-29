import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the trained model
with open('random_forest_classifier.pkl', 'rb') as file:
    model = pickle.load(file)

# Streamlit app
st.title('Customer Purchase Prediction')

# Input fields for the features
st.subheader('Input Customer Details')

age = st.number_input('Age', min_value=0, max_value=120, value=25)
income = st.number_input('Income', min_value=0, value=50000)
recency = st.number_input('Recency (days since last purchase)', min_value=0, value=10)
total_purchase = st.number_input('Total Purchase', min_value=0, value=1)

# Button for making predictions
if st.button('Predict'):
    # Create a DataFrame for the input data
    input_data = pd.DataFrame({
        'Age': [age],
        'Income': [income],
        'Recency': [recency],
        'Total Purchase': [total_purchase]
    })

    # Make predictions
    prediction = model.predict(input_data)[0]

    # Display the prediction
    st.subheader('Prediction')
    st.write(f'The predicted value is: {prediction}')