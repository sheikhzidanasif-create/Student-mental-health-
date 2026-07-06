
import streamlit as st
import pandas as pd
import pickle

st.title('Anxiety Prediction App')
st.write('Predicting \'Do you have Anxiety?\' based on Age using a Linear Regression Model.')

# Load the trained model
try:
    with open('linear_regression_model.pkl', 'rb') as file:
        model = pickle.load(file)
    st.success('Model loaded successfully!')
except FileNotFoundError:
    st.error('Error: linear_regression_model.pkl not found. Please ensure the model is saved.')
    st.stop()

# Input for Age
age = st.slider('Select Age', min_value=18, max_value=60, value=25)

if st.button('Predict Anxiety'):
    # Create a DataFrame for prediction
    input_data = pd.DataFrame({'Age': [age]})

    # Make prediction
    prediction = model.predict(input_data)

    # The model predicts a continuous value. We can interpret it as a probability or a score.
    # For simplicity, let's say if the prediction is > 0.5, it indicates 'Yes' for anxiety.
    # Note: Linear Regression might produce values outside [0, 1]. A classification model would be better for binary outcomes.
    st.subheader('Prediction Result:')
    if prediction[0] > 0.5:
        st.write(f"Based on the age {age}, the model predicts: **Likely to have Anxiety**")
    else:
        st.write(f"Based on the age {age}, the model predicts: **Less likely to have Anxiety**")
    st.write(f"(Raw prediction score: {prediction[0]:.2f})")
