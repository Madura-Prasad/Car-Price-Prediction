import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the trained model
with open('Car_price_prediction.pkl', 'rb') as file:
    model = pickle.load(file)

# Function to preprocess input data
def preprocess_input(data):
    # Encode categorical features
    cat_features = ['model', 'motor_type', 'color', 'type', 'status','running_km']
    for feature in cat_features:
        data[feature] = data[feature].astype('category').cat.codes
    return data

# Function to predict car prices
def predict_price(data):
    # Preprocess the input data
    processed_data = preprocess_input(data)
    
    # Predict using the trained model
    prediction = model.predict(processed_data)
    
    return prediction

# Main function for Streamlit app
def main():
    # Title of the web app
    st.write("<h1 style='color: blue;'>Car Price Prediction</h1>", unsafe_allow_html=True)
    
   
    
    # Input fields for features
    model = st.text_input("Enter Model")
    year = st.number_input("Enter Year")
    motor_type = st.text_input("Enter Motor Type")
    color = st.text_input("Enter Color")
    car_type = st.text_input("Enter Car Type")
    status = st.text_input("Enter Status")
    motor_volume = st.number_input("Enter Motor Volume")
    running_km = st.number_input("Enter Running Kilometers")

    # Predict car prices when button is clicked
    if st.button('Predict Price'):
        # Create a dataframe from user inputs
        user_data = pd.DataFrame({
            'model': [model],
            'year': [year],
            'motor_type': [motor_type],
            'color': [color],
            'type': [car_type],
            'status': [status],
            'motor_volume': [motor_volume],
            'running_km': [running_km]
        })

        # Make predictions
        prediction = predict_price(user_data)

        # Display the prediction as a label
        st.write('**Predicted Price:**')
        st.write(f"<font color='red'>${prediction[0]:,.2f}</font>", unsafe_allow_html=True)

if __name__ == '__main__':
    main()
