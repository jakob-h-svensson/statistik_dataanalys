import joblib
import streamlit as st
import pandas as pd
import numpy as np

loaded_model = joblib.load("car_price_linear_regression_model.pkl")
model_columns = joblib.load("model_columns.pkl")

df_price_data = pd.read_csv("car_price_dataset.csv", delimiter=";")

# Visuals
st.markdown("""
# ProbablyAccurate.ai – Enterprise-Grade Automotive Price Intelligence
""")

st.write("Set your preferences below and click Predict to receive an estimated price for the vehicle.")

brand_options = df_price_data['Brand'].unique()
brand = st.selectbox(
    "Select car brand:",
    brand_options
)

model_options = df_price_data[df_price_data['Brand'] == brand]['Model'].unique()
model = st.selectbox(
    "Select model:",
    model_options
)

year_min = df_price_data['Year'].min()

year = st.slider(
    "Select year:",
    min_value=year_min,
    max_value=2026,
    value=2026
)

engine_size_min = df_price_data['Engine_Size'].min()
engine_size_max = df_price_data['Engine_Size'].max()

engine_size = st.slider(
    "Select Engine Size (L):",
    min_value=engine_size_min,
    max_value=engine_size_max,
)

fuel_type_option = df_price_data[df_price_data['Model'] == model]['Fuel_Type'].unique()
fuel = st.selectbox(
    "Select Fuel Type:",
    fuel_type_option
)

transmission_option = df_price_data['Transmission'].unique()
transmission = st.selectbox(
    "Select Transmission:",
    transmission_option
)

milage_option_min = df_price_data['Mileage'].min()
milage_option_max = df_price_data['Mileage'].max()
milage = st.number_input(
    "Select Mileage:"
)

doors = st.slider(
    "Select Doors:",
    min_value=1,
    max_value=5,
)

owner_count = st.number_input(
    "Select Owner Count:",
    min_value=0,
    max_value=20,
)

input_data = {
    "Brand": brand,
    "Model": model,
    "Year": year,
    "Engine_Size": engine_size,
    "Fuel_Type": fuel,
    "Transmission": transmission,
    "Mileage": milage,
    "Doors": doors,
    "Owner_Count": owner_count
}

input_df = pd.DataFrame([input_data])
input_encoded = pd.get_dummies(input_df)

input_encoded = input_encoded.reindex(columns=model_columns, fill_value=0)

prediction = loaded_model.predict(input_encoded)[0]

if st.button("Predict"):

    st.subheader("Predicted Price")
    st.success(f"{prediction}")






