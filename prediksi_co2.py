import pickle
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Load the model
model = pickle.load(open('prediksi_co2.sav', 'rb'))

# Load and preprocess the dataset
df = pd.read_excel("CO2 dataset.xlsx")
df['Year'] = pd.to_datetime(df['Year'], format='%Y')
df.set_index(['Year'], inplace=True)

st.title('Forecasting CO2')
year = st.slider("Tentukan Tahun", 1, 30, step=1)

# Generate predictions
pred = model.forecast(year)
pred = pd.DataFrame(pred, columns=['CO2'])

# Create a DateTimeIndex for predictions
last_year = df.index[-1].year
pred_index = pd.date_range(start=f'{last_year + 1}-01-01', periods=year, freq='YS')
pred.index = pred_index

# Ensure the 'CO2' column is numeric
pred['CO2'] = pd.to_numeric(pred['CO2'], errors='coerce')

# Handle missing values
pred['CO2'].fillna(method='ffill', inplace=True)  # Forward fill as an example

if st.button("Predict"):
    col1, col2 = st.columns([2, 3])
    with col1:
        st.dataframe(pred)
    with col2:
        fig, ax = plt.subplots()
        df['CO2'].plot(style='--', color='gray', legend=True, label='known')
        pred['CO2'].plot(color='b', legend=True, label='prediction')
        st.pyplot(fig)
