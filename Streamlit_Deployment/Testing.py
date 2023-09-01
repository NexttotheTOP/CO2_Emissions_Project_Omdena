import streamlit as st
import pandas as pd
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
import matplotlib as mpl

# Read data from CSV file and set index
def read_data():
    df = pd.read_csv('/Users/wout_vp/Code/CO2_Emissions_Predicting_End_to_End_Omdena/DataSets /WORLD-OWID-Features-Yearly')
    df.set_index('year', inplace=True)
    return df

# Load data into session state
df = read_data()
if 'df' not in st.session_state:
    st.session_state['df'] = df

# Load the model
model_path = '/Users/wout_vp/Code/CO2_Emissions_Predicting_End_to_End_Omdena/Transformers_Best_Models_and_weights/TestSet_Multivariate_monthly_best_model_weights_656.3905.hdf5'
model = keras.models.load_model(model_path)

print(df.shape)
# Define function for forecasting
def forecast_multivariate(data, window_length, forecast_horizon):
    num_features = data.shape[1]
    forecasted_values = []

    for i in range(len(data) - window_length + 1):
        window = data[i:i + window_length]

        forecast = model.predict(window)
        forecast = forecast.reshape(forecast_horizon, num_features)
        forecasted_values.append(forecast)

    return np.array(forecasted_values)


# Streamlit App
st.title("Multivariate Time Series Forecasting")

# User input for forecast horizon
forecast_horizon = 5

# Get the last 5 rows of the dataset
input_data = df.iloc[-5:].values

# Perform forecasting
forecasted_values = forecast_multivariate(input_data, window_length=5, forecast_horizon=forecast_horizon)

# Create a DataFrame for visualization
forecast_df = pd.DataFrame(forecasted_values, columns=df.columns)
forecast_index = pd.date_range(start=df.index[-5], periods=forecast_horizon, freq=df.index.freq)
forecast_df.index = forecast_index

# Display forecasted values
st.write("Forecasted Values:")
st.write(forecast_df)

# Visualize forecasted values
st.write("Forecasted Values Visualization:")
plt.figure(figsize=(10, 6))
for column in forecast_df.columns:
    plt.plot(forecast_df.index, forecast_df[column], label=column)

plt.xlabel("Year")
plt.ylabel("Forecasted Value")
plt.title("Forecasted Values")
plt.legend()
st.pyplot(plt)