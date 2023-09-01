import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMAResults
import matplotlib as mpl
from pandas.tseries.offsets import MonthBegin


def read_data():
    df = pd.read_csv('/Users/wout_vp/Code/CO2_Emissions_Predicting_End_to_End_Omdena/DataSets /WORLD-OWID-Features-Monthly')
    df.set_index('year', inplace=True)
    return df 

df = read_data()

loaded_models = {}

features = df.columns
for feature in features:
    model_path = f"/Users/wout_vp/Code/CO2_Emissions_Predicting_End_to_End_Omdena/Saved_ARIMA_Models/{feature}_ARIMA_Model.pkl"
    loaded_models[feature] = ARIMAResults.load(model_path)


def predict(model, months):
    forecast = model.forecast(steps=months)
    return forecast

def main():
    st.title('CO2 Emissions Forecasting App')
    
    selected_feature, visualization_options, months = st.columns(3)

    selected_feature = st.selectbox('Select your Industry to forecast:', features)
    visualization_options = visualization_options.radio('Choose plot option:', ['Forecasted Values', 'Whole Dataset'])
    months = st.slider('Select the number of months for prediction:', 1, 96, value=1)
    run_button = st.button('Run Forecast')
    
    if run_button:

        selected_model = loaded_models[selected_feature]
        forecast_values = predict(selected_model, months)
        
        plt.figure(figsize=(10, 6))
        
        if visualization_options == 'Forecasted Values':
            forecast_dates = pd.date_range(start=df.index[-1], periods=months + 1, freq = 'MS')[1:]
            plt.plot(forecast_dates, forecast_values, label='Forecasted Values', color='orange')
        elif visualization_options == 'Whole Dataset':
            #starting_point = st.date_input('Select starting point:', df.index.min(), df.index.max(), value=df.index.min(), key='starting_point')
            plt.plot(df.index, df[selected_feature], label='Original Data', color='blue')
            #forecast_dates = pd.date_range(start=df.index[-1], periods=months + 1, freq='MS')
            #plt.plot(forecast_dates, forecast_values, label='Forecasted Values', color='orange')
            plt.plot(range(len(df), len(df) + len(forecast_values)), forecast_values, label='Forecasted Values', color='orange')

        plt.xlabel('Date')
        plt.ylabel(selected_feature)
        plt.title(f'{selected_feature} Forecast')
        plt.legend()
        st.pyplot(plt.gcf())
    
if __name__ == '__main__':
    main()