import streamlit as st
import pandas as pd 
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl

# @st.cache_data
def read_data():
    df = pd.read_csv('/Users/wout_vp/Code/CO2_Emissions_Predicting_End_to_End_Omdena/DataSets /WORLD-OWID-Features')
    df.set_index('year', inplace=True)
    mean_co2 = df.loc[[2019, 2021], 'co2'].mean()
    df.loc[2020, 'co2'] = mean_co2
    return df 

df = read_data()

if df not in st.session_state:
    st.session_state['df'] = df

#def generate_synthetic_input(df, horizon):
    # Get the last available data point
    #last_data_point = df.iloc[-1]

    # Generate synthetic input data for future years
    #synthetic_input = []
    #for year in range(1, horizon + 1):
        #synthetic_input.append(last_data_point)


model = keras.models.load_model('/Users/wout_vp/Code/CO2_Emissions_Predicting_End_to_End_Omdena/Transformers_Best_Models_and_weights/TestSet_only_best_model_weights_946.6525.hdf5')

def generate_synthetic_input(start_year, selected_feature, horizon):
    #num_features = 8
    last_year_data = df.loc[start_year, selected_feature]
    input_sequence = np.concatenate([last_year_data, np.zeros(horizon - 1)])
    return np.array([input_sequence])

def forecast(input_data):
    forecast = model.predict(input_data)
    return forecast

def plot(series):
    df = pd.DataFrame(series, columns=["predicted_mean"])
    sns.set(style="whitegrid")
    plt.figure(figsize=(8, 3))
    fig = sns.lineplot(x=df.index, y=df.predicted_mean, data=df, markers=True, palette="flare", hue_norm=mpl.colors.LogNorm())
    plt.xlabel("Year")
    plt.ylabel("Carbon Emission Value")
    plt.title("Yearly Carbon Emission Values")
    st.pyplot(fig.get_figure())

def main():
    st.title('Forecasting the CO2 Emissions of different Industries')

    Industry, Forecast_horizon = st.columns(2)

    target = Industry.selectbox('Select your Industry', df.columns)
    horizon = Forecast_horizon.slider('Choose your Forecast horizon', min_value=1, max_value=20, value=5)

    forecast_btn = st.button('Start your Forecast')

    if forecast_btn:
        selected_feature = target
        start_year = df.index.max() 
        num_features = len(df.columns)


        synthetic_input = generate_synthetic_input(start_year+1, selected_feature, horizon)
        feature_forecast = forecast(synthetic_input)
        st.write("Forecast: ", feature_forecast)


if __name__ == "__main__":
    main()

    #forecast = model.predict(steps=horizon)
    #selected_feature_index = df.columns.get_loc(selected_feature)
    #feature_pred = forecast[:, selected_feature_index]
    #return feature_pred

#def create_feature_vector(historical_data horizon):
    #last_row = 




 


