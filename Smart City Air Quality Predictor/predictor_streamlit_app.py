# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
from datetime import datetime, timedelta
from pymongo import MongoClient
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Load the trained model and scaler
model = joblib.load('models/aqi_model.joblib')
scaler = joblib.load('models/scaler.joblib')

# MongoDB connection
try:
    client = MongoClient(os.getenv('MONGODB_URI'), serverSelectionTimeoutMS=5000)
    db = client['air_quality_db']
    collection = db['air_quality_data']

    # Test connection
    client.server_info()
except Exception as e:
    st.error(f"Failed to connect to MongoDB: {e}")
    collection = None



def load_latest_data():
    """Load the latest data from MongoDB"""
    cursor = collection.find().sort('timestamp', -1).limit(100)
    data = list(cursor)
    return pd.DataFrame(data)


def predict_aqi(features):
    """Make AQI predictions using the trained model"""
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)
    return prediction[0]


# Streamlit UI
st.title('Smart City Air Quality Predictor')

# Sidebar for city selection
st.sidebar.title('Settings')
selected_city = st.sidebar.selectbox(
    'Select City',
    ['Beijing', 'London', 'New York', 'Tokyo', 'Delhi']
)

# Main content
st.header(f'Air Quality Analysis for {selected_city}')

if collection is not None:
    df = load_latest_data()
    if not df.empty:
        city_data = df[df['city'] == selected_city].copy()

        if not city_data.empty:
            # Current AQI
            latest_aqi = city_data.iloc[0].get('aqi', np.nan)
            if not pd.isna(latest_aqi):
                st.metric(
                    "Current AQI",
                    f"{latest_aqi:.1f}",
                    delta=f"{latest_aqi - city_data.iloc[1]['aqi']:.1f}" if len(city_data) > 1 else None
                )

                # AQI Category
                def get_aqi_category(aqi):
                    if aqi <= 50:
                        return "Good"
                    elif aqi <= 100:
                        return "Moderate"
                    elif aqi <= 150:
                        return "Unhealthy for Sensitive Groups"
                    elif aqi <= 200:
                        return "Unhealthy"
                    elif aqi <= 300:
                        return "Very Unhealthy"
                    else:
                        return "Hazardous"

                st.info(f"Air Quality Category: {get_aqi_category(latest_aqi)}")

                # Historical AQI Trend
                st.subheader('Historical AQI Trend')
                fig = px.line(city_data, x='timestamp', y='aqi',
                              title=f'AQI Trend in {selected_city}')
                st.plotly_chart(fig)

                # Pollutant Analysis
                st.subheader('Pollutant Levels')
                pollutants = ['pm25', 'pm10', 'o3', 'no2', 'so2', 'co']
                latest_pollutants = {p: city_data.iloc[0][f'pollutants.{p}'] for p in pollutants}

                col1, col2 = st.columns(2)
                with col1:
                    fig = px.bar(x=list(latest_pollutants.keys()),
                                 y=list(latest_pollutants.values()),
                                 title='Current Pollutant Levels')
                    st.plotly_chart(fig)

                # Weather Conditions
                with col2:
                    st.subheader('Current Weather Conditions')
                    weather_data = {
                        'Temperature': f"{city_data.iloc[0]['weather.temp']:.1f}°C",
                        'Humidity': f"{city_data.iloc[0]['weather.humidity']:.1f}%",
                        'Wind Speed': f"{city_data.iloc[0]['weather.wind_speed']:.1f} m/s",
                        'Pressure': f"{city_data.iloc[0]['weather.pressure']:.1f} hPa"
                    }
                    for param, value in weather_data.items():
                        st.metric(param, value)

                # Prediction Section
                st.header('AQI Prediction')
                col1, col2 = st.columns(2)

                with col1:
                    pm25 = st.number_input('PM2.5 Level', value=float(latest_pollutants['pm25']))
                    pm10 = st.number_input('PM10 Level', value=float(latest_pollutants['pm10']))
                    o3 = st.number_input('O3 Level', value=float(latest_pollutants['o3']))
                    no2 = st.number_input('NO2 Level', value=float(latest_pollutants['no2']))

                with col2:
                    so2 = st.number_input('SO2 Level', value=float(latest_pollutants['so2']))
                    co = st.number_input('CO Level', value=float(latest_pollutants['co']))
                    temp = st.number_input('Temperature (°C)', value=float(city_data.iloc[0]['weather.temp']))
                    humidity = st.number_input('Humidity (%)', value=float(city_data.iloc[0]['weather.humidity']))

                if st.button('Predict AQI'):
                    features = pd.DataFrame([[pm25, pm10, o3, no2, so2, co, temp, humidity]],
                                            columns=['pollutants.pm25', 'pollutants.pm10', 'pollutants.o3',
                                                     'pollutants.no2', 'pollutants.so2', 'pollutants.co',
                                                     'weather.temp', 'weather.humidity'])
                    predicted_aqi = predict_aqi(features)
                    st.success(f'Predicted AQI: {predicted_aqi:.1f} ({get_aqi_category(predicted_aqi)})')
        else:
            st.error(f"No data available for {selected_city}")
    else:
        st.error("No data found in the database.")
else:
    st.error("Database connection failed. Please check your MongoDB configuration.")

# Footer
st.markdown("---")
st.markdown("Built with Streamlit • Data from WAQI API")
