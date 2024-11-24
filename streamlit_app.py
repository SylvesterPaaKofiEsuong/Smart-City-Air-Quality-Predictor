import streamlit as st
import pandas as pd
import numpy as np
import certifi
import joblib
import plotly.express as px
from datetime import datetime, timedelta
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
import os
from dotenv import load_dotenv
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize session state for MongoDB connection
if 'mongodb_connected' not in st.session_state:
    st.session_state.mongodb_connected = False
if 'connection_error' not in st.session_state:
    st.session_state.connection_error = None

import certifi


def init_mongodb():
    """Initialize MongoDB connection with proper error handling"""
    try:
        uri = "mongodb+srv://sylvester:PvlOeLmrNfoqmtuZ@cluster0.vtd8d.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"

        import ssl
        client = MongoClient(
            uri,
            tls=True,
            tlsAllowInvalidCertificates=True,
            server_api=ServerApi('1'),
            ssl_cert_reqs=ssl.CERT_NONE,
            tls_insecure=True
        )

        db = client['air_quality_db']
        collection = db['air_quality_data']

        # Test connection
        client.admin.command('ping')

        st.session_state.mongodb_connected = True
        return client, db, collection

    except Exception as e:
        print(f"Error: {e}")
        raise e


# Load the trained model and scaler
try:
    model = joblib.load('models/aqi_model.joblib')
    scaler = joblib.load('models/scaler.joblib')
except Exception as e:
    st.error(f"Failed to load model files: {e}")
    st.stop()

# Initialize MongoDB connection
try:
    client, db, collection = init_mongodb()
except Exception as e:
    st.error(str(e))
    st.error("""
    Please check:
    1. MongoDB service is running
    2. MONGODB_URI in .env file is correct
    3. Network connectivity to MongoDB server
    4. MongoDB server is accepting connections
    """)
    collection = None


def load_latest_data():
    """Load the latest data from MongoDB with error handling"""
    try:
        if not st.session_state.mongodb_connected:
            raise ConnectionError("MongoDB is not connected")

        cursor = collection.find().sort('timestamp', -1).limit(100)
        data = list(cursor)
        return pd.DataFrame(data)
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        st.error(f"Failed to load data: {str(e)}")
        return pd.DataFrame()


def predict_aqi(features):
    """Make AQI predictions using the trained model"""
    try:
        features_scaled = scaler.transform(features)
        prediction = model.predict(features_scaled)
        return prediction[0]
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        st.error(f"Failed to make prediction: {str(e)}")
        return None


# Streamlit UI
st.title('Smart City Air Quality Predictor')

# Connection status indicator in sidebar
st.sidebar.title('System Status')
if st.session_state.mongodb_connected:
    st.sidebar.success('✓ Connected to MongoDB')
else:
    st.sidebar.error('✗ MongoDB Disconnected')
    if st.session_state.connection_error:
        st.sidebar.error(f"Error: {st.session_state.connection_error}")
    if st.sidebar.button('Retry Connection'):
        try:
            client, db, collection = init_mongodb()
            st.experimental_rerun()
        except Exception as e:
            st.sidebar.error(f"Reconnection failed: {str(e)}")

# Add city selection
cities = ['New York', 'London', 'Tokyo', 'Paris', 'Singapore']  # Add your cities
selected_city = st.selectbox('Select City', cities)

# Create tabs for different views
tab1, tab2, tab3 = st.tabs(['Current Data', 'Predictions', 'Historical Trends'])

with tab1:
    st.header('Current Air Quality Data')

    # Load and display latest data
    df = load_latest_data()
    if not df.empty:
        # Display current AQI and pollutant levels
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Current AQI", f"{df['aqi'].iloc[0]:.0f}")
        with col2:
            st.metric("PM2.5", f"{df['pm25'].iloc[0]:.1f} µg/m³")
        with col3:
            st.metric("Temperature", f"{df['temperature'].iloc[0]:.1f}°C")

        # Display detailed data table
        st.subheader('Recent Measurements')
        st.dataframe(df[['timestamp', 'aqi', 'pm25', 'temperature', 'humidity']])

with tab2:
    st.header('AQI Prediction')

    # Input form for predictions
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        with col1:
            temperature = st.number_input('Temperature (°C)', value=25.0)
            humidity = st.number_input('Humidity (%)', value=60.0)
        with col2:
            pm25 = st.number_input('PM2.5 (µg/m³)', value=15.0)
            wind_speed = st.number_input('Wind Speed (m/s)', value=2.0)

        submitted = st.form_submit_button("Predict AQI")

        if submitted:
            # Prepare features for prediction
            features = np.array([[temperature, humidity, pm25, wind_speed]])
            prediction = predict_aqi(features)

            if prediction is not None:
                st.success(f'Predicted AQI: {prediction:.2f}')

                # Add AQI category
                if prediction <= 50:
                    st.info('Category: Good')
                elif prediction <= 100:
                    st.warning('Category: Moderate')
                else:
                    st.error('Category: Unhealthy')

with tab3:
    st.header('Historical Trends')

    # Date range selector
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input('Start Date', datetime.now() - timedelta(days=7))
    with col2:
        end_date = st.date_input('End Date', datetime.now())

    # Load historical data
    if st.session_state.mongodb_connected:
        query = {
            'timestamp': {
                '$gte': datetime.combine(start_date, datetime.min.time()),
                '$lte': datetime.combine(end_date, datetime.max.time())
            }
        }
        historical_data = list(collection.find(query))
        if historical_data:
            df_historical = pd.DataFrame(historical_data)

            # Create time series plot
            fig = px.line(df_historical,
                          x='timestamp',
                          y=['aqi', 'pm25'],
                          title='Air Quality Trends')
            st.plotly_chart(fig)

            # Display summary statistics
            st.subheader('Summary Statistics')
            summary_stats = df_historical[['aqi', 'pm25', 'temperature', 'humidity']].describe()
            st.dataframe(summary_stats)
        else:
            st.info('No historical data available for the selected date range')

# Footer with additional system info
st.markdown("---")
col1, col2 = st.columns(2)
with col1:
    st.markdown("Built with Streamlit • Data from WAQI API")
with col2:
    if st.session_state.mongodb_connected:
        try:
            server_status = client.admin.command('serverStatus')
            st.markdown(f"MongoDB Version: {server_status.get('version', 'Unknown')}")
        except:
            st.markdown("MongoDB Status: Connected")


# Cleanup connection on session end
def cleanup():
    if 'client' in globals():
        client.close()
        logger.info("MongoDB connection closed")


import atexit

atexit.register(cleanup)