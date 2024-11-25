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
import ssl
from pymongo.errors import ServerSelectionTimeoutError, ConnectionFailure

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

# Features used during model training
MODEL_FEATURES = [
    'weather.temp',
    'weather.humidity',
    'weather.wind_speed',
    'weather.pressure',
    'pollutants.pm25',
    'pollutants.pm10',
    'pollutants.o3',
    'pollutants.no2',
    'pollutants.so2',
    'pollutants.co'
]

def init_mongodb():
    try:
        uri = os.getenv('MONGODB_URI')

        # Create an SSL context
        ssl_context = ssl.create_default_context(cafile=certifi.where())
        ssl_context.verify_mode = ssl.CERT_REQUIRED

        uri = "mongodb+srv://sylvester:sly@cluster0.vtd8d.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"

        # Create a new client and connect to the server
        client = MongoClient(uri, server_api=ServerApi('1'))

        db = client['air_quality_db']
        collection = db['air_quality_data']

        st.session_state.mongodb_connected = True
        st.session_state.connection_error = None

        return client, db, collection

    except Exception as e:
        error_msg = f"Unexpected error connecting to MongoDB: {str(e)}"
        logger.error(error_msg)
        st.session_state.connection_error = error_msg
        raise Exception(error_msg)

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
    """Load and preprocess the latest data from MongoDB."""
    try:
        if not st.session_state.mongodb_connected:
            raise ConnectionError("MongoDB is not connected")

        # Fetch and preprocess data
        cursor = collection.find().sort('timestamp', -1).limit(100)
        raw_data = list(cursor)

        # Convert to DataFrame
        df = pd.DataFrame(raw_data)

        # Extract nested fields
        if not df.empty:
            # Pollutant extraction
            df['pm25'] = df['pollutants'].apply(lambda x: x.get('pm25', None))
            df['pm10'] = df['pollutants'].apply(lambda x: x.get('pm10', None))
            df['o3'] = df['pollutants'].apply(lambda x: x.get('o3', None))
            df['no2'] = df['pollutants'].apply(lambda x: x.get('no2', None))
            df['so2'] = df['pollutants'].apply(lambda x: x.get('so2', None))
            df['co'] = df['pollutants'].apply(lambda x: x.get('co', None))

            # Weather extraction
            df['temperature'] = df['weather'].apply(lambda x: x.get('temp', None))
            df['humidity'] = df['weather'].apply(lambda x: x.get('humidity', None))
            df['wind_speed'] = df['weather'].apply(lambda x: x.get('wind_speed', None))
            df['pressure'] = df['weather'].apply(lambda x: x.get('pressure', None))
        return df
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        st.error(f"Failed to load data: {str(e)}")
        return pd.DataFrame()


def predict_aqi(features, feature_names):
    try:
        # Validate the number of features
        if len(features[0]) != len(feature_names):
            raise ValueError(f"Expected {len(feature_names)} features, got {len(features[0])}")

        # Convert features to DataFrame for consistency with training
        features_df = pd.DataFrame(features, columns=feature_names)

        # Scale features
        features_scaled = scaler.transform(features_df)

        # Make prediction
        prediction = model.predict(features_scaled)
        return prediction[0]

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        st.error(f"Prediction failed: {str(e)}")
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
    # Display current AQI and pollutant levels
    col1, col2, col3 = st.columns(3)
    if 'aqi' in df.columns:
        col1.metric("Current AQI", f"{df['aqi'].iloc[0]:.0f}")
    else:
        col1.warning("AQI data unavailable.")

    if 'pm25' in df.columns:
        col2.metric("PM2.5", f"{df['pm25'].iloc[0]:.1f} µg/m³")
    else:
        col2.warning("PM2.5 data unavailable.")

    if 'temperature' in df.columns:
        col3.metric("Temperature", f"{df['temperature'].iloc[0]:.1f}°C")
    else:
        col3.warning("Temperature data unavailable.")

    # Display detailed data table
    st.subheader('Recent Measurements')
    st.dataframe(df[['timestamp', 'aqi', 'pm25', 'temperature', 'humidity']] if not df.empty else None)

with tab2:
    st.header('AQI Prediction')

    # Input form for predictions
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        with col1:
            temperature = st.number_input('Temperature (°C)', value=25.0, min_value=-50.0, max_value=50.0)
            humidity = st.number_input('Humidity (%)', value=60.0, min_value=0.0, max_value=100.0)
            wind_speed = st.number_input('Wind Speed (m/s)', value=2.0, min_value=0.0, max_value=50.0)
            pressure = st.number_input('Atmospheric Pressure (hPa)', value=1013.0, min_value=800.0, max_value=1200.0)
        with col2:
            pm25 = st.number_input('PM2.5 (µg/m³)', value=15.0, min_value=0.0, max_value=500.0)
            pm10 = st.number_input('PM10 (µg/m³)', value=20.0, min_value=0.0, max_value=500.0)
            o3 = st.number_input('Ozone (ppb)', value=30.0, min_value=0.0, max_value=200.0)
            no2 = st.number_input('Nitrogen Dioxide (ppb)', value=10.0, min_value=0.0, max_value=200.0)
            so2 = st.number_input('Sulfur Dioxide (ppb)', value=5.0, min_value=0.0, max_value=100.0)
            co = st.number_input('Carbon Monoxide (ppm)', value=1.0, min_value=0.0, max_value=50.0)

        submitted = st.form_submit_button("Predict AQI")

        if submitted:
            try:
                # Prepare features in the correct order matching MODEL_FEATURES
                features = np.array([[
                    temperature,    # weather.temp
                    humidity,       # weather.humidity
                    wind_speed,     # weather.wind_speed
                    pressure,       # weather.pressure
                    pm25,           # pollutants.pm25
                    pm10,           # pollutants.pm10
                    o3,             # pollutants.o3
                    no2,            # pollutants.no2
                    so2,            # pollutants.so2
                    co              # pollutants.co
                ]])

                prediction = predict_aqi(features, MODEL_FEATURES)

                if prediction is not None:
                    st.success(f'Predicted AQI: {prediction:.2f}')

                    # Display AQI category
                    if prediction <= 50:
                        st.info('Category: Good')
                    elif prediction <= 100:
                        st.warning('Category: Moderate')
                    elif prediction <= 150:
                        st.warning('Category: Unhealthy for Sensitive Groups')
                    elif prediction <= 200:
                        st.error('Category: Unhealthy')
                    elif prediction <= 300:
                        st.error('Category: Very Unhealthy')
                    else:
                        st.error('Category: Hazardous')
            except Exception as e:
                st.error(f"Prediction failed: {e}")

with tab3:
    st.header('Historical Trends')

    # Date range selector
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input('Start Date', datetime.now() - timedelta(days=7))
    with col2:
        end_date = st.date_input('End Date', datetime.now())

    # Logging for debugging
    logger.info(f"Fetching historical data from {start_date} to {end_date}")

    # Load historical data
    if st.session_state.mongodb_connected:
        try:
            query = {
                'timestamp': {
                    '$gte': datetime.combine(start_date, datetime.min.time()),
                    '$lte': datetime.combine(end_date, datetime.max.time())
                }
            }
            historical_data = list(collection.find(query))

            if historical_data:
                # Create DataFrame with robust error handling
                try:
                    df_historical = pd.DataFrame(historical_data)


                    # Safely extract nested data
                    def safe_extract(row, parent_key, child_key, default=0):
                        try:
                            return row.get(parent_key, {}).get(child_key, default) if isinstance(row.get(parent_key),
                                                                                                 dict) else default
                        except Exception as e:
                            logger.warning(f"Data extraction error: {e}")
                            return default


                    # Apply safe extraction
                    df_historical['aqi'] = df_historical['aqi'].fillna(0)
                    df_historical['pm25'] = df_historical.apply(lambda row: safe_extract(row, 'pollutants', 'pm25'),
                                                                axis=1)
                    df_historical['temperature'] = df_historical.apply(lambda row: safe_extract(row, 'weather', 'temp'),
                                                                       axis=1)

                    # Convert timestamp
                    df_historical['timestamp'] = pd.to_datetime(df_historical['timestamp'])

                    # Sort by timestamp to ensure correct line plot
                    df_historical = df_historical.sort_values('timestamp')

                    # Create interactive time series plot
                    fig = px.line(df_historical,
                                  x='timestamp',
                                  y=['aqi', 'pm25', 'temperature'],
                                  title='Air Quality and Weather Trends',
                                  labels={'value': 'Measurement', 'variable': 'Metric'},
                                  height=500)

                    # Customize plot appearance
                    fig.update_layout(
                        xaxis_title='Timestamp',
                        legend_title_text='Metrics',
                        hovermode='x unified'
                    )

                    st.plotly_chart(fig, use_container_width=True)

                    # Display summary statistics
                    st.subheader('Summary Statistics')
                    summary_stats = df_historical[['aqi', 'pm25', 'temperature']].describe()
                    st.dataframe(summary_stats)

                except Exception as data_error:
                    logger.error(f"DataFrame processing error: {data_error}")
                    st.error(f"Could not process historical data: {data_error}")

            else:
                st.info('No historical data available for the selected date range')

        except Exception as query_error:
            logger.error(f"Historical data query error: {query_error}")
            st.error(f"Failed to retrieve historical data: {query_error}")
    else:
        st.warning("MongoDB is not connected. Please check your connection.")

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
