import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
from datetime import datetime, timedelta
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
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


def init_mongodb():
    """Initialize MongoDB connection with proper error handling"""
    uri = os.getenv('MONGODB_URI')
    if not uri:
        raise ValueError("MONGODB_URI environment variable is not set")

    try:
        # MongoDB secure connection
        client = MongoClient(uri, server_api=ServerApi('1'))

        # Test the connection
        client.admin.command('ping')

        db = client['air_quality_db']
        collection = db['air_quality_data']

        st.session_state.mongodb_connected = True
        st.session_state.connection_error = None

        logger.info("Successfully connected to MongoDB")
        return client, db, collection

    except Exception as e:
        error_msg = f"Unexpected error connecting to MongoDB: {str(e)}"
        logger.error(error_msg)
        st.session_state.connection_error = error_msg
        raise Exception(error_msg)

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

# Rest of your existing code remains the same...
# [Previous code for city selection, data display, and predictions]

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