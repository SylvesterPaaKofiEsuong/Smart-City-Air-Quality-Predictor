import requests
import pandas as pd
from datetime import datetime
import os
from dotenv import load_dotenv
import time
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi

# Load environment variables
load_dotenv()
time.sleep(1)


class AQIDataCollector:
    def __init__(self):
        self.api_key = os.getenv('WAQI_API_KEY')
        self.base_url = "https://api.waqi.info/feed"
        print(f"Base URL initialized: {self.base_url}")

        # MongoDB secure connection
        uri = os.getenv('MONGODB_URI')
        if not uri:
            raise ValueError("MongoDB URI not found. Please set 'MONGODB_URI' in your environment.")

        try:
            self.mongo_client = MongoClient(uri, server_api=ServerApi('1'))
            self.mongo_client.admin.command('ping')  # Validate connection
            print("Pinged your MongoDB deployment. Successfully connected!")
        except Exception as e:
            raise ConnectionError(f"Failed to connect to MongoDB: {e}")

        self.db = self.mongo_client['air_quality_db']
        self.collection = self.db['air_quality_data']

        # Validate required environment variables
        if not self.api_key:
            raise ValueError("API key not found. Please set 'WAQI_API_KEY' in your environment.")

    def fetch_city_data(self, city):
        """
        Fetch current AQI data for a specific city
        """
        url = f"{self.base_url}/{city}/?token={self.api_key}"
        response = requests.get(url)

        if response.status_code == 200:
            data = response.json()
            if data['status'] == 'ok':
                return self._parse_response(data['data'])
            else:
                print(f"API returned status '{data['status']}' for {city}.")
        else:
            print(f"Failed to fetch data for {city}. HTTP Status: {response.status_code}")
        return None

    def _parse_response(self, data):
        """
        Parse the API response into a structured format
        """
        try:
            parsed_data = {
                'timestamp': datetime.fromtimestamp(data['time']['v']),
                'aqi': data['aqi'],
                'city': data['city']['name'],
                'lat': data['city']['geo'][0],
                'lon': data['city']['geo'][1],
                'pollutants': {
                    'pm25': data.get('iaqi', {}).get('pm25', {}).get('v'),
                    'pm10': data.get('iaqi', {}).get('pm10', {}).get('v'),
                    'o3': data.get('iaqi', {}).get('o3', {}).get('v'),
                    'no2': data.get('iaqi', {}).get('no2', {}).get('v'),
                    'so2': data.get('iaqi', {}).get('so2', {}).get('v'),
                    'co': data.get('iaqi', {}).get('co', {}).get('v')
                },
                'weather': {
                    'temp': data.get('iaqi', {}).get('t', {}).get('v'),
                    'pressure': data.get('iaqi', {}).get('p', {}).get('v'),
                    'humidity': data.get('iaqi', {}).get('h', {}).get('v'),
                    'wind_speed': data.get('iaqi', {}).get('w', {}).get('v')
                }
            }
            return parsed_data
        except Exception as e:
            print(f"Error parsing response: {e}")
            return None

    def save_to_mongodb(self, data):
        """
        Save or update the collected data in MongoDB.
        """
        if data:
            try:
                self.collection.update_one(
                    {'city': data['city'], 'timestamp': data['timestamp']},
                    {'$set': data},
                    upsert=True
                )
                print(f"Data saved/updated for {data['city']} at {data['timestamp']}")
            except Exception as e:
                print(f"Error saving to MongoDB: {e}")

    def collect_data_for_cities(self, cities):
        collected_data = []
        existing_cities = self.collection.distinct('city')  # Check MongoDB for already collected cities

        for city in cities:
            if city not in existing_cities:  # Avoid re-fetching cities already in the database
                print(f"Fetching data for {city}...")
                data = self.fetch_city_data(city)
                if data:
                    self.save_to_mongodb(data)
                    collected_data.append(data)
                print(f"Waiting before the next request...")
                time.sleep(1)  # Delay to respect API rate limits
            else:
                print(f"Data for {city} already exists. Skipping...")

        return collected_data

    def export_to_csv(self, data,
                      filename="air_quality_data.csv"):
        """
        Append collected data to CSV without overwriting existing data.
        """
        if data:
            try:
                # Ensure the directory exists
                directory = os.path.dirname(filename)
                if not os.path.exists(directory):
                    os.makedirs(directory)

                # Load existing data if the file exists
                if os.path.exists(filename):
                    existing_df = pd.read_csv(filename)
                    new_df = pd.json_normalize(data)
                    new_df = new_df.dropna(how='all', axis=1)  # Drop empty or NaN-only columns

                    # Drop duplicates based on 'timestamp' and 'city'
                    combined_df = pd.concat([existing_df, new_df], ignore_index=True).drop_duplicates(
                        subset=['timestamp', 'city'], keep='last')

                else:
                    combined_df = pd.json_normalize(data)

                # Save the combined data back to the CSV
                combined_df.to_csv(filename, index=False)
                print(f"Data appended to {filename}")
            except Exception as e:
                print(f"Error exporting to CSV: {e}")

    def export_to_parquet(self, data,
                          filename="air_quality_data.parquet"):
        """
        Export collected data to a Parquet file format.
        """
        if data:
            try:
                df = pd.json_normalize(data)
                df.to_parquet(filename, engine='pyarrow', index=False)
                print(f"Data exported to {filename}")
            except Exception as e:
                print(f"Error exporting to Parquet: {e}")


if __name__ == "__main__":
    collector = AQIDataCollector()
    cities = [
        'Beijing','Shanghai','Chengdu','Shenyang','Shenzhen', 'Guangzhou',
        'Qingdao', 'Xian', 'Tianjin', 'Saitama', 'Kyoto', 'Osaka', 'Seoul', 'Busan','Bogota',
        'Delhi', 'Jakarta', 'Ulaanbaatar', 'Hanoi', 'Chennai', 'Kolkata', 'Mumbai', 'Hyderabad',
        'Santiago', 'Lima', 'Saopaulo', 'Quito', 'Singapore', 'Kuala-lumpur', 'Ipoh', 'Perai', 'Miri',
        'New York', 'Seattle', 'Chicago', 'Boston', 'Atlanta'
    ]

    # Collect data for multiple cities
    collected_data = collector.collect_data_for_cities(cities)

    # Export collected data
    collector.export_to_csv(collected_data)
    collector.export_to_parquet(collected_data)
