import joblib
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib

matplotlib.use('Agg')  # Switch to a non-GUI backend


class AQIModelTrainer:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()

    def prepare_features(self, df):
        """
        Prepare features for model training
        """
        # Select relevant features
        feature_columns = [
            'pollutants.pm25', 'pollutants.pm10', 'pollutants.o3',
            'pollutants.no2', 'pollutants.so2', 'pollutants.co',
            'weather.temp', 'weather.pressure', 'weather.humidity',
            'weather.wind_speed'
        ]

        # Ensure required columns exist
        missing_columns = [col for col in feature_columns + ['aqi'] if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Required columns missing: {missing_columns}")

        # Print missing data summary
        print("Missing values per column:")
        print(df.isnull().sum())

        # Convert relevant columns to numeric (force invalid values to NaN)
        for col in feature_columns + ['aqi']:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Handle missing values using interpolation and column means
        df_clean = df[feature_columns + ['aqi']].interpolate(method='linear').fillna(df.mean(numeric_only=True))

        # Drop rows with remaining NaNs
        df_clean = df_clean.dropna(subset=feature_columns + ['aqi'])
        print(f"Data after cleaning and interpolation: {df_clean.shape}")

        if df_clean.shape[0] == 0:
            raise ValueError("Dataset is empty after preprocessing. Please check the data.")

        # Extract features and target
        X = df_clean[feature_columns]
        y = df_clean['aqi']

        return X, y

    def train_model(self, X, y):
        """
        Train the Random Forest model
        """
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Scale the features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Train the model
        self.model = RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        )
        self.model.fit(X_train_scaled, y_train)

        # Make predictions
        y_pred = self.model.predict(X_test_scaled)

        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        return {
            'mse': mse,
            'rmse': np.sqrt(mse),
            'r2': r2,
            'test_actual': y_test,
            'test_pred': y_pred
        }

    def save_model(self, model_path='models/aqi_model.joblib',
                   scaler_path='models/scaler.joblib'):
        """
        Save the trained model and scaler
        """
        # Ensure the models directory exists
        if not os.path.exists('models'):
            os.makedirs('models')

        joblib.dump(self.model, model_path)
        joblib.dump(self.scaler, scaler_path)

    def plot_feature_importance(self, X):
        """
        Plot feature importance
        """
        importance = self.model.feature_importances_
        features = X.columns

        # Ensure the directory exists
        if not os.path.exists('visualizations'):
            os.makedirs('visualizations')

        plt.figure(figsize=(10, 6))
        sns.barplot(x=importance, y=features)
        plt.title('Feature Importance in AQI Prediction')
        plt.xlabel('Importance')
        plt.tight_layout()
        plt.savefig('visualizations/feature_importance.png')

    def plot_prediction_scatter(self, actual, predicted):
        """
        Plot actual vs predicted values
        """
        plt.figure(figsize=(10, 6))
        plt.scatter(actual, predicted, alpha=0.5)
        plt.plot([actual.min(), actual.max()], [actual.min(), actual.max()], 'r--')
        plt.xlabel('Actual AQI')
        plt.ylabel('Predicted AQI')
        plt.title('Actual vs Predicted AQI Values')
        plt.tight_layout()
        plt.savefig('visualizations/prediction_scatter.png')


if __name__ == "__main__":
    # Load data
    data_path = 'C:\\Users\\Const\\PycharmProjects\\Smart City Air Quality Predictor\\air_quality_data.csv'

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at: {data_path}")

    df = pd.read_csv(data_path)
    print(f"Loaded dataset with shape: {df.shape}")

    # Initialize and train model
    trainer = AQIModelTrainer()
    X, y = trainer.prepare_features(df)

    # Train and evaluate
    metrics = trainer.train_model(X, y)
    print(f"Model Performance Metrics:")
    print(f"RMSE: {metrics['rmse']:.2f}")
    print(f"R2 Score: {metrics['r2']:.2f}")

    # Generate visualizations
    trainer.plot_feature_importance(X)
    trainer.plot_prediction_scatter(metrics['test_actual'], metrics['test_pred'])

    # Save model
    trainer.save_model()
