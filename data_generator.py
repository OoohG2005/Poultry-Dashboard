import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

class PoultrySensorData:
    """Generate realistic poultry house sensor data"""
    
    def __init__(self):
        self.sensor_types = {
            'temperature': {'min': 20, 'max': 35, 'normal': 25},
            'humidity': {'min': 40, 'max': 90, 'normal': 65},
            'ammonia': {'min': 0, 'max': 25, 'normal': 5},
            'co2': {'min': 300, 'max': 1500, 'normal': 600},
            'water_level': {'min': 0, 'max': 100, 'normal': 80},
            'feed_level': {'min': 0, 'max': 100, 'normal': 75},
            'light_intensity': {'min': 0, 'max': 1000, 'normal': 500},
            'air_quality': {'min': 0, 'max': 500, 'normal': 200}
        }
    
    def generate_data(self, days=30, interval_minutes=15):
        """Generate sensor data for specified days"""
        
        # Create timestamps
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days)
        timestamps = pd.date_range(start=start_time, end=end_time, freq=f'{interval_minutes}min')
        
        n = len(timestamps)
        np.random.seed(42)  # For reproducibility
        
        data = {
            'timestamp': timestamps,
            'temperature': self._generate_parameter('temperature', n),
            'humidity': self._generate_parameter('humidity', n),
            'ammonia': self._generate_parameter('ammonia', n),
            'co2': self._generate_parameter('co2', n),
            'water_level': self._generate_parameter('water_level', n),
            'feed_level': self._generate_parameter('feed_level', n),
            'light_intensity': self._generate_parameter('light_intensity', n),
            'air_quality': self._generate_parameter('air_quality', n)
        }
        
        df = pd.DataFrame(data)
        
        # Add disease outbreak simulation
        df = self._add_disease_scenario(df)
        
        # Add risk labels (0=Healthy, 1=Warning, 2=Disease Risk)
        df['risk_label'] = self._calculate_risk_labels(df)
        
        return df
    
    def _generate_parameter(self, param, n):
        """Generate realistic values for a parameter"""
        config = self.sensor_types[param]
        # Add some daily pattern
        base = np.random.normal(config['normal'], (config['max'] - config['min']) / 6, n)
        # Add diurnal variation
        hour_effect = np.sin(np.linspace(0, 2*np.pi*2, n)) * (config['max'] - config['min']) / 8
        values = np.clip(base + hour_effect, config['min'], config['max'])
        return values
    
    def _add_disease_scenario(self, df):
        """Simulate a disease outbreak event"""
        n = len(df)
        # Add disease event at 70% of the way through data
        start_idx = int(n * 0.7)
        end_idx = int(n * 0.75)
        
        # Simulate disease conditions
        df.loc[start_idx:end_idx, 'temperature'] += 5 + np.random.randn(end_idx - start_idx + 1) * 2
        df.loc[start_idx:end_idx, 'humidity'] += 15 + np.random.randn(end_idx - start_idx + 1) * 5
        df.loc[start_idx:end_idx, 'ammonia'] += 10 + np.random.randn(end_idx - start_idx + 1) * 3
        df.loc[start_idx:end_idx, 'co2'] += 300 + np.random.randn(end_idx - start_idx + 1) * 50
        
        return df
    
    def _calculate_risk_labels(self, df):
        """Calculate risk labels based on conditions"""
        labels = np.zeros(len(df))
        
        # Disease risk conditions
        condition1 = (df['temperature'] > 30) & (df['humidity'] > 80) & (df['ammonia'] > 10)
        condition2 = (df['temperature'] > 28) & (df['humidity'] > 85)
        condition3 = (df['ammonia'] > 15) | (df['co2'] > 1200)
        
        labels[condition1] = 2  # Disease Risk
        labels[condition2] = 1  # Warning
        labels[condition3] = 1  # Warning
        
        # Also mark recovery period
        recovery_condition = (df['temperature'] < 25) & (df['humidity'] < 70) & (df['ammonia'] < 5)
        labels[recovery_condition] = 0  # Healthy
        
        return labels.astype(int)

def get_latest_reading(df):
    """Get the most recent sensor reading"""
    return df.iloc[-1].to_dict()

def get_recent_data(df, hours=24):
    """Get data from the last N hours"""
    return df.tail(hours * 4)  # Assuming 15-minute intervals

# Test the generator
if __name__ == "__main__":
    generator = PoultrySensorData()
    df = generator.generate_data(days=30)
    print("✅ Data generated successfully!")
    print(f"📊 Shape: {df.shape}")
    print(f"📈 Columns: {list(df.columns)}")
    print("\n📋 Sample data:")
    print(df.head())
    print("\n📊 Risk distribution:")
    print(df['risk_label'].value_counts())