#!/usr/bin/env python3
"""
Poultry Disease Prediction Dashboard - Launcher
Run this file to start the dashboard
"""

import subprocess
import sys
import os
import webbrowser
import time

def check_dependencies():
    """Check if required packages are installed"""
    required = ['streamlit', 'pandas', 'numpy', 'sklearn', 'plotly', 'matplotlib', 'seaborn', 'joblib']
    missing = []
    
    for package in required:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing.append(package)
    
    if missing:
        print("📦 Installing missing dependencies...")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', *missing])
        print("✅ Dependencies installed!")
    else:
        print("✅ All dependencies are installed!")

def train_model():
    """Train the AI model first"""
    print("🤖 Training AI model...")
    try:
        import model
        from data_generator import PoultrySensorData
        
        generator = PoultrySensorData()
        df = generator.generate_data(days=30)
        
        predictor = model.PoultryDiseasePredictor()
        predictor.train(df)
        print("✅ Model trained successfully!")
    except Exception as e:
        print(f"⚠️ Error training model: {e}")
        print("Continuing with existing model if available...")

def main():
    """Main function to launch dashboard"""
    print("🐔 Starting Poultry Disease Prediction Dashboard")
    print("=" * 50)
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("⚠️ Python 3.8 or higher is required!")
        sys.exit(1)
    
    # Check and install dependencies
    check_dependencies()
    
    # Train model
    train_model()
    
    # Open browser
    print("🌐 Opening browser...")
    time.sleep(2)
    webbrowser.open("http://localhost:8501")
    
    # Run streamlit
    print("🚀 Launching dashboard at http://localhost:8501")
    print("Press Ctrl+C to stop")
    print("=" * 50)
    
    try:
        subprocess.run([sys.executable, '-m', 'streamlit', 'run', 'app.py'])
    except KeyboardInterrupt:
        print("\n👋 Shutting down...")
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\nTry running manually:")
        print("  streamlit run app.py")

if __name__ == "__main__":
    main()