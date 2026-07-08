import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os
from datetime import datetime
import json

class PoultryDiseasePredictor:
    """AI Model for predicting poultry disease risks"""
    
    def __init__(self):
        self.model = None
        self.feature_columns = [
            'temperature', 'humidity', 'ammonia', 'co2',
            'water_level', 'feed_level', 'light_intensity', 'air_quality'
        ]
        self.model_path = 'poultry_model.pkl'
        self.metadata_path = 'model_metadata.json'
        self.feature_importance = None
        self.accuracy = None
        
    def train(self, df):
        """Train the AI model on historical data"""
        
        print("🤖 Training AI Model...")
        
        # Prepare features and labels
        X = df[self.feature_columns]
        y = df['risk_label']
        
        # Check if we have all three classes
        unique_classes = np.unique(y)
        print(f"📊 Classes found in data: {unique_classes}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train Random Forest model
        self.model = RandomForestClassifier(
            n_estimators=150,
            max_depth=12,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            class_weight='balanced'
        )
        
        self.model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = self.model.predict(X_test)
        self.accuracy = accuracy_score(y_test, y_pred)
        
        # Get feature importance
        self.feature_importance = dict(
            zip(self.feature_columns, self.model.feature_importances_)
        )
        
        print(f"✅ Model trained with accuracy: {self.accuracy:.2%}")
        
        # Only show classification report if we have all classes
        try:
            # Get unique classes in test data
            test_classes = np.unique(y_test)
            print(f"📊 Classes in test data: {test_classes}")
            
            # Create labels based on what's in the data
            labels = {0: 'Healthy', 1: 'Warning', 2: 'Disease Risk'}
            target_names = [labels[i] for i in sorted(test_classes)]
            
            print("\n📊 Classification Report:")
            print(classification_report(y_test, y_pred, target_names=target_names))
            
            # Also show confusion matrix
            print("\n📊 Confusion Matrix:")
            print(confusion_matrix(y_test, y_pred))
            
        except Exception as e:
            print(f"⚠️ Could not generate classification report: {e}")
            print("✅ Model training completed successfully anyway!")
        
        # Save model
        self._save_model(df)
        
        return self.accuracy
    
    def _save_model(self, df):
        """Save model and metadata"""
        # Save model
        joblib.dump(self.model, self.model_path)
        
        # Save metadata
        metadata = {
            'training_date': datetime.now().isoformat(),
            'accuracy': float(self.accuracy),
            'features': self.feature_columns,
            'feature_importance': self.feature_importance,
            'samples_used': len(df),
            'risk_distribution': df['risk_label'].value_counts().to_dict()
        }
        
        with open(self.metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"💾 Model saved to {self.model_path}")
        print(f"📝 Metadata saved to {self.metadata_path}")
    
    def load_model(self):
        """Load trained model"""
        if os.path.exists(self.model_path):
            self.model = joblib.load(self.model_path)
            
            # Load metadata if exists
            if os.path.exists(self.metadata_path):
                with open(self.metadata_path, 'r') as f:
                    metadata = json.load(f)
                    self.accuracy = metadata.get('accuracy')
                    self.feature_importance = metadata.get('feature_importance')
            
            print("✅ Model loaded successfully!")
            return True
        else:
            print("⚠️ No trained model found. Please train first.")
            return False
    
    def predict(self, data):
        """Make prediction from sensor data"""
        if self.model is None:
            if not self.load_model():
                return None
        
        # Convert to DataFrame if needed
        if isinstance(data, dict):
            df = pd.DataFrame([data])
        else:
            df = data
        
        # Select features
        X = df[self.feature_columns]
        
        # Make prediction
        prediction = self.model.predict(X)[0]
        probabilities = self.model.predict_proba(X)[0]
        
        # Map to labels
        labels = {0: 'Healthy', 1: 'Warning', 2: 'Disease Risk'}
        status = labels.get(prediction, 'Unknown')
        confidence = max(probabilities) * 100
        
        # Ensure we have all probability values
        probs_dict = {}
        for i in range(3):  # 0, 1, 2
            label = labels.get(i, f'Class {i}')
            probs_dict[label] = float(probabilities[i] * 100 if i < len(probabilities) else 0)
        
        return {
            'status': status,
            'status_code': prediction,
            'confidence': confidence,
            'probabilities': probs_dict,
            'timestamp': datetime.now().isoformat()
        }
    
    def predict_batch(self, df):
        """Make predictions for multiple readings"""
        if self.model is None:
            if not self.load_model():
                return None
        
        X = df[self.feature_columns]
        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)
        
        labels = {0: 'Healthy', 1: 'Warning', 2: 'Disease Risk'}
        results = []
        for pred, prob in zip(predictions, probabilities):
            probs_dict = {}
            for i in range(3):
                label = labels.get(i, f'Class {i}')
                probs_dict[label] = float(prob[i] * 100 if i < len(prob) else 0)
            
            results.append({
                'status': labels.get(pred, 'Unknown'),
                'confidence': max(prob) * 100,
                'probabilities': probs_dict
            })
        
        return results
    
    def get_model_info(self):
        """Get model information for display"""
        return {
            'accuracy': self.accuracy,
            'feature_importance': self.feature_importance,
            'features': self.feature_columns
        }

# Test the model
if __name__ == "__main__":
    from data_generator import PoultrySensorData
    
    # Generate data
    generator = PoultrySensorData()
    df = generator.generate_data(days=30)
    
    # Train model
    predictor = PoultryDiseasePredictor()
    predictor.train(df)
    
    # Test prediction on latest data
    latest = df.iloc[-1].to_dict()
    result = predictor.predict(latest)
    
    print("\n🔮 Prediction on latest data:")
    print(f"Status: {result['status']}")
    print(f"Confidence: {result['confidence']:.1f}%")
    print(f"Probabilities: {result['probabilities']}")
    
    print("\n🔑 Feature Importance:")
    if predictor.feature_importance:
        for feature, importance in sorted(
            predictor.feature_importance.items(), 
            key=lambda x: x[1], 
            reverse=True
        ):
            print(f"  {feature}: {importance:.2%}")