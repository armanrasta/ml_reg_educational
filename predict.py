"""
Car Price Prediction Interface

This module provides functions to make predictions using the trained
car price prediction model.

Functions:
    predict_price_with_confidence: Predicts car price with confidence intervals
"""

import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

def predict_price_with_confidence(features):
    """
    Predict car price with confidence interval.

    Args:
        features (list): List of feature values in the correct order:
            [year, km_driven, fuel, seller_type, transmission, owner,
             mileage, engine, max_power, seats]

    Returns:
        dict: Prediction results including:
            - predicted_price: The predicted car price
            - confidence_interval: (min_price, max_price) if available
            - confidence_score: Confidence score if available
    """
    model = joblib.load('best_car_price_model.joblib')
    scaler = joblib.load('scaler.joblib')
    
    # Scale features
    features_scaled = scaler.transform([features])
    
    # Make prediction
    prediction = model.predict(features_scaled)[0]
    
    # For Random Forest, calculate confidence using std of tree predictions
    if hasattr(model, 'estimators_'):
        predictions = np.array([tree.predict(features_scaled)[0] 
                              for tree in model.estimators_])
        confidence = np.std(predictions)
        return {
            'predicted_price': prediction,
            'confidence_interval': (prediction - 2*confidence, 
                                  prediction + 2*confidence),
            'confidence_score': 1 - (confidence / prediction)
        }
    
    return {'predicted_price': prediction}
