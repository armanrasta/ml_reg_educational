import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

def predict_price_with_confidence(features):
    """
    Predict car price with confidence interval
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
