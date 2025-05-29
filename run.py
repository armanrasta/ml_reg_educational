import pandas as pd
import joblib
from car_price_model import train_models
from predict import predict_price_with_confidence

def main():
    print("Car Price Prediction System")
    print("1. Train new model")
    print("2. Make prediction")
    choice = input("Enter your choice (1/2): ")

    if choice == '1':
        print("\nTraining new model...")
        model, scaler = train_models()
        print("Training complete! Model and scaler saved.")
        
    elif choice == '2':
        print("\nEnter car details:")
        features = {
            'year': int(input("Year: ")),
            'km_driven': float(input("Kilometers driven: ")),
            'fuel': int(input("Fuel type (0:Petrol, 1:Diesel, 2:CNG, 3:LPG): ")),
            'seller_type': int(input("Seller type (0:Individual, 1:Dealer): ")),
            'transmission': int(input("Transmission (0:Manual, 1:Automatic): ")),
            'owner': int(input("Owner (0:First, 1:Second, 2:Third, 3:Fourth+): ")),
            'mileage': float(input("Mileage (kmpl): ")),
            'engine': float(input("Engine (CC): ")),
            'max_power': float(input("Max Power (bhp): ")),
            'seats': int(input("Seats: "))
        }
        
        # Convert to list in correct order
        feature_list = [
            features['year'], features['km_driven'], features['fuel'],
            features['seller_type'], features['transmission'], features['owner'],
            features['mileage'], features['engine'], features['max_power'],
            features['seats']
        ]
        
        result = predict_price_with_confidence(feature_list)
        print("\nPrediction Results:")
        print(f"Predicted Price: ${result['predicted_price']:,.2f}")
        if 'confidence_interval' in result:
            print(f"Confidence Interval: ${result['confidence_interval'][0]:,.2f} to ${result['confidence_interval'][1]:,.2f}")
            print(f"Confidence Score: {result['confidence_score']:.2%}")

if __name__ == "__main__":
    main()
