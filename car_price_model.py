import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
from report_generator import ModelReportGenerator

def preprocess_data(df):
    # Clean column names
    df.columns = [col.strip().lower() for col in df.columns]
    
    # Clean numerical columns
    df['year'] = pd.to_numeric(df['year'], errors='coerce')
    df['selling_price'] = pd.to_numeric(df['selling_price'], errors='coerce')
    df['km_driven'] = pd.to_numeric(df['km_driven'], errors='coerce')
    
    # Extract numerical values from string columns
    df['mileage'] = df['mileage'].str.extract('(\d+\.?\d*)').astype(float)
    df['engine'] = df['engine'].str.extract('(\d+)').astype(float)
    df['max_power'] = df['max_power'].str.extract('(\d+\.?\d*)').astype(float)
    df['seats'] = pd.to_numeric(df['seats'], errors='coerce')
    
    # Feature engineering
    df['car_age'] = 2024 - df['year']
    
    return df

def train_models():
    # Read and preprocess data
    df = pd.read_csv('Car_details-v3.csv', engine='python', encoding='utf-8', sep=',', on_bad_lines='skip')
    df = preprocess_data(df)
    
    # Convert categorical variables
    le = LabelEncoder()
    categorical_columns = ['fuel', 'seller_type', 'transmission', 'owner']
    for col in categorical_columns:
        df[col] = le.fit_transform(df[col].astype(str))
    
    # Select features
    features = ['car_age', 'km_driven', 'fuel', 'seller_type', 'transmission',
               'owner', 'mileage', 'engine', 'max_power', 'seats']
    
    # Drop rows with missing values
    df = df.dropna(subset=features + ['selling_price'])
    
    X = df[features]
    y = df['selling_price']
    
    # Split and scale data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train models
    models = {
        'Linear': LinearRegression(),
        'Ridge': Ridge(),
        'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42)
    }
    
    best_model = None
    best_score = -float('inf')
    
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        score = r2_score(y_test, model.predict(X_test_scaled))
        print(f"{name} R2 Score: {score:.4f}")
        
        if score > best_score:
            best_score = score
            best_model = model
    
    # Feature importance
    feature_importance = None
    if hasattr(best_model, 'feature_importances_'):
        feature_importance = best_model.feature_importances_
    elif hasattr(best_model, 'coef_'):
        feature_importance = best_model.coef_
    
    # Predictions
    y_pred = best_model.predict(X_test_scaled)
    
    # Report generation
    feature_importance_df = pd.DataFrame({
        'Feature': X.columns,
        'Importance': best_model.feature_importances_ if hasattr(best_model, 'feature_importances_') 
                     else abs(best_model.coef_)
    }).sort_values('Importance', ascending=False)
    
    report_gen = ModelReportGenerator()
    report_path = report_gen.generate_report(
        model=best_model,
        X_train=X_train_scaled,
        X_test=X_test_scaled,
        y_train=y_train,
        y_test=y_test,
        y_pred=y_pred,
        feature_importance=feature_importance_df
    )
    
    print(f"\nDetailed report generated: {report_path}")
    
    # Save best model and scaler
    joblib.dump(best_model, 'best_car_price_model.joblib')
    joblib.dump(scaler, 'scaler.joblib')
    
    return best_model, scaler

if __name__ == "__main__":
    train_models()
