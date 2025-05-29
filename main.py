import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Read CSV and fix column names
df = pd.read_csv('Car_details-v3.csv', 
                 engine='python',
                 encoding='utf-8',
                 sep=',',
                 on_bad_lines='skip')

# Clean column names
df.columns = df.columns.str.strip()

# Rename columns properly
df.columns = ['name', 'year', 'selling_price', 'km_driven', 'fuel', 'seller_type', 
              'transmission', 'owner', 'mileage', 'engine', 'max_power', 'torque', 'seats']

# Data preprocessing with better cleaning
# Clean numerical columns first
df['year'] = pd.to_numeric(df['year'], errors='coerce')
df['selling_price'] = pd.to_numeric(df['selling_price'], errors='coerce')
df['km_driven'] = pd.to_numeric(df['km_driven'], errors='coerce')

# Clean and extract numerical values from string columns
df['mileage'] = df['mileage'].str.extract('(\d+\.?\d*)').astype(float)
df['engine'] = df['engine'].str.extract('(\d+)').astype(float)
df['max_power'] = df['max_power'].str.extract('(\d+\.?\d*)').astype(float)
df['seats'] = pd.to_numeric(df['seats'], errors='coerce')

# Convert categorical variables
le = LabelEncoder()
categorical_columns = ['fuel', 'seller_type', 'transmission', 'owner']
for col in categorical_columns:
    df[col] = le.fit_transform(df[col].astype(str))

# Drop rows with missing values
df = df.dropna(subset=['year', 'selling_price', 'km_driven', 'fuel', 
                      'seller_type', 'transmission', 'owner', 
                      'mileage', 'engine', 'max_power', 'seats'])

# Print shape before and after cleaning
print("\nShape before cleaning:", len(df))

# Select features for prediction
X = df[['year', 'km_driven', 'fuel', 'seller_type', 'transmission', 
        'owner', 'mileage', 'engine', 'max_power', 'seats']]
y = df['selling_price']

print("Shape after cleaning:", len(df))
print("\nFeature statistics:")
print(X.describe())

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Performance:")
print(f"Mean Squared Error: {mse}")
print(f"R-squared Score: {r2}")

# Feature importance
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': abs(model.coef_)
})
print("\nFeature Importance:")
print(feature_importance.sort_values(by='Importance', ascending=False))

# Example prediction
print("\nExample Prediction:")
example = X_test.iloc[0]
predicted_price = model.predict([example])[0]
actual_price = y_test.iloc[0]
print(f"Predicted Price: {predicted_price:,.2f}")
print(f"Actual Price: {actual_price:,.2f}")

# Enhanced Visualizations
# 1. Interactive Scatter Plot with Plotly
fig = px.scatter(df, x='year', y='selling_price', 
                 size='engine', color='fuel',
                 hover_data=['name', 'km_driven', 'mileage'],
                 title='Car Prices by Year, Engine Size, and Fuel Type')
fig.update_layout(template='plotly_dark')
fig.show()

# 2. Feature Importance Plot
fig_importance = go.Figure(go.Bar(
    x=feature_importance['Importance'],
    y=feature_importance['Feature'],
    orientation='h'
))
fig_importance.update_layout(
    title='Feature Importance in Price Prediction',
    template='plotly_dark',
    xaxis_title='Importance',
    yaxis_title='Features'
)
fig_importance.show()

# 3. Correlation Heatmap with Seaborn
plt.figure(figsize=(12, 8))
sns.heatmap(X.corr(), annot=True, cmap='coolwarm', center=0,
            fmt='.2f', square=True, linewidths=.5)
plt.title('Feature Correlation Matrix', pad=20)
plt.tight_layout()
plt.show()

# 4. Price Distribution with KDE
plt.figure(figsize=(12, 6))
sns.histplot(data=df, x='selling_price', hue='fuel', multiple="stack", bins=50)
sns.kdeplot(data=df, x='selling_price', color='red', linewidth=2)
plt.title('Price Distribution by Fuel Type')
plt.xlabel('Price')
plt.ylabel('Count')
plt.show()

# 5. Interactive Scatter Matrix
fig_scatter = px.scatter_matrix(df,
    dimensions=['selling_price', 'year', 'km_driven', 'mileage', 'engine'],
    color='fuel',
    title='Interactive Scatter Matrix of Numerical Features')
fig_scatter.update_layout(template='plotly_dark')
fig_scatter.show()

# 6. Actual vs Predicted Prices
fig_pred = px.scatter(x=y_test, y=y_pred,
                     labels={'x': 'Actual Price', 'y': 'Predicted Price'},
                     title='Actual vs Predicted Prices')
fig_pred.add_trace(
    go.Scatter(x=[y_test.min(), y_test.max()], 
               y=[y_test.min(), y_test.max()],
               line=dict(color='red', dash='dash'),
               name='Perfect Prediction')
)
fig_pred.update_layout(template='plotly_dark')
fig_pred.show()

# 7. Box Plot of Prices by Transmission and Fuel Type
plt.figure(figsize=(12, 6))
sns.boxplot(data=df, x='transmission', y='selling_price', hue='fuel')
plt.title('Price Distribution by Transmission and Fuel Type')
plt.xticks(rotation=45)
plt.show()

# 8. Interactive Time Series of Average Prices
yearly_avg = df.groupby('year')['selling_price'].mean().reset_index()
fig_time = px.line(yearly_avg, x='year', y='selling_price',
                   title='Average Car Prices Over Years')
fig_time.update_layout(template='plotly_dark')
fig_time.show()
