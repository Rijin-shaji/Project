import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# 1. Load dataset
# -----------------------------
df = pd.read_csv("F:/uber_full_dataset_50000.csv")

# -----------------------------
# 2. Convert datetime & extract features
# -----------------------------
df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])
df['hour'] = df['pickup_datetime'].dt.hour
df['day_of_week'] = df['pickup_datetime'].dt.dayofweek  
df['day'] = df['pickup_datetime'].dt.day
df['month'] = df['pickup_datetime'].dt.month

# -----------------------------
# 3. Aggregate rides per driver-hour
# -----------------------------
df_agg = df.groupby(['Driver_Id','hour','day_of_week']).agg({
    'passenger_count':'sum',
    'Ride_Distance':'sum',
    'fare_amount':'sum',
    'Total_tips_driver_received':'sum',
    'Avg_VTAT':'mean',
    'Avg_CTAT':'mean',
    'Cancelled_Rides_by_Driver':'sum',
    'accept_to_cancel_ratio':'mean',
    'traffic_density':'first' 
}).reset_index()

# Target: number of rides
df_agg['num_rides'] = df.groupby(['Driver_Id','hour','day_of_week']).size().values

# -----------------------------
# 4. Features and target
# -----------------------------
X = df_agg.drop('num_rides', axis=1)
y = df_agg['num_rides']

# Encode categorical variables
cat_cols = X.select_dtypes(include=['object']).columns
X = pd.get_dummies(X, columns=cat_cols, drop_first=True)

# -----------------------------
# 5. Train-test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# 6. XGBoost Regressor
# -----------------------------
model = xgb.XGBRegressor(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

model.fit(X_train, y_train)

# -----------------------------
# 7. Predict & evaluate
# -----------------------------
y_pred = model.predict(X_test)
mse=mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print("MSE:", mse)
print("RMSE:", rmse)
print("R2 Score:", r2_score(y_test, y_pred))

# -----------------------------
# 8. Feature Importance (Top 20)
# -----------------------------
importance = model.feature_importances_
features = X.columns
indices = np.argsort(importance)[-20:] 

plt.figure(figsize=(10,8))
plt.barh(range(len(indices)), importance[indices], align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel("Feature Importance")
plt.title("Top 20 Feature Importances")
plt.show()

# -----------------------------
# 9. Predict rides for a specific driver & datetime
# -----------------------------

dt = pd.to_datetime('2025-05-26 13:05:00')

new_data = pd.DataFrame({
    'Driver_Id':[2244],
    'hour':[dt.hour],
    'day_of_week':[dt.dayofweek],
    'day':[dt.day],
    'month':[dt.month],
    'passenger_count':[1],
    'Ride_Distance':[20],
    'fare_amount':[1700],
    'Total_tips_driver_received':[0],
    'Avg_VTAT':[20],
    'Avg_CTAT':[0],
    'Cancelled_Rides_by_Driver':[5],
    'accept_to_cancel_ratio':[0.5],
    'traffic_density_High':[5]  
})

# Ensure columns match training data
missing_cols = set(X_train.columns) - set(new_data.columns)
for col in missing_cols:
    new_data[col] = 0
new_data = new_data[X_train.columns]

predicted_rides = model.predict(new_data)
print("Will he get a ride? ", "Yes" if predicted_rides[0]>=1 else "No")
