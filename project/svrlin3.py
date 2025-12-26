import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler

# ----------------------------
# Load dataset
# ----------------------------
df = pd.read_csv("F:/telecom_churn_403.csv", encoding='latin-1')
df = df.dropna()

# ----------------------------
# Encode categorical variables
# ----------------------------
categorical_cols = ["Plan type", "Payment method", "Gender", "Churn status"]
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# ----------------------------
# Feature selection
# ----------------------------
feature_cols = [
    "Previous_Month_Usage_GB",
    "Month1_Usage",
    "Month2_Usage",
    "Month3_Usage",
    "Voice minutes used",
    "Number of SMS sent",
    "Number of international calls",
    "Number of customer service complaints",
    "Monthly data usage (GB)",
    "Average monthly bill",
    "Tenure in months",
    "Age",
    "Customer satisfaction score"
]

# Add encoded columns automatically
encoded_cols = [col for col in df.columns if any(prefix in col for prefix in categorical_cols)]
feature_cols += encoded_cols

# Final X and y
X = df[feature_cols]
y = df["Next_Month_Data_Usage"]

# ----------------------------
# Train-Test Split
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ----------------------------
# Feature Scaling (important for SVR)
# ----------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ----------------------------
# Train SVR
# ----------------------------
svr_model = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1)
svr_model.fit(X_train_scaled, y_train)

# Predictions
y_pred = svr_model.predict(X_test_scaled)

# ----------------------------
# Metrics
# ----------------------------
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mse)

print("\nMetrics for Next Month Data Usage Prediction (SVR):")
print("R2 :", r2)
print("MSE :", mse)
print("RMSE :", rmse)
print("MAE :", mae)
