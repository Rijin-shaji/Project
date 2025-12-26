import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

# Load dataset
df = pd.read_csv("F:/telecom_churn_403.csv", encoding='latin-1')
df = df.dropna()

# Encode categorical variables

categorical_cols = ["Plan type", "Payment method", "Gender", "Churn status"]
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# Feature selection for regression

regression_features = [
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

# Add all encoded categorical columns
encoded_cols = [col for col in df.columns if any(prefix in col for prefix in categorical_cols)]
regression_features += encoded_cols

X_reg = df[regression_features]
y_reg = df["Next_Month_Data_Usage"]


# Standardization for Regression

scaler_reg = StandardScaler()
X_reg_scaled = scaler_reg.fit_transform(X_reg)


# Linear Regression for Next Month Data Usage

lin_model = LinearRegression()
lin_model.fit(X_reg_scaled, y_reg)
y_pred_reg = lin_model.predict(X_reg_scaled)

# Regression metrics
mse = mean_squared_error(y_reg, y_pred_reg)
r2 = r2_score(y_reg, y_pred_reg)
mae = mean_absolute_error(y_reg, y_pred_reg)
rmse = np.sqrt(mse)

print("\n--- Next Month Data Usage Prediction ---")
print("R2 :", r2)
print("MSE :", mse)

# Classification: Predict Customer Churn

churn_cols = [col for col in df.columns if "Churn status" in col]
if len(churn_cols) != 1:
    raise ValueError(f"Expected exactly 1 encoded churn column, found: {churn_cols}")
churn_target = churn_cols[0]

# Features for classification (exclude churn column)
classification_features = [col for col in X_reg.columns if "Churn status" not in col]
X_clf = df[classification_features]
y_clf = df[churn_target]

# Standardization for Classification

scaler_clf = StandardScaler()
X_clf_scaled = scaler_clf.fit_transform(X_clf)

# Logistic Regression for churn

clf_model = LogisticRegression(max_iter=1000)
clf_model.fit(X_clf_scaled, y_clf)
y_pred_clf = clf_model.predict(X_clf_scaled)

# Classification metrics
accuracy = accuracy_score(y_clf, y_pred_clf)
conf_matrix = confusion_matrix(y_clf, y_pred_clf)

print("\n--- Customer Churn Classification ---")
print("Churn column used:", churn_target)
print("Accuracy :", accuracy)
print("Confusion Matrix:\n", conf_matrix)


# Identify behavioral factors influencing churn

coef_df = pd.DataFrame({
    "Feature": classification_features,
    "Coefficient": clf_model.coef_[0]
})

# Sort by absolute impact
coef_df["Abs_Coefficient"] = coef_df["Coefficient"].abs()
coef_df = coef_df.sort_values(by="Abs_Coefficient", ascending=False)

print("\n--- Features Strongly Influencing Churn ---")
print(coef_df[["Feature", "Coefficient"]].head(10))  # top 10 features



