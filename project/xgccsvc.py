import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load dataset
df = pd.read_csv("F:/uber_full_dataset_50000.csv")
df=df.dropna()

# Features and target
X = df.drop('is_driver_fraud', axis=1)
Y = df['is_driver_fraud']

# Encode categorical columns
cat_cols = X.select_dtypes(include=['object']).columns
X = pd.get_dummies(X, columns=cat_cols, drop_first=True)

# Train-test split
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42, stratify=Y
)

# ----------------------------
# Standardization for SVC
# ----------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ----------------------------
# SVC Classifier
# ----------------------------
model = SVC(
    kernel='rbf',
    C=1,
    gamma='scale',
    probability=True  # enables predict_proba()
)

model.fit(X_train, Y_train)

# Predict & evaluate
Y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(Y_test, Y_pred))
print("Classification Report:\n", classification_report(Y_test, Y_pred))

# Confusion Matrix
cm = confusion_matrix(Y_test, Y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[0,1], yticklabels=[0,1])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix (SVC)")
plt.show()

# ----------------------------------------------------
# Feature importance does NOT exist in SVC (RBFR Kernel)
# ----------------------------------------------------
print("\nSVC (RBF kernel) does NOT provide feature_importances_.")
print("If you want feature importance, use:")
print("1. SVC(kernel='linear') OR")
print("2. Permutation Importance OR")
print("3. SHAP values")


