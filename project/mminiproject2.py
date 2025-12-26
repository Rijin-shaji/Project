import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import  ConfusionMatrixDisplay, accuracy_score, confusion_matrix, classification_report, f1_score, precision_score, recall_score
from scipy.special import expit

#reading data
df = pd.read_csv("F:/framingham.csv")
#cleaning data
df = df.dropna()
print(df.head())

X = df[['male','age','currentSmoker', 'cigsPerDay','BPMeds','prevalentStroke','prevalentHyp','diabetes', 'totChol', 'sysBP','diaBP','BMI','heartRate', 'glucose']]  
y = df['TenYearCHD']

#splitting data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train) 

#making predictions
y_pred = model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print(f"\nConfusion Matrix: {cm}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

#predicting new data
new_data = np.array([[4,27,1,20,0,0,1,0,240,140,90,28.5,80,100]])
prediction = model.predict(new_data)
print("\nPredicted (1=Pass, 0=Fail):", prediction[0])
print(model.predict_proba(X_test))
print(model.score(X_test,y_test))
y_proba = model.predict_proba(X_test)

#printing model details
print(f"Model Probabilities: {y_proba}")
print("Model Score : ", model.score(X_test,y_test))
prec = precision_score(y_train, model.predict(X_train), pos_label=1)
rec = recall_score(y_train, model.predict(X_train), pos_label=1)
f1 = f1_score(y_train, model.predict(X_train), pos_label=1)
print(f"Precision (age): {prec}")
print(f"Recall (age): {rec}")
print(f"F1-score (age): {f1}")
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")

#standard scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train_scaled, X_test_scaled, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
model.fit(X_train_scaled, y_train)
y_pred_scaled = model.predict(X_test_scaled)
print("Accuracy (after scaling):", accuracy_score(y_test, y_pred_scaled))

# Visualizing the confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['X','Y'])
disp.plot(cmap='summer')
plt.show()

# Plotting the sigmoid function
plt.figure(figsize=(10, 6))

# Get the age feature and scale it for visualization
age_values = X['age'].values.reshape(-1, 1)
age_scaled = scaler.fit_transform(age_values)

# Get probabilities for actual data points
y_probabilities = model.predict_proba(X_scaled)[:, 1]

# Create smooth sigmoid curve
x = np.linspace(-4, 4, 200)
y = expit(x)

# Plot both the sigmoid curve and actual data points
plt.scatter(age_scaled, y_probabilities, color='blue', alpha=0.5, label='Actual Data Points')
plt.plot(x, y, 'r-', linewidth=2, label='Sigmoid Curve')

# Customize the plot
plt.axhline(y=0.5, color='green', linestyle='--', alpha=0.5)
plt.axvline(x=0, color='green', linestyle='--', alpha=0.5)
plt.grid(True, alpha=0.3)
plt.title('Sigmoid Curve with Actual Data Points')
plt.xlabel('Standardized Age')
plt.ylabel('Probability of CHD')
plt.legend()
plt.ylim(-0.1, 1.1)
plt.tight_layout()
plt.show()

