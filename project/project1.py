import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import TruncatedSVD

# Load data
df = pd.read_csv("F:/telecom_churn_402.csv", encoding='latin-1')
df=df.dropna()
print(df.head(5))
df = pd.get_dummies(df, drop_first=True)

# Split features & labels
X = df.drop("Churn status", axis=1)
y = df["Churn status"]

# 2D projection using Truncated SVD
svd = TruncatedSVD(n_components=2, random_state=42)
X_2d = svd.fit_transform(X)

# Train-test split (corrected)
X_train, X_test, y_train, y_test, X_train_2d, X_test_2d = train_test_split(
    X, y, X_2d, test_size=0.2, random_state=42, stratify=y
)

# Train SVM on original full features
model = SVC(kernel='rbf', C=1.0, gamma='scale')
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Train SVM on 2D projected data (for boundary plot)
model_2d = SVC(kernel='rbf', C=1.0, gamma='scale')
model_2d.fit(X_train_2d, y_train)

# Decision boundary
x_min, x_max = X_train_2d[:, 0].min() - 1, X_train_2d[:, 0].max() + 1
y_min, y_max = X_train_2d[:, 1].min() - 1, X_train_2d[:, 1].max() + 1

xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 300),
    np.linspace(y_min, y_max, 300)
)

Z = model_2d.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plotting boundary
plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)

scatter = plt.scatter(
    X_train_2d[:, 0],
    X_train_2d[:, 1],
    c=y_train,
    cmap=plt.cm.coolwarm,
    edgecolor='k',
    s=50
)

plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.title("SVM Decision Boundary (TruncatedSVD 2D Projection)")

# Legend
class_labels = ["No", "Yes"]
handles = scatter.legend_elements()[0]
plt.legend(handles, class_labels, title="Churn")

plt.show()



