# Import libraries
import pandas as pd
from sklearn.discriminant_analysis import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv("F:/telecom_churn_402.csv", encoding='latin-1')
df=df.dropna()
print(df.head(5))
df = pd.get_dummies(df, drop_first=True)

# Split features & labels
X = df.drop("Churn status", axis=1)
y = df["Churn status"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Initialize Random Forest Classifier
rf = RandomForestClassifier(n_estimators=100,max_depth=10,min_samples_split=4,min_samples_leaf=2,max_features='sqrt',bootstrap=True,random_state=42)

# Train the model
rf.fit(X_train, y_train)

# Make predictions
y_pred = rf.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# Initialize Logistic Regression
lr = LogisticRegression(max_iter=1000, random_state=42) 
# Train the Logistic Regression model
lr.fit(X_train, y_train)
# Make predictions with Logistic Regression
y_pred_lr = lr.predict(X_test)
# Evaluate the Logistic Regression model
accuracy_lr = accuracy_score(y_test, y_pred_lr)
print(f"Logistic Regression Accuracy: {accuracy_lr}")
print("\nLogistic Regression Classification Report:")
print(classification_report(y_test, y_pred_lr))

# naive bayes
nb = GaussianNB()
nb.fit(X_train, y_train)
y_pred_nb = nb.predict(X_test)
accuracy_nb = accuracy_score(y_test, y_pred_nb)
print(f"Naive Bayes Accuracy: {accuracy_nb}")
print("\nNaive Bayes Classification Report:")
print(classification_report(y_test, y_pred_nb))


#decision tree
dt = DecisionTreeClassifier(criterion='entropy', random_state=42)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)
accuracy_dt = accuracy_score(y_test, y_pred_dt)
print(f"Decision Tree Accuracy: {accuracy_dt}")
print("\nDecision Tree Classification Report:")
print(classification_report(y_test, y_pred_dt))


#which model is best
accuracies = {
    "Random Forest": accuracy,
    "Logistic ": accuracy_lr,
    "Naive Bayes": accuracy_nb,
    "Decision Tree": accuracy_dt
}
best_model = max(accuracies, key=accuracies.get)
print(f"The best model is: {best_model} with an accuracy of {accuracies[best_model]}")