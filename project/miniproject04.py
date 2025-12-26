import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


df = pd.read_csv("F:/data.csv")
X = df[['bedrooms','bathrooms','sqft_living','sqft_lot','floors','sqft_above','sqft_basement']]
y = df['price']


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


reg = linear_model.LinearRegression()
reg.fit(X_scaled, y)
y_pred = reg.predict(X_scaled)


print("Slope : ", reg.coef_)
print("Intercept :", reg.intercept_)

mse = mean_squared_error(y, y_pred)
rmse = np.sqrt(mse)

print("R2 : ", r2_score(y, y_pred))
print(f"MSE : {mse}")
print(f"RMSE : {rmse}")

