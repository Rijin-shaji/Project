import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler


df = pd.read_csv("F:/data.csv")
x = df[['bedrooms','bathrooms','sqft_living','sqft_lot','floors','sqft_above','sqft_basement']]
y = df['price']


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)


scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)


model = LinearRegression()
model.fit(x_train_scaled, y_train) 


y_pred = model.predict(x_test_scaled)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("R2 : ", r2)
print(f"MSE : {mse}")
print(f"RMSE : {rmse}")



coef = pd.Series(model.coef_, index=x.columns)
print(coef)


new_X = pd.DataFrame({
    'bedrooms': [3],
    'bathrooms': [2],
    'sqft_living': [2000],
    'sqft_lot': [5000],
    'floors': [1],
    'sqft_above': [1500],
    'sqft_basement': [500]
})


new_X_scaled = scaler.transform(new_X)
predicted_price = model.predict(new_X_scaled)
print("Predicted price for new input:", predicted_price)
