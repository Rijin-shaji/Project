import pandas as pd
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.metrics import mean_absolute_error

df = pd.read_csv("F:/data.csv")
sleep=df[['bedrooms','bathrooms','sqft_living','sqft_lot','floors','sqft_above','sqft_basement']]
prd=df['price']
#print(df)
# plt.xlabel('sleep')
# plt.ylabel('prd')
# plt.scatter(sleep,prd,color='Red',marker='+')
# plt.show()
# new_df=df.drop('prd',axis='columns')
# print(new_df)
# prd1=prd
reg=linear_model.LinearRegression()
reg.fit(sleep,prd)
#print(reg.predict([[6]]))
print("Slope : ",reg.coef_)
print("Intercept :",reg.intercept_)
# plt.xlabel('sleep')
# plt.ylabel('prd')
# plt.scatter(sleep,prd,color='Red',marker='+')
# plt.plot(df[["Engine_Size (L)"]],reg.predict(sleep))
# plt.show()
mse= mean_squared_error(prd,reg.predict(sleep))
print("R2 : ",r2_score(prd,reg.predict(sleep)))
print(f"MSE : {mse}")
rmse=np.sqrt(mse)
print(f"RMSE : {rmse}")
mae=mean_absolute_error(prd,reg.predict(sleep))
print(f"MAE : {mae}")
mbe=np.mean(reg.predict(sleep)-prd)
print(f"MBE : {mbe}")