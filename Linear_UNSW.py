import pandas as pd
import numpy as np
from Linear_Reg import Linear_Regression

df = pd.read_csv("unsw-nb15/versions/1/UNSW_NB15_training-set.csv")

x = df["spkts"].values
y = df["sbytes"].values
dataset = [x, y]

model = Linear_Regression(dataset)
model.OLS()

print("Slope:", model.coefficients[0])
print("Intercept:", model.coefficients[1])
print("RMSE:", model.calculate_error_RMSE())
print("R²:", model.calculate_R2())

model.visualize()
model.visualize3D(zoom_c= 100)