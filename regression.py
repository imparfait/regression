import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

df = pd.read_csv("rides.csv") 
X = df[['Hour']]
y = df['Duration']
poly = PolynomialFeatures(degree=3)
X_poly = poly.fit_transform(X)
model = LinearRegression()
model.fit(X_poly, y)
y_pred = model.predict(X_poly)
mae = mean_absolute_error(y, y_pred)
mse = mean_squared_error(y, y_pred)
print(f"MAE: {mae:.2f}")
print(f"MSE: {mse:.2f}")
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='blue', label='Фактичні дані')
x_range = np.linspace(0, 23, 500).reshape(-1, 1)
x_range_poly = poly.transform(x_range)
y_range_pred = model.predict(x_range_poly)
plt.plot(x_range, y_range_pred, color='red', label='Крива регресії')
plt.xlabel('Година')
plt.ylabel('Тривалість поїздки (хв)')
plt.title('Поліноміальна регресія тривалості поїздки')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

custom_hours = np.array([[10.5], [0.0], [2.67]])
custom_hours_poly = poly.transform(custom_hours)
predicted_durations = model.predict(custom_hours_poly)

times = ['10:30', '00:00', '02:40']
for time, duration in zip(times, predicted_durations):
    print(f"Тривалість поїздки о {time}: {duration:.2f} хв")
