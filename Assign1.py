import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
data = {
    'year': [1970, 1971, 1972, 1973, 1974, 1975, 1976, 1977, 1978, 1979, 1980, 1990, 2000, 2010],
    'per_capita_income': [3399.29, 3768.29, 4251.17, 4804.46, 5576.51, 5998.14, 7062.13, 7100.12, 7247.96, 7602.91, 8355.96, 16838.67, 18987.38, 38420.52]
}
df = pd.DataFrame(data)
X = df[['year']]  
y = df['per_capita_income']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
reg = LinearRegression()
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)
print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred):.2f}")
print(f"R² Score: {r2_score(y_test, y_pred):.4f}")
future_year = np.array([[2030]])
predicted_income = reg.predict(future_year)
print(f"\nPredicted Per Capita Income for year 2030: ${predicted_income[0]:.2f}")
plt.scatter(X, y, color='red', label='Actual Data')
plt.plot(X, reg.predict(X), color='green', linewidth=2, label='ML Linear Regression Line')
plt.xlabel('Year')
plt.ylabel('Per Capita Income (US$)')
plt.title('Canada Per Capita Income ML Model')
plt.legend()
plt.show()