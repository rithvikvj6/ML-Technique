import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
data = {
    'YearsExperience': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Salary': [25000, 30000, 35000, 40000, 45000,
               50000, 55000, 60000, 65000, 70000]
}
df = pd.DataFrame(data)
print("Dataset:")
print(df)
X = df[['YearsExperience']]
y = df['Salary']
model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)
print("\nPredicted Salaries:")
print(y_pred)
print("\nMean Squared Error:",
      mean_squared_error(y, y_pred))
print("R2 Score:", r2_score(y, y_pred))
new_salary = model.predict([[11]])
print("\nSalary for 11 years experience:", new_salary[0])
plt.scatter(X, y, color='blue', label='Actual Salary')
plt.plot(X, y_pred, color='red', label='Regression Line')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.title('Linear Regression - Salary Prediction')
plt.legend()
plt.show()