import pandas as pd
import statsmodels.api as sm

# Sample dataset
data = {
    'Age': [22, 25, 47, 52, 46, 56, 55, 60, 62, 61],
    'Salary': [15000, 29000, 48000, 60000, 52000, 61000, 57000, 65000, 70000, 73000],
    'Purchased': [0, 0, 1, 1, 1, 1, 1, 1, 1, 1]
}
df = pd.DataFrame(data)

# Define independent and dependent variables
X = df[['Age', 'Salary']]
X = sm.add_constant(X)  # Add constant for intercept
y = df['Purchased']

# Fit the logistic regression model
model = sm.Logit(y, X)
result = model.fit()

# Display summary
print(result.summary())

# Predict on new data
new_data = pd.DataFrame({'const': 1, 'Age': [30], 'Salary': [35000]})
prediction = result.predict(new_data)
print("Probability of Purchase:", prediction[0])
