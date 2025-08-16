import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Create a dictionary with the column names and data
data = {
    'Experience': [1, 2, 3, 5, 6],
    'Salary': [40, 50, 55, 75, 80]
}

# Create the DataFrame from the dictionary
df = pd.DataFrame(data)

# Print the DataFrame
print(df)


X = df['Experience'].values.reshape(-1, 1) # Assign the 'Experience' column to X and reshape for model
y = df['Salary'].values # Assign the 'Salary' column to y

X_1 = np.c_[np.ones(len(X)),X]
beta_c = np.linalg.inv(X_1.T @ X_1) @ X_1.T @ y

intercept = beta_c[0]
slope = beta_c[1]

#

#Print results
print(f"Intercept of Beta0: {beta_c[0]:.2f}")
print(f"The slope B1 : {slope:.2f}")
print(f"Final Equation: Salary = {intercept:.2f} + {slope:.2f} * Experience")

#View results
plt.figure(figsize=(10,6))
plt.scatter(df['Experience'],df['Salary'],color='blue',label='Real data', s=100)

predictions = X_1 @ beta_c
plt.plot(df['Experience'],predictions,color='red', linewidth=3,label='Regression line')

plt.title('Salary vs Years of Experience',fontsize=16)
plt.xlabel('Years of Experience')
plt.ylabel('Salary(thousands of euros)',fontsize=12)
plt.legend()
plt.grid(True)
plt.show()
