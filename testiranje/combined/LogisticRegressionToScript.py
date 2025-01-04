# %% [markdown]
# # Data import
# What datasets are you using, later explain the variables a little bit more
# 

# %%
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler, MinMaxScaler


# %%
print("Current Working Directory:", os.getcwd())


# Load the datasets
mat_data = pd.read_csv('student-mat.csv', sep=';')
por_data = pd.read_csv('student-por.csv', sep=';')

# View the first few rows
print(mat_data.head())
print(por_data.head())

# %%
print(mat_data.describe())
print(por_data.describe())

# %% [markdown]
# # Encoding the data
# binary features are encoded using label encoding (0 and 1), 
# while for categorical features one-hot encoding is used.
# As for the numerical features, they are normilized using min-max scaling.

# %%
# combine the data
combined_data = pd.concat([mat_data, por_data], ignore_index=True)

# %%
from sklearn.preprocessing import LabelEncoder

# encoding the binary features
binary_features = ['school', 'sex', 'address', 'famsize', 'Pstatus', 
                   'schoolsup', 'famsup', 'paid', 'activities', 
                   'nursery', 'higher', 'internet', 'romantic']

label_encoder = LabelEncoder()
for feature in binary_features:
    combined_data[feature] = label_encoder.fit_transform(combined_data[feature])


# one-hot endocing the categorical features
categorical_features = ['Mjob', 'Fjob', 'reason', 'guardian']

combined_data = pd.get_dummies(combined_data, columns=categorical_features, drop_first=True)

# normalizing numeric features using min-max scaling
numeric_features = ['age', 'Medu', 'Fedu', 'traveltime', 'studytime', 
                    'failures', 'famrel', 'freetime', 'goout', 'Dalc', 
                    'Walc', 'health', 'absences', 'G1', 'G2', 'G3']

scaler = MinMaxScaler()
combined_data[numeric_features] = scaler.fit_transform(combined_data[numeric_features])



# %% [markdown]
# # Training the Linear Regression Model
# 

# %%
# Selecting the features and targer variable
# For mat_data or por_data individually
X = combined_data.drop(columns=['G3'])  # Drop the target variable
y = combined_data['G3']  # Target variable (G3)


# %%
# Spliting the data = 80% training, 20% testing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt


# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# %%
# Train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# %%
# Make predictions
y_pred = model.predict(X_test)
print(y_pred)

# %%
# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print evaluation metrics
print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")


# %%
# Plot actual vs predicted grades
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.7, color='b', label='Predicted vs Actual')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='r', linestyle='--', label='Perfect Prediction Line')
plt.xlabel("Actual Grades (G3)")
plt.ylabel("Predicted Grades (G3)")
plt.title("Actual vs Predicted Grades")
plt.legend()
plt.show()



# %%
# Residuals Plot
residuals = y_test - y_pred
plt.figure(figsize=(10, 6))
plt.scatter(y_pred, residuals, alpha=0.7, color='g')
plt.hlines(y=0, xmin=y_pred.min(), xmax=y_pred.max(), color='r', linestyle='--')
plt.xlabel("Predicted Grades (G3)")
plt.ylabel("Residuals (Actual - Predicted)")
plt.title("Residuals Plot")
plt.show()

# %%
# Distribution of Residuals
plt.figure(figsize=(10, 6))
plt.hist(residuals, bins=30, color='c', edgecolor='black')
plt.xlabel("Residuals (Actual - Predicted)")
plt.ylabel("Frequency")
plt.title("Distribution of Residuals")
plt.show()


# %%
# Actual vs Predicted Grades Histogram
plt.figure(figsize=(10, 6))
plt.hist(y_test, bins=20, alpha=0.7, label='Actual Grades', color='b')
plt.hist(y_pred, bins=20, alpha=0.7, label='Predicted Grades', color='r')
plt.xlabel("Grades (G3)")
plt.ylabel("Frequency")
plt.title("Actual vs Predicted Grades Histogram")
plt.legend()
plt.show()


# %%
# Convert 'G3' to binary classification: Pass (1) if G3 >= 10, else Fail (0)
combined_data['Pass'] = (combined_data['G3'] >= 0.5).astype(int)
X = combined_data.drop(columns=['G3', 'Pass'])
y = combined_data['Pass']


# %%

# Split data into train and test sets
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# %%

# Train a logistic regression model
from sklearn.linear_model import LogisticRegression

logistic_model = LogisticRegression()
logistic_model.fit(X_train, y_train)


# %%

# Predict on test data
y_pred = logistic_model.predict(X_test)


# %%

# Evaluate the model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

# Display confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", conf_matrix)


# %%

# Visualize the confusion matrix
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Fail', 'Pass'], yticklabels=['Fail', 'Pass'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()



