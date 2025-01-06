# %% [markdown]
# # Data import
# What datasets are you using, later explain the variables a little bit more
# 

# %%
import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder


# %%
print("Current Working Directory:", os.getcwd())


# Load the datasets
mat_data = pd.read_csv('../../student/student-mat.csv', sep=';')
por_data = pd.read_csv('../../student/student-por.csv', sep=';')

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
# # Binary Classification
# Predict whether G3 is above or below a threshold (e.g., 0.5)
# 

# %%
# Create a binary target variable
combined_data['G3_binary'] = (combined_data['G3'] > 0.5).astype(int)

# Define features and target
X_binary = combined_data.drop(columns=['G3', 'G3_binary'])
y_binary = combined_data['G3_binary']

# %%
# Split the data
X_train_binary, X_test_binary, y_train_binary, y_test_binary = train_test_split(
    X_binary, y_binary, test_size=0.2, random_state=42
)


# %%
# Train Random Forest Classifier
rf_binary = RandomForestClassifier(random_state=42)
rf_binary.fit(X_train_binary, y_train_binary)

# %%
# Predict and evaluate
y_pred_binary = rf_binary.predict(X_test_binary)
print("Binary Classification Results:")
print(f"Accuracy: {accuracy_score(y_test_binary, y_pred_binary)}")
print("Classification Report:")
print(classification_report(y_test_binary, y_pred_binary))
print("Confusion Matrix:")
print(confusion_matrix(y_test_binary, y_pred_binary))

# Visualize the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(y_test_binary, y_pred_binary), annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix for Binary Classification")
plt.show()

# %% [markdown]
# # 5-Level Classification
# Categorize G3 into 5 performance levels

# %%
# Create 5 levels for the target variable (G3)
combined_data['G3_5_levels'] = pd.cut(
    combined_data['G3'],
    bins=5,
    labels=[0, 1, 2, 3, 4],
    include_lowest=True
)

# %%
# Define features and target
X_5_levels = combined_data.drop(columns=['G3', 'G3_binary', 'G3_5_levels'])
y_5_levels = combined_data['G3_5_levels']

# %%
# Split the data
X_train_5, X_test_5, y_train_5, y_test_5 = train_test_split(
    X_5_levels, y_5_levels, test_size=0.2, random_state=42
)

# %%
# Train Random Forest Classifier
rf_5_levels = RandomForestClassifier(random_state=42)
rf_5_levels.fit(X_train_5, y_train_5)

# %%
# Predict and evaluate
y_pred_5 = rf_5_levels.predict(X_test_5)
print("5-Level Classification Results:")
print(f"Accuracy: {accuracy_score(y_test_5, y_pred_5)}")
print("Classification Report:")
print(classification_report(y_test_5, y_pred_5))
print("Confusion Matrix:")
print(confusion_matrix(y_test_5, y_pred_5))

# %%
# Visualize the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(y_test_5, y_pred_5), annot=True, fmt="d", cmap="Blues", 
            xticklabels=[0, 1, 2, 3, 4], yticklabels=[0, 1, 2, 3, 4])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix for 5-Level Classification")
plt.show()

# %%
# Feature importance visualization for binary classification
importances_binary = rf_binary.feature_importances_
feature_names_binary = X_binary.columns

# Sorting the feature importances in descending order
sorted_indices_binary = importances_binary.argsort()[::-1]
sorted_importances_binary = importances_binary[sorted_indices_binary]
sorted_feature_names_binary = feature_names_binary[sorted_indices_binary]

plt.figure(figsize=(10, 6))
sns.barplot(x=sorted_importances_binary, y=sorted_feature_names_binary, color="b")
plt.xlabel("Feature Importance")
plt.ylabel("Feature")
plt.title("Feature Importance for Binary Classification")
plt.show()

# Feature importance visualization for 5-level classification
importances_5_levels = rf_5_levels.feature_importances_
feature_names_5_levels = X_5_levels.columns

# Sorting the feature importances in descending order
sorted_indices_5_levels = importances_5_levels.argsort()[::-1]
sorted_importances_5_levels = importances_5_levels[sorted_indices_5_levels]
sorted_feature_names_5_levels = feature_names_5_levels[sorted_indices_5_levels]

plt.figure(figsize=(10, 6))
sns.barplot(x=sorted_importances_5_levels, y=sorted_feature_names_5_levels, color="g")
plt.xlabel("Feature Importance")
plt.ylabel("Feature")
plt.title("Feature Importance for 5-Level Classification")
plt.show()


