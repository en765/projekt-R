
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Load the datasets
mat_data = pd.read_csv('student-mat.csv', sep=';')
por_data = pd.read_csv('student-por.csv', sep=';')

print(mat_data.describe())
print(por_data.describe())

df = mat_data.copy()
# Assuming 'df' is your DataFrame containing the dataset
features_to_normalize = ['age', 'Medu', 'Fedu', 'traveltime', 'studytime', 'failures',
                         'famrel', 'freetime', 'goout', 'Dalc', 'Walc', 'health',
                         'absences', 'G1', 'G2', 'G3']

missing_features = [feat for feat in features_to_normalize if feat not in df.columns]
if missing_features:
    raise ValueError(f"The following features are missing from the DataFrame: {missing_features}")

# Min-Max Scaling
min_max_scaler = MinMaxScaler()
df_minmax_scaled = df.copy()
df_minmax_scaled[features_to_normalize] = min_max_scaler.fit_transform(df[features_to_normalize])

# Z-Score Standardization
standard_scaler = StandardScaler()
df_standard_scaled = df.copy()
df_standard_scaled[features_to_normalize] = standard_scaler.fit_transform(df[features_to_normalize])

# Print the first few rows to verify the results
print("Original Data (first 5 rows):")
print(df.head())

print("\nMin-Max Scaled Data (first 5 rows):")
print(df_minmax_scaled[features_to_normalize].head())

print("\nZ-Score Standardized Data (first 5 rows):")
print(df_standard_scaled[features_to_normalize].head())


