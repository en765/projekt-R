import pandas as pd
import os

print("Current Working Directory:", os.getcwd())


# Load the datasets
mat_data = pd.read_csv('student-mat.csv', sep=';')
por_data = pd.read_csv('student-por.csv', sep=';')

# View the first few rows
print(mat_data.head())
print(por_data.head())
