import pandas as pd

# 1. Load the dataset into a DataFrame
df = pd.read_csv('C:\\Users\\Putti Sampath\\Sampath\\AI and ML\\ml\\Python Script mini project 2\\churnguard_data.csv')

# 2. Print the shape of the dataset (rows, columns)
print(df.shape)

# 3. Print the first 5 rows
print(df.head())

# 4. Print column names and data types using .info()
print(df.info())

# 5. Print the count of missing values in each column
print(df.isnull().sum())

# 6. Print the number of duplicate rows
print(df.duplicated().sum())

# 7. Print the value counts of the Churn column
print(df['Churn'].value_counts())

# 8. Print the unique values in the Contract column
print(df['Contract'].unique())