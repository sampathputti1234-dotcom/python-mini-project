import pandas as pd 
df = pd.read_csv('C:\\Users\\Putti Sampath\\Sampath\\AI and ML\\ml\\Python Script mini project 2\\churnguard_data.csv')


df.drop(columns = ['customerID'] , inplace = True)

df.drop_duplicates()

df.shape

df['gender'].str.strip()

df['PaymentMethod'].str.strip()

df['Churn'].str.strip().str.title()

df['PhoneService'].str.strip().str.title()

df['PaperlessBilling'].str.strip().str.title()

contract_map = {
  'month to month': 'Month-to-month',
  'month-to-month':'Month-to-month',
  'monthly': 'Month-to-month',
  '1 year': 'One year',
  'one year': 'One year',
  '2 year': 'Two year',
  'two year':'Two year'
}
df['Contract'] = df['Contract'].str.lower().str.strip().map(contract_map).fillna(df['Contract'])


internetservice_map = {

 'dsl': 'DSL',
    'fibre optic': 'Fiber optic',
    'fiberoptic': 'Fiber optic',
    'fiber optic': 'Fiber optic',
    'none': 'No',
    'no': 'No'
}

df['InternetService'] = df['InternetService'].str.lower().str.strip().map(internetservice_map).fillna(df['InternetService'])

df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors = 'coerce')

df = df[df['tenure'] > 0]

df = df[(df['MonthlyCharges'] >= 10) & (df['MonthlyCharges'] <= 200)]

df['MonthlyCharges'] = df['MonthlyCharges'].fillna(df['MonthlyCharges'].mean())
df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].mean())
df['tenure'] = df['tenure'].fillna(round(df['tenure'].median()))

print("Cleaned DataFrame shape:", df.shape)

print("\nMissing value counts:")
print(df.isnull().sum())