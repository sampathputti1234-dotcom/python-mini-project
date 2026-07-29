import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression


df = pd.read_csv('C:\\Users\\Putti Sampath\\Sampath\\AI and ML\\ml\\Python Script mini project 2\\churnguard_data.csv')

df = df.drop(columns=['customerID'])
df = df.drop_duplicates()

df['gender'] = df['gender'].str.strip()
df['PaymentMethod'] = df['PaymentMethod'].str.strip()

df['Churn'] = df['Churn'].str.strip().str.title()
df['PhoneService'] = df['PhoneService'].str.strip().str.title()
df['PaperlessBilling'] = df['PaperlessBilling'].str.strip().str.title()

contract_map = {
    'month to month': 'Month-to-month',
    'month-to-month': 'Month-to-month',
    'monthly': 'Month-to-month',
    '1 year': 'One year',
    'one year': 'One year',
    '2 year': 'Two year',
    'two year': 'Two year'
}
df['Contract'] = df['Contract'].str.lower().str.strip().map(contract_map).fillna(df['Contract'])

internet_map = {
    'dsl': 'DSL',
    'fibre optic': 'Fiber optic',
    'fiberoptic': 'Fiber optic',
    'fiber optic': 'Fiber optic',
    'none': 'No',
    'no': 'No'
}
df['InternetService'] = df['InternetService'].str.lower().str.strip().map(internet_map).fillna(df['InternetService'])

df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df = df[df['tenure'] > 0]
df = df[(df['MonthlyCharges'] >= 10) & (df['MonthlyCharges'] <= 200)]

df['MonthlyCharges'] = df['MonthlyCharges'].fillna(df['MonthlyCharges'].mean())
df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].mean())
df['tenure'] = df['tenure'].fillna(round(df['tenure'].median()))


df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})


contract_numeric_map = {
    'Month-to-month': 0,
    'One year': 1,
    'Two year': 2
}
df['Contract'] = df['Contract'].map(contract_numeric_map)


features = ['tenure', 'MonthlyCharges', 'TotalCharges', 'SeniorCitizen', 'Contract']
X = df[features]
y = df['Churn']


model = LogisticRegression(max_iter=1000)
model.fit(X, y)

tenure = int(input("Enter tenure (months): "))
monthly_charges = float(input("Enter Monthly Charges: "))
total_charges = float(input("Enter Total Charges: "))
senior_citizen = int(input("Senior Citizen? (1 = Yes, 0 = No): "))
contract = int(input("Contract type (0 = Month-to-month, 1 = One year, 2 = Two year): "))


user_data = pd.DataFrame([[tenure, monthly_charges, total_charges, senior_citizen, contract]], columns=features)


prediction = model.predict(user_data)[0]


if prediction == 1:
    print("Prediction: This customer is likely to CHURN.")
else:
    print("Prediction: This customer is likely to STAY.")