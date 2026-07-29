import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


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

categorical_cols = ['gender', 'PhoneService', 'InternetService', 'Contract', 'PaperlessBilling', 'PaymentMethod']
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)


X = df.drop(columns=['Churn'])
y = df['Churn']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)


acc = accuracy_score(y_test, y_pred)
print(f"Accuracy Score: {acc}")


report = classification_report(y_test, y_pred, target_names=['Stay', 'Churn'])
print("\nClassification Report:")
print(report)
