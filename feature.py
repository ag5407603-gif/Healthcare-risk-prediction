import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv("healthcare_disease_prediction_dataset.csv")
print(df.head())
print(df.shape)
print(df.columns)
print(df.info())
print(df.describe())

print(df.isnull().sum())
df.columns = df.columns.str.strip()
 
df['Gender'] = df['Gender'].map({'Male':1,'Female':0})

cols_yes_no = ['Smoking','Alcohol Consumption','Exercise','Family History']
for col in cols_yes_no:
    df[col] = df[col].map({'Yes':1, 'No':0}) 

mapping = {'Low':0, 'Normal':1, 'High':2}
df['Blood Pressure'] = df['Blood Pressure'].map(mapping)
df['Cholesterol'] = df['Cholesterol'].map(mapping)
df['Glucose'] = df['Glucose'].map(mapping)


plt.figure(figsize=(12,8))
sns.heatmap(df.corr(), annot=False)
plt.title("Correlation Hearmap")
plt.show()

target = 'Diabetes'

from sklearn.model_selection import train_test_split
x = df.drop(target, axis=1)
y = df[target]
x_train , x_test , y_train , y_test = train_test_split(x, y, test_size=0.2, random_state=42)

from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(x_train, y_train)

from sklearn.metrics import accuracy_score, confusion_matrix
y_pred = model.predict(x_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test , y_pred))

sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d')
plt.title("confusion_matrix")
plt.show()

importance = pd.Series(model.feature_importances_, index=x.columns)
importance.sort_values().plot(kind='barh')
plt.title("Feature Importance")
plt.show()

input_data = [[45,1,2,2,1,1,0,1,28.5,1,0,0,0,0,0,0,0,0,0]]
prediction = model.predict(input_data)
if prediction[0] == 1:
    print("High Risk of Diabetes")

else:
    print("Low Risk")
