import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import sklearn.tree as tree


file_path = 'creditcard.csv'
data = pd.read_csv(file_path)

data.head()

print(data.type.value_counts())

type=data["type"].value_counts()
transactions=type.index
quantity=type.values

import plotly.express as px
figure=px.pie(data,
             values=quantity,
             names=transactions,
             hole=0.5,
             title="Distribution of Transaction Type")
figure.show()

newbalanceOrig=list(data["newbalanceOrig"])
errors=["C865699625"]
for i in range(len(newbalanceOrig)):
    if newbalanceOrig[i]==errors[0]:
        newbalanceOrig[i]=0
data["newbalanceOrig"]=newbalanceOrig

newbalanceDest=list(data["newbalanceDest"])
error=["M1425940901"]
for i in range(len(newbalanceDest)):
    if newbalanceDest[i]==error[0]:
        newbalanceDest[i]=0
data["newbalanceDest"]=newbalanceDest

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

data_clean = data.drop(columns=['nameDest', 'Unnamed: 9', 'Unnamed: 10'])

numeric_cols = data_clean.select_dtypes(include=np.number).columns
data_clean[numeric_cols] = data_clean[numeric_cols].fillna(data_clean[numeric_cols].mean())

label_encoder = LabelEncoder()
data_clean['type'] = label_encoder.fit_transform(data_clean['type'])

X = data_clean.drop(columns=['isFraud'])
y = data_clean['isFraud']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

y_predl = model.predict_proba(X_test)

accuracy = accuracy_score(y_test, y_pred)
classification_report_output = classification_report(y_test, y_pred)

accuracy, classification_report_output
