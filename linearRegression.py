import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

df_data = pd.read_csv('data.csv')

rssi = df_data[['Rssi']]
riding = df_data['Riding']

train_features, test_features, train_rssi, test_rssi = train_test_split(rssi, riding)

scaler = StandardScaler()
train_features = scaler.fit_transform(train_features)
test_features = scaler.transform(test_features)

model = LogisticRegression()
model.fit(train_features, train_rssi)

print('Train score : ', model.score(train_features, train_rssi))

print('Test score : ', model.score(test_features, test_rssi))

newRssi1 = np.array([83])
newRssi2 = np.array([78])
newRssi3 = np.array([89])

newRssis = np.array([newRssi1, newRssi2, newRssi3])

newRssis = scaler.transform(newRssis)

print(model.predict(newRssis))

print(model.predict_proba(newRssis))

x = df_data['Rssi']
y = df_data['Riding']

sns.regplot(x = x, y = y, data = df_data, logistic = True, ci = None, scatter_kws = {'color': 'black'}, line_kws = {'color': 'red'})

