import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve, auc, roc_auc_score

data = pd.read_csv("data_1018.csv")

print('Data Shape :', data.shape)

rssi = (data[['Rssi']].values) * (-1)
riding = data['Riding'].values

X_train, X_test, Y_train, Y_test = train_test_split(rssi, riding, test_size = 0.3, random_state = 101)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = LogisticRegression(solver = 'liblinear', penalty = 'l1', max_iter = 100, C = 0.01)
model.fit(X_train, Y_train)

newRssi1 = np.array([68.1339]) #기존 리빙랩 데이터 임계값
newRssi2 = np.array([63.69]) #한국타이어 리빙랩 데이터 임계값
newRssi3 = np.array([89])

newRssis = np.array([newRssi1, newRssi2, newRssi3])
newRssis = scaler.transform(newRssis)

print(model.predict(newRssis))
print(model.predict_proba(newRssis))

Y_pred = model.predict(X_test)

print('confusion matrix = \n', confusion_matrix(y_true = Y_test, y_pred = Y_pred))
print('accuracy = ', accuracy_score(y_true = Y_test, y_pred = Y_pred))
print('precision = ', precision_score(y_true = Y_test, y_pred = Y_pred))
print('recall = ', recall_score(y_true = Y_test, y_pred = Y_pred))
print('f1 score = ', f1_score(y_true = Y_test, y_pred = Y_pred))

Y_score = model.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_true = Y_test, y_score = Y_score)

plt.plot(fpr, tpr, label = 'roc curve (area = %0.3f)' % auc(fpr, tpr))
plt.plot([0, 1], [0, 1], linestyle='--', label = 'random')
plt.plot([0, 0, 1], [0, 1, 1], linestyle='--', label = 'ideal')
plt.legend()
plt.xlabel('false positive rate')
plt.ylabel('true positive rate')
plt.show()

print('auc = ', roc_auc_score(y_true = Y_test, y_score = Y_score))

from sklearn.model_selection import GridSearchCV

model = LogisticRegression() 

params = {
            'penalty' : ['l2'],
            'C': [0.01, 0.05, 0.1, 0.5, 1, 5, 10],
            'solver':['sag', 'lbfgs', 'newton-cg','saga', 'liblinear'],
            'max_iter':[100, 500, 1000, 5000, 10000]
         }

grid_search = GridSearchCV(model, param_grid=params, cv=5)

grid_search.fit(X_train, Y_train)
grid_search.best_params_

params = {
            'penalty' : ['l1'],
            'C': [0.01, 0.05, 0.1, 0.5, 1, 5, 10],
            'solver':['saga', 'liblinear'],
            'max_iter':[100, 500, 1000, 5000, 10000]
         }

grid_search = GridSearchCV(model, param_grid=params, cv=5)

grid_search.fit(X_train, Y_train)
grid_search.best_params_