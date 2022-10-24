import numpy as np
import pandas as pd
import seaborn as sns
import json
import time
import csv
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.model_selection import GridSearchCV

df_data = pd.read_csv('./logisticRegressionTest/db/data.csv')

print('Data Shape :', df_data.shape)

rssi = (df_data[['Rssi']].values)*(-1)
riding = df_data['Riding'].values

X_train, X_test, Y_train, Y_test = train_test_split(rssi, riding, test_size = 0.3, random_state = 101)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = LogisticRegression(solver = 'liblinear', max_iter = 3000, C = 0.1)
model.fit(X_train, Y_train)

print('Train data score :', model.score(X_train, Y_train))
print('Test data score :', model.score(X_test, Y_test))


def modelScore():
    
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

def identifyWorker():

    workData = dict()
    workDataKeys = ["riding", "notRiding"]
    workDataValues = [[]]*2
    ridingData = []
    notRidingData = []
    
    loadFilePath = './logisticRegressionTest/db/db.json'

    with open(loadFilePath, 'r') as f:
        jsonData = json.load(f)

    print('db.json open ')

    vehicleWorkLength = len(jsonData["vehicle"]["work"])
    vehicleWorkDataRssi = []
    ridingPredict = []
    newRssi = ()

    for i in range(0, vehicleWorkLength):
        vehicleWorkData = jsonData["vehicle"]["work"][i]
        vehicleWorkDataRssi.append([vehicleWorkData['rssi']])

        newRssi = np.array([vehicleWorkDataRssi[i]])
        newRssi = scaler.transform(newRssi)

        ridingPredict.append(model.predict(newRssi))

        if(ridingPredict[i] == 1):
            ridingData.append(jsonData["vehicle"]["work"][i])
            workDataValues[0] = ridingData
        elif(ridingPredict[i] == 0):
            notRidingData.append(jsonData["vehicle"]["work"][i])
            workDataValues[1] = notRidingData

        workData = dict(zip(workDataKeys, workDataValues))

    writeFilePath = './logisticRegressionTest/db/work.json'

    with open(writeFilePath, 'w', encoding = 'utf-8') as makeFile:
        json.dump(workData, makeFile, ensure_ascii = False, indent = '\t')

    print('work.json write')
    print('time sleep 10sec')
    time.sleep(10)

def dataGet():

    print('Wait for 50sec to start dataGet')
    time.sleep(50) 

    vehicleWorkData = []
    vehicleWorkRssi = []
    rssi = []

    readDataPath = './logisticRegressionTest/db/db.json'
    dataFilePath = './logisticRegressionTest/db/data.csv'

    with open(readDataPath, 'r', encoding = 'utf-8') as readData:
        scanData = json.load(readData)
    
    vehicleWorkLength = len(scanData["vehicle"]["work"])

    for i in range(vehicleWorkLength):
        vehicleWorkData = scanData["vehicle"]["work"][i]
        vehicleWorkRssi.append(vehicleWorkData['rssi'])
        rssi.append(-(vehicleWorkRssi[i]))

        file = open(dataFilePath,'a', newline='')
        writeFile = csv.writer(file)
        writeFile.writerow([rssi[i], 1])
        file.close()
    
    print('close dataGet')

def gridSearch():

    model = LogisticRegression() 

    params = {
                'penalty' : ['l1', 'l2'],
                'C': [0.01, 0.05, 0.1, 0.5, 1, 5, 10],
                'solver':['saga', 'liblinear'],
                'max_iter':[100, 500, 1000, 5000, 10000]
            }

    grid_search = GridSearchCV(model, param_grid = params, cv=5)

    grid_search.fit(X_train, Y_train)
    print(grid_search.best_params_)

def main():
    modelScore()
    gridSearch()
    while(1):
        identifyWorker()
        dataGet()

if __name__ == "__main__":
	main()