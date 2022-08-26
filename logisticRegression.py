import numpy as np
import pandas as pd
import json
import time

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def logisticRegression():
    df_data = pd.read_csv('data.csv')

    rssi = df_data[['Rssi']].values
    riding = df_data['Riding'].values

    train_features, test_features, train_rssi, test_rssi = train_test_split(rssi, riding)

    scaler = StandardScaler()
    train_features = scaler.fit_transform(train_features)
    test_features = scaler.transform(test_features)

    model = LogisticRegression()
    model.fit(train_features, train_rssi)

    print('Train score : ', model.score(train_features, train_rssi))
    print('Test score : ', model.score(test_features, test_rssi))

    return scaler, model

def identifyWorker(scaler, model):

    workData = dict()
    workDataKeys = ["riding", "notRiding"]
    workDataValues = [[]]*2
    ridingData = []
    notRidingData = []
    
    loadFilePath = './db/db.json'

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

    writeFilePath = './db/work.json'

    with open(writeFilePath, 'w', encoding = 'utf-8') as makeFile:
        json.dump(workData, makeFile, ensure_ascii = False, indent = '\t')

    print('work.json write')
    print('time sleep 10sec')
    time.sleep(10)

def main():
    scaler, model = logisticRegression()
    while(1):
        identifyWorker(scaler, model)

if __name__ == "__main__":
	main()