import numpy as np
import pandas as pd
import seaborn as sns
import json
import time
import csv


from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def logisticRegression():
    df_data = pd.read_csv('./logisticRegressionTest/db/data.csv')

    rssi = (df_data[['Rssi']].values)*(-1)
    riding = df_data['Riding'].values

    train_features, test_features, train_rssi, test_rssi = train_test_split(rssi, riding, test_size = 0.30, random_state = 101)

    scaler = StandardScaler()
    train_features = scaler.fit_transform(train_features)
    test_features = scaler.transform(test_features)

    model = LogisticRegression(solver = 'liblinear', max_iter = 3000)
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

def main():
    scaler, model = logisticRegression()
    while(1):
        identifyWorker(scaler, model)
        dataGet()

if __name__ == "__main__":
	main()