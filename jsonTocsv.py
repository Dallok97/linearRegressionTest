# json 형태로 수집한 데이터를 csv 로 변환

import numpy as np
import pandas as pd
import json
import csv

def dataGet():

    vehicleWorkData = []
    vehicleWorkRssi = []
    vehicleAreaData = []
    vehicleAreaRssi = []
    workRssi = []
    areaRssi = []

    readDataPath = './db/data_1018.json'
    dataFilePath = './db/data_1018.csv'

    with open(readDataPath, 'r', encoding = 'utf-8') as readData:
        scanData = json.load(readData)

    vehicleWorkLength = len(scanData["riding"])
    vehicleAreaLength = len(scanData["notRiding"])

    for i in range(vehicleWorkLength):
        vehicleWorkData = scanData["riding"][i]
        workRssi.append(vehicleWorkData["rssi"])

    for i in range(vehicleWorkLength):
        file = open(dataFilePath,'a', newline='')
        writeFile = csv.writer(file)
        writeFile.writerow([-(workRssi[i]), 1])
        file.close()

    for j in range(vehicleAreaLength):
        vehicleAreaData = scanData["notRiding"][j]
        areaRssi.append(vehicleAreaData["rssi"])

    for j in range(vehicleAreaLength):
        file = open(dataFilePath,'a', newline='')
        writeFile = csv.writer(file)
        writeFile.writerow([-(areaRssi[j]), 0])
        file.close()

def main():
    dataGet()

if __name__ == "__main__":
	main()
        