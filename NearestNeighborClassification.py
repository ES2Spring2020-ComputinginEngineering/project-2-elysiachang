#Please put your code for Step 2 and Step 3 in this file.


import numpy as np
import matplotlib.pyplot as plt
import random


# FUNCTIONS
def openckdfile():
    glucose, hemoglobin, classification = np.loadtxt('ckd.csv', delimiter=',', skiprows=1, unpack=True)
    return glucose, hemoglobin, classification

def createTestCase():
    newhemoglobin = np.random.random(1)
    newglucose = np.random.random(1)
    return newhemoglobin, newglucose

def calculateDistanceArray(newglucose, newhemoglobin, glucose_scaled, hemoglobin_scaled):
    distancearray = []
    for i in range(159):
        distancearray = np.sqrt((newhemoglobin-hemoglobin_scaled)**2+(newglucose-glucose_scaled)**2)
    return distancearray

def nearestNeighborClassifier(newglucose, newhemoglobin, glucose_scaled, hemoglobin_scaled, classification):
    min_index_hemoglobin = np.argmin(distanceArray[0])
    nearest_class_hemoglobin = classification[min_index_hemoglobin]
    min_index_glucose = np.argmin(distanceArray[1])
    nearest_class_glucose = classification[min_index_glucose]
    return nearest_class_hemoglobin, nearest_class_glucose

def kNearestNeighborClassifier(k, newglucose, newhemoglobin, glucose_scaled, hemoglobin_scaled, classification):
    sorted_indices_hemoglobin = np.argsort(distanceArray[0])
    k_indices_hemoglobin = sorted_indices_hemoglobin[:k]
    k_classification_hemoglobin = classification[k_indices_hemoglobin]
    sorted_indices_glucose = np.argsort(distanceArray[1])
    k_indices_glucose = sorted_indices_glucose[:k]
    k_classification_glucose = classification[k_indices_glucose]
    return k_classification_hemoglobin, k_classification_glucose

# MAIN SCRIPT
glucose, hemoglobin, classification = openckdfile()

# NORMALIZE DATA
hemoglobin_scaled = (hemoglobin-3.1)/(17.8-3.1)
glucose_scaled = (glucose-70)/(490-70)

# GRAPH DATA
plt.figure()
plt.plot(hemoglobin_scaled[classification==1],glucose_scaled[classification==1], "k.", label = "Class 1")
plt.plot(hemoglobin_scaled[classification==0],glucose_scaled[classification==0], "r.", label = "Class 0")
plt.xlabel("Hemoglobin")
plt.ylabel("Glucose")
plt.legend()
plt.show()

testCase = createTestCase()

distanceArray = calculateDistanceArray(testCase, testCase, glucose_scaled, hemoglobin_scaled)

neighbor = nearestNeighborClassifier(testCase, testCase, glucose_scaled, hemoglobin_scaled, classification)

k_neighbor = kNearestNeighborClassifier(5, testCase, testCase, glucose_scaled, hemoglobin_scaled, classification)

plt.figure()
plt.plot(hemoglobin_scaled[classification==1],glucose_scaled[classification==1], "k.", label = "not CKD")
plt.plot(hemoglobin_scaled[classification==0],glucose_scaled[classification==0], "r.", label = "CKD")
plt.plot(testCase[classification==neighbor],testCase[classification==neighbor], "g.", label = "test case", markersize = 15)
plt.xlabel("Hemoglobin")
plt.ylabel("Glucose")
plt.legend()
plt.show()