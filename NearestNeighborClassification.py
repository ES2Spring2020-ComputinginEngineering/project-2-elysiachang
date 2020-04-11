# Please put your code for Step 2 and Step 3 in this file.

# iMPORT STATEMENTS
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as ss

# FUNCTIONS
def openckdfile():
    glucose, hemoglobin, classification = np.loadtxt('ckd.csv', delimiter=',', skiprows=1, unpack=True)
    return glucose, hemoglobin, classification

def createTestCase():
    newglucose = np.random.random(1)
    newhemoglobin = np.random.random(1)
    return newglucose, newhemoglobin

def calculateDistanceArray(newglucose, newhemoglobin, glucose_scaled, hemoglobin_scaled):
    distancearray = []
    for i in range(159):
        distancearray = np.sqrt((newglucose-glucose_scaled)**2+(newhemoglobin-hemoglobin_scaled)**2)
    return distancearray

def nearestNeighborClassifier(newglucose, newhemoglobin, glucose_scaled, hemoglobin_scaled, classification_scaled):
    min_index = np.argmin(distanceArray)
    nearest_class = classification_scaled[min_index]
    return nearest_class

def kNearestNeighborClassifier(k, newglucose, newhemoglobin, glucose_scaled, hemoglobin_scaled, classification_scaled):
    sorted_indices = np.argsort(distanceArray)
    k_indices = sorted_indices[:k]
    k_classification = classification[k_indices]
    return k_classification

# MAIN SCRIPT
glucose, hemoglobin, classification = openckdfile()

# NORMALIZE DATA
glucose_scaled = (glucose-70)/(490-70)
hemoglobin_scaled = (hemoglobin-3.1)/(17.8-3.1)
classification_scaled = classification

# GRAPH DATA
plt.figure()
plt.plot(hemoglobin_scaled[classification_scaled==1],glucose_scaled[classification_scaled==1], "k.", label = "not CKD")
plt.plot(hemoglobin_scaled[classification_scaled==0],glucose_scaled[classification_scaled==0], "r.", label = "CKD")
plt.xlabel("Hemoglobin")
plt.ylabel("Glucose")
plt.title("Normalized Original Data")
plt.legend()
plt.show()

# CREATE TEST CASE
testCase = createTestCase()

# CALCULATE DISTANCE ARRAY
distanceArray = calculateDistanceArray(testCase[0], testCase[1], glucose_scaled, hemoglobin_scaled)

# NEAREST NEIGHBOR CLASSIFIER
neighbor = nearestNeighborClassifier(testCase[0], testCase[1], glucose_scaled, hemoglobin_scaled, classification_scaled)

# GRAPH TEST CASE - NEAREST NEIGHBOR
plt.figure()
plt.plot(hemoglobin_scaled[classification_scaled==1],glucose_scaled[classification_scaled==1], "k.", label = "not CKD")
plt.plot(hemoglobin_scaled[classification_scaled==0],glucose_scaled[classification_scaled==0], "r.", label = "CKD")
if neighbor==1:
    plt.plot(testCase[1], testCase[0], "k.", label = "not CKD", markersize = 15)
else:
    plt.plot(testCase[1], testCase[0], "r.", label = "CKD", markersize = 15)
plt.xlabel("Hemoglobin")
plt.ylabel("Glucose")
plt.title("Nearest Neighbor Graph")
plt.legend()
plt.show()

# K NEAREST
k_neighbor = kNearestNeighborClassifier(7, testCase[0], testCase[1], glucose_scaled, hemoglobin_scaled, classification_scaled)

# K MODE
k_mode = ss.mode(k_neighbor)

# GRAPH TEST CASE - K NEAREST NEIGHBOR
plt.figure()
plt.plot(hemoglobin_scaled[classification_scaled==1],glucose_scaled[classification_scaled==1], "k.", label = "not CKD")
plt.plot(hemoglobin_scaled[classification_scaled==0],glucose_scaled[classification_scaled==0], "r.", label = "CKD")
if k_mode[0]==1:
    plt.plot(testCase[1], testCase[0], "k.", label = "not CKD", markersize = 15)
else:
    plt.plot(testCase[1], testCase[0], "r.", label = "CKD", markersize = 15)
plt.xlabel("Hemoglobin")
plt.ylabel("Glucose")
plt.title("K-Nearest Neighbor Graph")
plt.legend()
plt.show()