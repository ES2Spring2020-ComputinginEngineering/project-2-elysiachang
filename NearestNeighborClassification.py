# Please put your code for Step 2 and Step 3 in this file.

# iMPORT STATEMENTS
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as ss

# FUNCTIONS
def openckdfile():
# saving & loading text files into the program
    glucose, hemoglobin, classification = np.loadtxt('ckd.csv', delimiter=',', skiprows=1, unpack=True)
    return glucose, hemoglobin, classification

def createTestCase():
# a random test case that falls within is the minimum and maximum values of the training hemoglobin and glucose data
# returns newglucose, newhemoglobin for the randomly generated test case
    newglucose = np.random.random(1)
    newhemoglobin = np.random.random(1)
    return newglucose, newhemoglobin

def calculateDistanceArray(newglucose, newhemoglobin, glucose_scaled, hemoglobin_scaled):
# returns the distance array (an array, the same length as glucose and hemoglobin) which contains the distance 
# calculated to the new point (newglucose, newhemoglobin) from each point in the existing dataset
    distancearray = []
    for i in range(159):
        distancearray = np.sqrt((newglucose-glucose_scaled)**2+(newhemoglobin-hemoglobin_scaled)**2)
    return distancearray

def nearestNeighborClassifier(newglucose, newhemoglobin, glucose_scaled, hemoglobin_scaled, classification_scaled):
# returns the classification for the point nearest_class either a 1 or 0 based on the nearest neighbor
# calls on the function calculateDistanceArray
    min_index = np.argmin(distanceArray)
    nearest_class = classification_scaled[min_index]
    return nearest_class

def kNearestNeighborClassifier(k, newglucose, newhemoglobin, glucose_scaled, hemoglobin_scaled, classification_scaled):
# returns the classification for the point newglucose, newhemoglobin either a 1 or 0 based on the k (odd int) nearest neighbors
# the classification held by the majority of the k nearest points to the new point will be the value assigned to the new point
# K should always be an odd number to avoid ties
    sorted_indices = np.argsort(distanceArray)
    k_indices = sorted_indices[:k]
    k_classification = classification[k_indices]
    return k_classification

# MAIN SCRIPT
glucose, hemoglobin, classification = openckdfile()

# NORMALIZE DATA
# used to normalize three NumPy arrays of equal length and return the three normalized arrays
glucose_scaled = (glucose-70)/(490-70)
hemoglobin_scaled = (hemoglobin-3.1)/(17.8-3.1)
classification_scaled = classification

# GRAPH DATA
# a scatter plot of the glucose (Y) and hemoglobin (X) with the points graphed colored based on the classification
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