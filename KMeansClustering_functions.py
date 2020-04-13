# ELYSIA CHANG
# ES2 PROJECT 2
# K-MEANS CLUSTERING FUNCTIONS

# ******************************

# Please place your FUNCTION code for step 4 here.

# IMPORT STATEMENTS
import numpy as np
import matplotlib.pyplot as plt
import random

# FUNCTIONS
def openckdfile():
    glucose, hemoglobin, classification = np.loadtxt('ckd.csv', delimiter=',', skiprows=1, unpack=True)
    return glucose, hemoglobin, classification

def normalizeData(glucose, hemolobin, classification):
    glucose_list = []
    hemoglobin_list = []
    for line in glucose:
        new_glucose = (line-70)/(490-70)
        glucose_list.append(new_glucose)
    for line in hemoglobin:
        new_hemoglobin = (line- 3.1)/(17.8-3.1)
        hemoglobin_list.append(new_hemoglobin)
    glucose_scaled = np.array(glucose_list)
    hemoglobin_scaled = np.array(hemoglobin_list)
    classification = np.array(classification)
    return glucose_scaled, hemoglobin_scaled, classification

def generateCentroids(number):
    generated_centroids = []
    for i in range(number):
        g = random.uniform(0,1)
        h = random.uniform(0,1)
        generated_centroids.append([g,h])
    new_centroids = np.array(generated_centroids)
    return new_centroids

def calculateDistanceArray(new_centroids, glucose_scaled, hemoglobin_scaled):
    new_distance = []
    for i in range(len(new_centroids)):
        centroid_array = new_centroids[i]
        distance = np.sqrt(((centroid_array[0]-glucose_scaled)**2) +((centroid_array[1]-hemoglobin_scaled)**2))
        new_distance.append(distance)
    distance_array = np.array(new_distance)
    return distance_array

def assign(new_centroids, hemoglobin_scaled, glucose_scaled):
    K = new_centroids.shape[0]
    distance = np.zeros((K, len(hemoglobin_scaled)))
    for i in range(K):
        distance = calculateDistanceArray(new_centroids, glucose_scaled, hemoglobin_scaled)
    assignments = np.argmin(distance, axis=0)
    return assignments

def update(assignments, glucose_scaled, hemoglobin_scaled, new_centroids):
    K = new_centroids.shape[0]
    updated_centroids = np.zeros((K,2))
    new_centers = assignments
    for i in range (K):
        updated_centroids[i][1] = np.mean(glucose_scaled[new_centers==i])
        updated_centroids[i][0] = np.mean(hemoglobin_scaled[new_centers==i])
    return updated_centroids

def iterationData(assignments, updated_centroids):
    iteration = 0
    while iteration < 100:
        assignments = assign(updated_centroids, hemoglobin_scaled, glucose_scaled)
        updated_centroids = update(assignments, glucose_scaled, hemoglobin_scaled, updated_centroids)
        iteration += 1
    return assignments, updated_centroids

def graphingkMeans(glucose_scaled, hemoglobin_scaled, assignments, updated_centroids):
    plt.figure()
    for i in range(int(assignments.max()+1)):
        rcolor = np.random.rand(3)
        plt.plot(hemoglobin_scaled[assignments==i],glucose_scaled[assignments==i], ".", label = "Class " + str(i), color = rcolor)
        plt.plot(updated_centroids[i,0], updated_centroids[i,1], "D", label = "Centroid " + str(i), color = rcolor)
    plt.xlabel("Hemoglobin")
    plt.ylabel("Glucose")
    plt.title("K-Means Clustering Graph")
    plt.legend()
    plt.show()

def positivesNegatives(classification, assignments):
# True Positives Rate (Sensitivity) = what percentage of CKD patients were correctly labeled by K-Means
# False Positives Rate = what percentage of non-CKD were incorrectly labelled by K-Means as being in the CKD cluster
# True Negatives Rate (Specificity) = what percentage of non-CKD patients were correctly labelled by K-Means
# False Negatives Rate = what percentage of CKD were incorrectly labelled by K-Means as being in the non-CKD cluster
    truePositives = 0
    falsePositives = 0
    trueNegatives = 0
    falseNegatives = 0
    for i in range(len(classification)):
        if (assignments[i]==0 and classification[i]==0):
            truePositives += 1
        if (assignments[i]==0 and classification[i]==1):
            falsePositives += 1
        if (assignments[i]==0 and classification[i]==0):
            trueNegatives += 1
        if (assignments[i]==1 and classification[i]==0):
            falseNegatives += 1
    return truePositives, falsePositives, trueNegatives, falseNegatives

# MAIN SCRIPT
glucose, hemoglobin, classification = openckdfile()

glucose_scaled, hemoglobin_scaled, classification = normalizeData(glucose, hemoglobin, classification)

new_centroids = generateCentroids(2)

assignments = assign(new_centroids, hemoglobin_scaled, glucose_scaled)

updated_centroids = update(assignments, glucose_scaled, hemoglobin_scaled, new_centroids)

assignments, updated_centroids = iterationData(assignments, updated_centroids)

graphingkMeans(glucose_scaled, hemoglobin_scaled, assignments, updated_centroids)

plusminus = positivesNegatives(classification, assignments)

truePositives = plusminus[0]

falsePositives = plusminus[1]

trueNegatives = plusminus[2]

falseNegatives = plusminus[3]

sensitivity = (truePositives/(truePositives+falsePositives))*100

falsePositivesPercentage = (falsePositives/(falsePositives+trueNegatives))*100

specificity = (trueNegatives/(trueNegatives+falsePositives))*100

falseNegativesPercentage = (falseNegatives/(falseNegatives+truePositives))*100

print("The true positives rate (sensitivity) is", sensitivity, "%")

print("The false positives rate is", falsePositivesPercentage, "%")

print("The true negatives rate (specificity) is", specificity, "%")

print("The false negatives rate is", falseNegativesPercentage, "%")