#Please place your FUNCTION code for step 4 here

#Import Statements

import numpy as np

import matplotlib.pyplot as plt

import random

 

#Functions

def openckdfile():

    glucose, hemoglobin, classification = np.loadtxt('ckd.csv', delimiter=',', skiprows=1, unpack=True)

    return glucose, hemoglobin, classification

 

def normalize(glucose, hemolobin, classification):

    g = []

    h = []

    for l in glucose:

        newg = (l-70)/(490-70)

        g.append(newg)

    for l in hemoglobin:

        newh = (l- 3.1)/(17.8-3.1)

        h.append(newh)

    glucose_scaled = np.array(g)

    hemoglobin_scaled = np.array(h)

    classification = np.array(classification)

    return glucose_scaled, hemoglobin_scaled, classification

 

def select(K):

    return np.random.random((K,2))

 

def generate_centroids(k):

    centroids_generated = []

    for i in range (k):

        g = random.uniform (0,1)

        h = random.uniform(0,1)

        centroids_generated.append([g,h])

    newcentroids = np.array(centroids_generated)

    return newcentroids

 

def calculateDistanceArray(newcentroids, glucose_scaled, hemoglobin_scaled):

    newdistance = []

    for i in range(len(newcentroids)):

        centroid_array = newcentroids[i]

        distance = np.sqrt(((centroid_array[0]-glucose_scaled)**2) +((centroid_array[1]-hemoglobin_scaled)**2))

        newdistance.append(distance)

    distance_array = np.array(newdistance)

    return distance_array

 

def assign(newcentroids, hemoglobin_scaled, glucose_scaled):

    K = newcentroids.shape[0]

    distance = np.zeros((K, len(hemoglobin_scaled)))

    for i in range(K):

        distance = calculateDistanceArray (newcentroids, glucose_scaled, hemoglobin_scaled)

    final_assignments = np.argmin(distance, axis = 0)

    return final_assignments

 

def update(final_assignments, glucose_scaled, hemoglobin_scaled, newcentroids):

    K = newcentroids.shape[0]

    updated_centroids = np.zeros((K,2))

#    assignK = final_assignments.sort()
    
    assignK = final_assignments

    for i in range (K):
        
        updated_centroids[i][1] = np.mean(glucose_scaled[assignK==i])
        
        updated_centroids[i][0] = np.mean(hemoglobin_scaled[assignK==i])
    
    print(updated_centroids)

    return updated_centroids



def interation_data(final_assignments, updated_centroids):

    interation = 0

    while interation < 100:

        final_assignments = assign(updated_centroids, hemoglobin_scaled, glucose_scaled)

        updated_centroids = update(final_assignments, glucose_scaled, hemoglobin_scaled, updated_centroids)

        interation+=1

    return final_assignments, updated_centroids

 

def graphingkMeans(glucose_scaled, hemoglobin_scaled, final_assignments,updated_centroids):

    plt.figure()

    for i in range(int(final_assignments.max()+1)):

        rcolor = np.random.rand(3)

        plt.plot(hemoglobin_scaled[final_assignments==i],glucose_scaled[final_assignments==i], ".", label = "Class " + str(i), color = rcolor)

        plt.plot(updated_centroids[i,0], updated_centroids[i,1], "D", label = "Centroid " + str(i), color = rcolor)

    plt.xlabel("Hemoglobin")

    plt.ylabel("Glucose")

    plt.title("K Mean ")

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

#Main Script

glucose, hemoglobin, classification = openckdfile()

glucose_scaled, hemoglobin_scaled, classification = normalize(glucose, hemoglobin, classification)

#newcentroids = select(2)

newcentroids = generate_centroids(2)

print(newcentroids)

final_assignments = assign(newcentroids, hemoglobin_scaled, glucose_scaled)

print(final_assignments)

updated_centroids = update(final_assignments, glucose_scaled, hemoglobin_scaled, newcentroids)

print(updated_centroids)

final_assignments, updated_centroids = interation_data(final_assignments, updated_centroids)

graphingkMeans(glucose_scaled, hemoglobin_scaled, final_assignments, updated_centroids)

plusminus = positivesNegatives(classification, final_assignments)


## IMPORT STATEMENTS
#import numpy as np
#import matplotlib.pyplot as plt
#import random
#
## CUSTOM FUNCTIONS
#def openckdfile():
#    glucose, hemoglobin, classification = np.loadtxt('ckd.csv', delimiter=',', skiprows=1, unpack=True)
#    return glucose, hemoglobin, classification
#
#
## JENN'S STARTER CODE
#def select(K):
#    return np.random.random((K, 2))
#
#def assign(centroids, hemoglobin_scaled, glucose_scaled):
#    K = centroids.shape[0]
#    distances = np.zeros((K, len(hemoglobin_scaled)))
#    for i in range(K):
#        g = centroids[i,1]
#        h = centroids[i,0]
#        distances[i] = np.sqrt((hemoglobin_scaled-h)**2+(glucose_scaled-g)**2)
#        np.append(new_point, distances)
#    assignments = np.argmin(distances, axis = 0) 
#    print(assignments)
#    return assignments
#
## MAIN SCRIPT
#glucose, hemoglobin, classification = openckdfile()
#
## NORMALIZE DATA
## used to normalize three NumPy arrays of equal length and return the three normalized arrays
#glucose_scaled = (glucose-70)/(490-70)
#hemoglobin_scaled = (hemoglobin-3.1)/(17.8-3.1)
#classification_scaled = classification
#
#plt.figure()
#plt.plot(hemoglobin_scaled[classification_scaled==1],glucose_scaled[classification_scaled==1], "k.", label = "not CKD")
#plt.plot(hemoglobin_scaled[classification_scaled==0],glucose_scaled[classification_scaled==0], "r.", label = "CKD")
#plt.xlabel("Hemoglobin")
#plt.ylabel("Glucose")
#plt.legend()
#plt.show()
#
#centroids = select(10)
#
#new_point = random.uniform(0.0,1.0)
#
#assignments = assign(centroids, hemoglobin_scaled, glucose_scaled)

## IMPORT STATEMENTS
#import numpy as np
#import matplotlib.pyplot as plt
#import random
#
## FUNCTIONS
#def openckdfile():
#    glucose, hemoglobin, classification = np.loadtxt('ckd.csv', delimiter=',', skiprows=1, unpack=True)
#    return glucose, hemoglobin, classification
#
#def createNewPoint():
## a random test case that falls within is the minimum and maximum values of the training hemoglobin and glucose data
## returns newglucose, newhemoglobin for the randomly generated test case
#    newglucose = np.random.random(1)
#    newhemoglobin = np.random.random(1)
#    return newglucose, newhemoglobin
#
## JENN'S STARTER CODE
#def select(K):
## selects K random centroid points that fall within the range of the feature sets
#    return np.random.random((K, 2))
#
#def assign(centroids, hemoglobin, glucose):
## assigns to each data point a label based on the which centroid it is closest too --- using a method similar to the Nearest Neighbor Classification
#    K = centroids.shape[0]
#    distances = np.zeros((K, len(hemoglobin)))
#    for i in range(K):
#        g = centroids[i,1]
#        h = centroids[i,0]
#        distances[i] = np.sqrt((hemoglobin-h)**2+(glucose-g)**2)
#    assignments = np.argmin(distances, axis = 0)
#    return assignments
#
## MY CODE
#def update(assignments, hemoglobin, glucose, centroids):
## updates the location of each centroid by taking the means of all features of all observations (data points) currently assigned to that centroid
## those means are then used as the features for the updated centroid location
#    my_assignments = assignments.sort()
#    K = centroids.shape[0]
#    new_centroids = np.zeros((K, 2))
#    for i in range(K):
#        if my_assignments == i:
#            new_centroids[i,1] = np.mean(glucose[my_assignments==i])
#            new_centroids[i,0] = np.mean(hemoglobin[my_assignments==i])
#    return new_centroids
#
#def iterate(assignments, new_centroids):
## iterates by repeating the assign and update steps until an end condition is met
## the end condition in this case is: a maximum number of iterations have happened
#    i = 0
#    while i < 100: # ask an input for the number of times you want to iterate at the start of program
#        assignments = assign(new_centroids, hemoglobin, glucose)
#        centroids = update(assignments, hemoglobin, glucose, new_centroids)
#        i += 1
#    return assignments, centroids
#
#def graphingKMeans(glucose, hemoglobin, assignment, new_centroids):
## example code provided by Jenn on how to check if our function accomplishes the right things
#    plt.figure()
#    for i in range(assignment.max()+1):
#        rcolor = np.random.rand(3,)
#        plt.plot(hemoglobin[assignment==i],glucose[assignment==i], ".", label = "Class " + str(i), color = rcolor)
#        plt.plot(new_centroids[i, 0], new_centroids[i, 1], "D", label = "Centroid " + str(i), color = rcolor)
#    plt.xlabel("Hemoglobin")
#    plt.ylabel("Glucose")
#    plt.legend()
#    plt.show() # graph looks pretty random (step 1), it'll look better when you fix update (step 3)
#
#def positivesNegatives(hemoglobin, glucose, classification, assignments):
## True Positives Rate (Sensitivity) = what percentage of CKD patients were correctly labeled by K-Means
## False Positives Rate = what percentage of non-CKD were incorrectly labelled by K-Means as being in the CKD cluster
## True Negatives Rate (Specificity) = what percentage of non-CKD patients were correctly labelled by K-Means
## False Negatives Rate = what percentage of CKD were incorrectly labelled by K-Means as being in the non-CKD cluster
#    truePositives = 0
#    falsePositives = 0
#    trueNegatives = 0
#    falseNegatives = 0
#    for i in range(len(classification)):
#        if (assignments[i]==0 and classification[i]==0):
#            truePositives += 1
#        if (assignments[i]==0 and classification[i]==1):
#            falsePositives += 1
#        if (assignments[i]==0 and classification[i]==0):
#            trueNegatives += 1
#        if (assignments[i]==1 and classification[i]==0):
#            falseNegatives += 1
#    return truePositives, falsePositives, trueNegatives, falseNegatives
#
## MAIN SCRIPT - PUT ALL OF THIS INTO DRIVER
#glucose, hemoglobin, classification = openckdfile()
#
## NORMALIZE DATA
#glucose = (glucose-70)/(490-70)
#hemoglobin = (hemoglobin-3.1)/(17.8-3.1)
#
## CREATE NEW CENTROID
#newPoint = createNewPoint()
#
## PLOT GRAPH
#plt.figure()
#plt.plot(hemoglobin[classification==1],glucose[classification==1], "k.", label = "not CKD")
#plt.plot(hemoglobin[classification==0],glucose[classification==0], "r.", label = "CKD")
#plt.xlabel("Hemoglobin")
#plt.ylabel("Glucose")
#plt.legend()
#plt.show()
#
## VARIABLE ASSIGNMENT
#centroids = select(10)
#assignments = assign(centroids, hemoglobin, glucose)
#new_centroids = update(assignments, hemoglobin, glucose, centroids)
#iterate_assignments = iterate(assignments, new_centroids)[0]
#iterate_centroids = iterate(assignments, new_centroids)[1]
#
## FINAL GRAPH
#graphingKMeans(glucose, hemoglobin, assignments, new_centroids)
#
## POSITIVES AND NEGATIVES
#truePositives = ((positivesNegatives(hemoglobin, glucose, classification, assignments)[0])/158)*100
#falsePositives = ((positivesNegatives(hemoglobin, glucose, classification, assignments)[1])/158)*100
#trueNegatives = ((positivesNegatives(hemoglobin, glucose, classification, assignments)[2])/158)*100
#falseNegatives = ((positivesNegatives(hemoglobin, glucose, classification, assignments)[3])/158)*100

# ********** #

#DEAD CODE

#def update(K):
#    for i in centroids.keys():
#        centroids[i][0] = np.mean(hemoglobin[hemoglobin['closest'] == i]['x'])
#        centroids[i][1] = np.mean(glucose[glucose['closest'] == i]['y'])
#    return K

#def update(assignments, hemoglobin, glucose, centroids):
#    Y = []
#    for i in range(K):
#        Y[i+1]=np.array([]).reshape(2,0)
#    for j in range(len(hemoglobin)):
#        Y[assignments[j]]=np.c_[Y[assignments[j]],X[j]]
#    for a in range(K):
#        Y[a+1]=Y[a+1].T
#    for b in range(K):
#        centroids[:,b]=np.mean(Y[b+1],axis=0)
#    return Y

#def update(assignments, hemoglobin, glucose, centroids):
#    # calculate mean of individual clusters
#    # move the the centroid to calculated mean
#    old_centroids = np.zeros(centroids.shape)
#    new_centroids = deepcopy(centroids)
#    clusters = np.zeros(K)
#    distances = np.zeros((K, len(hemoglobin)))
#    error = np.linalg.norm(new_centroids - old_centroids)
#    while error!=0:
#        for i in range(K):
#            distances[:,i] = np.linalg.norm(distances - new_centroids[i], axis=1)
#        clusters = np.argmin(distances, axis = 1)
#        old_centroids = deepcopy(new_centroids)
#        for i in range(K):
#            new_centroids[i] = np.mean(distances[clusters == i], axis=0)
#        error = np.linalg.norm(new_centroids - old_centroids)
##    for i in range(K):
##        if sorted_indices==i:
##            new_centroids[i,1]=np.mean(glucose[sorted_indices==i])
##            new_centroids[i,0]=np.mean(hemoglobin[sorted_indices==i])
#    return new_centroids

#def update(assignments, hemoglobin, glucose, centroids):
#    # calculate mean of individual clusters
#    # move the the centroid to calculated mean
#    my_assignments = assignments.sort()
#    K = centroids.shape[0]
#    new_centroids = np.zeros((K, 2))
#    for i in range(K):
#        if my_assignments==i:
#            new_centroids[i,1]=np.mean(glucose[my_assignments==i])
#            new_centroids[i,0]=np.mean(hemoglobin[my_assignments==i])
#    return new_centroids

#def update(assignments, hemoglobin, glucose, centroids):
#    # calculate mean of individual clusters
#    # move the the centroid to calculated mean
#    my_assignments = assignments.sort()
#    K = centroids.shape[0]
#    new_centroids = np.zeros((K, 2))
#    for i in range(K):
#        if my_assignments==i:
#            new_centroids[i,1]=np.mean(glucose[my_assignments==i])
#            new_centroids[i,0]=np.mean(hemoglobin[my_assignments==i])
#        else:
#            new_centroids[i,1]=np.mean(new_centroids[my_assignments==i])
#            new_centroids[i,0]=np.mean(new_centroids[my_assignments==i])
#    return new_centroids

#def update(assignments, hemoglobin, glucose, centroids):
#    # make a new point
#    # take a point and assign to a new centroid it's closest to
#    # take the points of the same classifications
#    # find the mean x & y centroid point
#    K = centroids.shape[0]
#    new_centroids = np.zeros((K, 2))
#    for i in range(K):
#        if len(glucose[assignments==i])!=0:
#            new_centroids[i,1]=np.mean(glucose[assignments==i])
#            new_centroids[i,0]=np.mean(hemoglobin[assignments==i])
#        else:
#            new_centroids[i,0]==100
#            new_centroids[i,1]==100
#    return new_centroids

#def update(assignments, hemoglobin, glucose, centroids):
#    my_assignments = assignments.sort()
#    K = centroids.shape[0]
#    new_centroids = np.zeros((K, 2))
#    for i in range(K):
#        if my_assignments==i:
#            new_centroids[i,1]=np.mean(glucose[my_assignments==i])
#            new_centroids[i,0]=np.mean(hemoglobin[my_assignments==i])
#    return new_centroids

# MY CODE
#def update(assignments, hemoglobin, glucose, centroids):
#    centroids_old = np.zeros(centroids.shape)    
#    clusters = np.zeros(len(2))
#    error = dist(centroids, centroids_old, None)
#    while error != 0:
#        for i in range(len(2)):
            
#def update(assignments, hemoglobin, glucose, centroids):
#    # make a new point
#    # take a point and assign to a new centroid it's closest to
#    # take the points of the same classifications
#    # find the mean x & y centroid point
#    K = centroids.shape[0]
#    new_centroids = np.zeros((K, 2))
#    for i in range(K):
#        if len(glucose[assignments==i])!=0:
#            new_centroids[i,1]=np.mean(glucose[assignments==i])
#            new_centroids[i,0]=np.mean(hemoglobin[assignments==i])
#        else:
#            new_centroids[i,0]==100
#            new_centroids[i,1]==100
#    return new_centroids

#def update(assignments, hemoglobin, glucose, centroids):
#    K = centroids.shape[0]
#    new_centroids = np.zeros((K, 2))
#    for i in range(K):
#        if np.any(assignments==i):
#            new_centroids[i,1]=np.mean(glucose[assignments==i])
#            new_centroids[i,0]=np.mean(hemoglobin[assignments==i])
#    return new_centroids

#def update(assignments, hemoglobin, glucose, centroids):
#    K = centroids.shape[0]
#    Y = []
#    new_centroids = np.zeros((K, 2))
#    for i in range(K):
#        centroids[:,i]=np.mean(Y[i+1],axis=0)
#    return new_centroids
    
#def update(assignments, hemoglobin, glucose, centroids):
#    Y = []
#    for i in range(K):
#        Y[i+1]=np.array([]).reshape(2,0)
#    return 
#    
#def update(assignments, hemoglobin, glucose, centroids):
#    Y = []
#    for k in range(K):
#        centroids[:,k]=np.mean(Y[k+1],axis=0)
#    my_assignments = assignments.sort()
#    K = centroids.shape[0]
#    new_centroids = np.zeros((K, 2))
#    for i in range(my_assignments):
#        if my_assignments[0]==i:
#            new_centroids[i,1]=np.mean(glucose[my_assignments==i])
#            new_centroids[i,0]=np.mean(hemoglobin[my_assignments==i])
#    return new_centroids



## IMPORT STATEMENTS
#import math
#import random
#import numpy as np
#import matplotlib.pyplot as plt
#
## CUSTOM FUNCTIONS
#def openckdfile():
#    glucose, hemoglobin, classification = np.loadtxt('ckd.csv', delimiter=',', skiprows=1, unpack=True)
#    return glucose, hemoglobin, classification
#
#def normalizeData(glucose, hemoglobin, classification):
#    g_list = []
#    h_list = []
#    
#    for line in glucose:
#        g_scaled = (line-70)/(490-70)
#        g_list.append(g_scaled)
#    
#    for line in hemoglobin:
#        h_scaled = (line-3.1)/(17.8-3.1)
#        h_list.append(h_scaled)
#    
#    glucose_scaled = np.array(g_list)
#    hemoglobin_scaled = np.array(h_list)
#    classification = np.array(classification)
#    
#    return glucose_scaled, hemoglobin_scaled, classification
#
#def initialCentroids(k):
#    scaled_centroids = []
#    
#    for i in range(k):
#        
#        g = random.uniform(0,1)
#        h = random.uniform(0,1)
#        centroid_k = [g,h,i]
#        scaled_centroids.append(centroid_k)
#        
#    centroid_array = np.array(scaled_centroids)
#    
#    return centroid_array
#
#def calculateDistanceArray(centroid_array, glucose_value, hemoglobin_value):
#    distance = []
#    
#    for i in range(len(centroid_array)):
#        centroid = centroid_array[i]
#        
#        d = math.sqrt((centroid[0] - glucose_value)**2 + (centroid[1] - hemoglobin_value)**2)
#        distance.append(d)
#        
#    distance_array = np.array(distance)
#    
#    return distance_array
#
#def kMeansClustering(k, glucose_scaled, hemoglobin_scaled):
#    iteration = 0
#    centroid_array = initialCentroids(k)
#    
#    while iteration < 100:
#        assignments = []
#        
#        """ Assignment Step """
#        for i in range(len(glucose_scaled)):
#            glucose_value = glucose_scaled[i]
#            hemoglobin_value = hemoglobin_scaled[i]
#            distance_array = calculateDistanceArray(centroid_array, glucose_value, hemoglobin_value)
#            
#            min_index = np.argmin(distance_array)
#            nearest_centroid = centroid_array(min_index)[2]
#            
#            assignments.append([glucose_value, hemoglobin_value, nearest_centroid])
#        
#        assignment_array = np.array(assignments)
#        new_classes = assignment_array[:,2]
#        
#        """ Update Step """
#        for i in range(len(centroid_array)):
#            centroid_array[i][0] = np.mean(assignment_array[new_classes==i][:,0])
#            centroid_array[i][1] = np.mean(assignment_array[new_classes==i][:,1])
#            
#        iteration += 1
#    
#    print(centroid_array)
#    
#    return centroid_array, new_classes
#
#def unscaledCentroids(centroid_array):
#    c = []
#    for i in range(len(centroid_array)):
#        c1 = (centroid_array[i][0] * (490-70) + 70)
#        c2 = (centroid_array[i][1] * (17.8-3.1) + 3.1)
#        c3 = centroid_array[i][2]
#        
#        c.append([c1, c2, c3])
#        
#    unscaled_centroids = np.array(c)
#    
#    return unscaled_centroids
#
#def graphingKMeans(glucose, hemoglobin, new_classes, unscaled_centroids):
#    plt.figure()
#    for i in range(int(new_classes.max()+1)):
#        rcolor = np.random.rand(3,)
#        plt.plot(hemoglobin[new_classes==i],glucose[new_classes==i], ".", label = "Class " + str(i), color = rcolor)
#        plt.plot(unscaled_centroids[i, 1], unscaled_centroids[i, 0], "D", label = "Centroid " + str(i), color = rcolor)
#    plt.xlabel("Hemoglobin")
#    plt.ylabel("Glucose")
#    plt.legend()
#    plt.show()
#    
## MAIN SCRIPT
#k = 2
#glucose, hemoglobin, classification = openckdfile()
#glucose_scaled, hemoglobin_scaled, classification = normalizeData(glucose, hemoglobin, classification)
#centroid_array, new_classes = kMeansClustering(k, glucose_scaled, hemoglobin_scaled)
#
#unscaled_centroids = unscaledCentroids(centroid_array)
#graphingKMeans(glucose, hemoglobin, new_classes, unscaled_centroids)





#        for i in range(len(centroid_array)):
#            centroid_array[i][0] = np.mean(assignment_array[new_classes==i][:,0])
#            centroid_array[i][1] = np.mean(assignment_array[new_classes==i][:,1])
#            
#        iteration += 1
    
#def update(assignments, hemoglobin, glucose, centroids):
## updates the location of each centroid by taking the means of all features of all observations (data points) currently assigned to that centroid
## those means are then used as the features for the updated centroid location
#    my_assignments = assignments.sort()
#    K = centroids.shape[0]
#    new_centroids = np.zeros((K, 2))
#    for i in range(K):
#        if my_assignments == i:
#            new_centroids[i,1] = np.mean(glucose[my_assignments==i])
#            new_centroids[i,0] = np.mean(hemoglobin[my_assignments==i])