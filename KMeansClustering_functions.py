#Please place your FUNCTION code for step 4 here.

# IMPORT STATEMENTS
import numpy as np
import matplotlib.pyplot as plt
import random
#from copy import deepcopy

# FUNCTIONS
def openckdfile():
    glucose, hemoglobin, classification = np.loadtxt('ckd.csv', delimiter=',', skiprows=1, unpack=True)
    return glucose, hemoglobin, classification

# JENN'S STARTER CODE
def select(K):
    return np.random.random((K, 2))

def assign(centroids, hemoglobin, glucose):
    K = centroids.shape[0]
    distances = np.zeros((K, len(hemoglobin)))
    for i in range(K):
        g = centroids[i,1]
        h = centroids[i,0]
        distances[i] = np.sqrt((hemoglobin-h)**2+(glucose-g)**2)
    assignments = np.argmin(distances, axis = 0)
    return assignments

def update(assignments, hemoglobin, glucose, centroids):
    my_assignments = assignments.sort()
    K = centroids.shape[0]
    new_centroids = np.zeros((K, 2))
    for i in range(K):
        if my_assignments==i:
            new_centroids[i,1]=np.mean(glucose[my_assignments==i])
            new_centroids[i,0]=np.mean(hemoglobin[my_assignments==i])
    return new_centroids

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

def iterate(assignments, new_centroids):
    i = 0
    while i < 3: # ask an input for the number of times you want to iterate at the start of program
        assignments = assign(centroids, hemoglobin, glucose)
        new_centroids = update(assignments, hemoglobin, glucose, centroids)
        i += 1
    return assignments, new_centroids

def graphingKMeans(glucose, hemoglobin, assignment, new_centroids):
    plt.figure()
    for i in range(assignment.max()+1):
        rcolor = np.random.rand(3,)
        plt.plot(hemoglobin[assignment==i],glucose[assignment==i], ".", label = "Class " + str(i), color = rcolor)
        plt.plot(new_centroids[i, 0], new_centroids[i, 1], "D", label = "Centroid " + str(i), color = rcolor)
    plt.xlabel("Hemoglobin")
    plt.ylabel("Glucose")
    plt.legend()
    plt.show() # graph looks pretty random (step 1), it'll look better when you fix update (step 3)

def positivesNegatives(hemoglobin, glucose, classification, assignments):
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

# MAIN SCRIPT - PUT ALL OF THIS INTO DRIVER
glucose, hemoglobin, classification = openckdfile()

# NORMALIZE DATA
glucose = (glucose-70)/(490-70)
hemoglobin = (hemoglobin-3.1)/(17.8-3.1)

# PLOT GRAPH
plt.figure()
plt.plot(hemoglobin[classification==1],glucose[classification==1], "k.", label = "not CKD")
plt.plot(hemoglobin[classification==0],glucose[classification==0], "r.", label = "CKD")
plt.xlabel("Hemoglobin")
plt.ylabel("Glucose")
plt.legend()
plt.show()

# VARIABLE ASSIGNMENT
#K = 10
centroids = select(10)
assignments = assign(centroids, hemoglobin, glucose)
#old_centroids = deepcopy(centroids)
#centroids = update(centroids)
new_centroids = update(assignments, hemoglobin, glucose, centroids)
iterate_assignments = iterate(assignments, new_centroids)[0]
iterate_centroids = iterate(assignments, new_centroids)[1]

# FINAL GRAPH
graphingKMeans(glucose, hemoglobin, assignments, new_centroids)

# POSITIVES AND NEGATIVES
truePositives = ((positivesNegatives(hemoglobin, glucose, classification, assignments)[0])/158)*100
falsePositives = ((positivesNegatives(hemoglobin, glucose, classification, assignments)[1])/158)*100
trueNegatives = ((positivesNegatives(hemoglobin, glucose, classification, assignments)[2])/158)*100
falseNegatives = ((positivesNegatives(hemoglobin, glucose, classification, assignments)[3])/158)*100