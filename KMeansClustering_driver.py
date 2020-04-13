# ELYSIA CHANG
# ES2 PROJECT 2
# APRIL 13, 2020
# K-MEANS CLUSTERING DRIVER

# ******************************

# Please place your FUNCTION code for step 4 here.

import KMeansClustering_functions as kmc # Use kmc to call your functions

# MAIN SCRIPT
glucose, hemoglobin, classification = kmc.openckdfile()

# NORMALIZE DATA
glucose_scaled, hemoglobin_scaled, classification = kmc.normalizeData(glucose, hemoglobin, classification)

# GENERATE CENTROIDS
new_centroids = kmc.generateCentroids(2)

# CREATE ASSIGNMENTS
assignments = kmc.assign(new_centroids, hemoglobin_scaled, glucose_scaled)

# UPDATE CENTROID LOCATIONS
updated_centroids = kmc.update(assignments, glucose_scaled, hemoglobin_scaled, new_centroids)

# ITERATE A MAXIMUM NUMBER OF TIMES
assignments, updated_centroids = kmc.iterationData(assignments, updated_centroids)

# GRAPH THE DATA
kmc.graphingkMeans(glucose_scaled, hemoglobin_scaled, assignments, updated_centroids)

# TRUE/FALSE POSITIVES/NEGATIVES
plusminus = kmc.positivesNegatives(classification, assignments)
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