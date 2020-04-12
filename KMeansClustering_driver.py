#Please place your FUNCTION code for step 4 here.
import KMeansClustering_functions as kmc #Use kmc to call your functions

glucose, hemoglobin, classification = kmc.openckdfile() # what is the purpose of this driver

glucose = (glucose-70)/(490-70)
hemoglobin = (hemoglobin-3.1)/(17.8-3.1)

centroids = kmc.select(10)

assignments = kmc.assign(centroids, hemoglobin, glucose)

updates = kmc.update(assignments, hemoglobin, glucose, centroids)

iterate_assignments = kmc.iterate(centroids)[0]

iterate_centroids = kmc.iterate(centroids)[1]

kmc.graphingKMeans(glucose, hemoglobin, assignments, centroids)

plusminus = kmc.positivesNegatives(hemoglobin, glucose, classification, assignments)