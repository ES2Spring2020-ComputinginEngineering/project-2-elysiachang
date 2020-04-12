This project is based on an example and dataset from Data Science course developed at Berkeley (Data8.org).

****************

In this GitHub Project 2 folder, there are three important documents. These three documents are Python Files labeled as "KMeansClustering_driver.py", "KMeansClustering_functions.py" and "NearestNeighborClassification.py".

****************

In each respective file you will find the following information:

1. In the "NearestNeighborClassification.py" file, you will find all the code for Steps 2 and 3 of Project 2. This file contains the respective import statements at the top as well as 5 very important functions. These functions and their purposes are as follows:
- openckdfile(): unloads and opens up the data file to be used in this project
- createTestCase(): generates 2 random numbers within range to be used as a test case
- calculateDistanceArray(): takes in 4 arguments and applies the distance formula to create a new array of the distances between each data point and the test case
- nearestNeighborClassifier(): takes in 5 arguments and calculates the minimum index at which the test case is closest to its "neighbor" data point
- kNearestNeighborClassifier(): takes in 6 arguments and calculates the minimum index at which the test case is closest to "k" amount of neighbors (where k is a value specified by the user)
In the main script of this file, you will find graphs of both the Nearest Neighbor and K-Nearest Neighbor, and both of these graphs incorporate the use of if/else statements to make sure that the test case being plotted is correct amongst the data provided

2. In the "KMeansClustering_functions.py" file, you will find all the code for Step 4 of Project 2. This file contains the respective import statements at the top as well as 4 very important functions. These functions and their purposes are as follows:
- select(): picks random points from the data to make as random centroids for first trial
- assign(): reads through the ckd file and determines which data point is closest to whichever respective centroid
- update(): recomputes the centroids of the newly formed clusters by taking the mean of previous centroids and updating the rest of the data points
- iterate(): repeats the assign and update steps until an end criteria is met
These functions are based off of the K-Means Clustering algorithm which can be read more about in the text provided on the Canvas assignment. It is important to note that the stopping criteria can be one of three things: (1) all centroids didn't change during the last update, (2) all points kept the same classification during the last assign step, and (3) a maximum number of interactions happened 

3. Lastly, in the "KMeansClustering_driver.py" file, you will find all the code that is essentially the "main script" of the K-Means Clustering algorithm. The reason for keeping the functions and driver separately is so that it keeps the code clean, and it is easier to follow for both the coder and the grader. If you run this file, you will find that it gives the results of the "KMeansClustering_functions.py" file, but in a separate Python file, and all functions in this file can be called upon by using "kmc" in front of the function.

****************

If you follow all of these instructions, you will be able to easily navigate the three files in this GitHub folder. Each file produces the results you will read about in the Project 2 Report submission on Canvas. 

Additionally, if you would like more information on each respective function, there is a report that is written up in Canvas and submitted as a PDF. This report goes through each respective function in much more detail. Refer to that for more in depth description.

****************