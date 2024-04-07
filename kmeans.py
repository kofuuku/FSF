import numpy as np
import matplotlib.pyplot as plt


class KmeansClustering:

    def __init__(self,k=3):     #constructor to set the K (no. of clusters)
        self.k=k
        self.centroids=None

    #static method as its just a calculation
    @staticmethod
    def euclidean_distance(data_point,centroids):
        #to find the distacnne between one data point and all the centroids
        return np.sqrt(np.sum((centroids-data_point)**2, axis=1))    #the formula to find euclidean distance between one data point and n centroids


    def fit(self,X,max_iterations=200):
        #randomly initalize the centroids first
        #we sue numpy to generate random centroids
        self.centroids = np.random.uniform(np.amin(X, axis=0), np.amax(X, axis=0),
                                           size=(self.k, X.shape[1]))   #centroid needs to be in the boundary of the data that we have   
                                                                        #the uniform function gives it a minimum and a maximum

        #creating the clusters
        for _ in range(max_iterations):
            y=[]

            for data_point in X: 
                distances = KmeansClustering.euclidean_distance(data_point, self.centroids)  #running the function for a datapoint
                cluster_num = np.argmin(distances)      #argmin returns the index of the smallest distance which will be assigned to one cluster 
                y.append(cluster_num)                      #giving us the index of the centroid of the cluster 

            y=np.array(y)



        #readjust the centroid positions based on these indexes  

        cluster_indices = []  

        #forloop to readjust the centroid positions till the error decreases to less than 0.001
        for i in range(self.k):
            cluster_indices.append(np.argwhere(y==i))    #preparing for the next evaluation

        cluster_centers = []

        for i, indices in enumerate(cluster_indices):    #repositioning the centroids
            if len(indices) == 0:   #if k is too large we could endup with empty clusters
                cluster_centers.append(self.centroids[i])    #so if there are no members, well just take the current centroid
                                                                # and set it as the new centroid

            else:
                cluster_centers.append(np.mean(X[indices], axis=0)[0])  #else find the centroid of a cluster with members


        if np.max(self.centroids - np.array(cluster_centers))<0.001 :   #if the diffrence between the centroids is minimum then break.
            return 0
        else:
            self.centroids = np.array(cluster_centers) #else reposition
        
        return y    
                

from sklearn.datasets import make_blobs
#using a random dataset to test the algorithm

data = make_blobs(n_samples=50, n_features=2, centers=3)   
random_points = data[0]

kmeans = KmeansClustering(k=3)

labels = kmeans.fit(random_points)

plt.scatter(random_points[:,0],random_points[:,1],c=labels)
plt.scatter(kmeans.centroids[:,0], kmeans.centroids[:,1],c=range(len(kmeans.centroids)),
            marker="*",s=100)

plt.show()
