from sklearn.cluster import KMeans
import numpy as np
import pickle

kmeans = pickle.load(open("Oldmodel.pkl", "rb"))
print(np.shape(kmeans.cluster_centers_))