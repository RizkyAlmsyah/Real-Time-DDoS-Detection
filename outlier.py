from scipy.spatial.distance import cdist, euclidean, pdist
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np

class OutlierDetection:

  def __init__(self, benign=None, cluster_n=None, percentile=None):
    if isinstance(benign, pd.DataFrame):
        data = benign.values
    else:
        data = benign
    self.percentile = percentile
    self.benign = data
    self.cluster_n = cluster_n
    self.kmeans = KMeans(n_clusters=self.cluster_n).fit(self.benign)
    
  def cluster(self):
    labels = self.kmeans.labels_
    centroids = self.kmeans.cluster_centers_
    
    averaged = dict(enumerate(centroids, 0))
    dic= {label: self.benign[labels==label] for label in np.unique(labels)}
    avgDistance = {}
    for i in dic:
      distance = []
      for j in dic[i]:
        distance.append(euclidean(j, averaged[i]))
      avgDistance[i] = np.percentile(distance, self.percentile)
    
    return avgDistance, averaged
  
  def predict(self, ddos=None):      
    davg, averaged = self.cluster()
    
    anomaly = []
    if isinstance(ddos, pd.DataFrame):
        data = ddos.values
    else:
        data = ddos
        
    for i in data:
        labels = self.kmeans.predict([i])
        dist = euclidean(i, averaged[labels[0]])
        if dist <= davg[labels[0]]:
            anomaly.append(1)
        else:
            anomaly.append(-1) 
    return anomaly 