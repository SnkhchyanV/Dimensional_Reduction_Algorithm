from sklearn.metrics import pairwise_distances
import numpy as np

class MDS:
  def __init__(self, n_components=2, max_iter=300, eps=0.001):
    self.__n_components = n_components
    self.__max_iter = max_iter
    self.__eps = eps
    self.embedding_ = None
    self.stress_ = None

  def fit(self, X):
    X_distances = pairwise_distances(X)
    
    new_coord = np.random.uniform(size=(X.shape[0], self.__n_components))
    alpha = 1 / X.shape[0] * np.sum(X_distances)
    
    for k in range(self.__max_iter):
      for i in range(X.shape[0]):
        stress_derivative = np.zeros_like(new_coord)
        for j in range(X.shape[0]):
          stress_derivative[i] += (X_distances[i,j] - np.linalg.norm(new_coord[i] - new_coord[j])) * -2 * (new_coord[i] - new_coord[j])


        new_coord[i] = new_coord[i] - alpha * self.__eps * stress_derivative[i]
        
    
    new_distances = pairwise_distances(new_coord)   
    self.stress_ = np.sum((X_distances-new_distances).T @ (X_distances-new_distances))
    self.embedding_ = new_coord
    
    return new_coord, new_distances


