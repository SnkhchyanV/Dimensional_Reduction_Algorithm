import numpy as np
from sklearn.metrics import pairwise_distances

class t_SNE:
    def __init__(self, n_components=2, perplexity=25, learning_rate=0.01, n_epochs=200):
        self.perplexity = perplexity
        self.n_components = n_components
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.affinities = None
        self.affinities_Y = None

    def __pairwise_affinities(self, norms):
        affinities = np.zeros_like(norms)
        sigma = 2 * self.perplexity
        for i in range(norms.shape[0]):
            numerator = np.exp(-norms[i] / sigma)
            numerator[i] = 0
            denominator = np.sum(numerator)
            affinities[i] = numerator / denominator

        return affinities

    def fit_transform(self, X):
        n, m = X.shape

        distance_matrix = pairwise_distances(X)
        norms = distance_matrix
        #norms = np.power(distance_matrix, 2)
        self.affinities = self.__pairwise_affinities(norms)

        Y = np.random.randn(n, self.n_components)
        gains = np.ones((n, self.n_components))

        for k in range(self.n_epochs):
            distances_Y = pairwise_distances(Y)
            #norms_Y = np.power(distances_Y, 2)
            norms_Y = distances_Y
            self.affinities_Y = self.__pairwise_affinities(norms_Y)

            grad = np.zeros((n, self.n_components))
            for i in range(n):
                diff = self.affinities_Y[i] - self.affinities[i]
                grad[i] = 4.0 * np.dot(diff, Y[i] - Y)

            gains = (gains + 0.2) * ((grad > 0.0) != (gains > 0.0)) + (gains * 0.8) * (
                        (grad > 0.0) == (gains > 0.0))
            gains[gains < 0.01] = 0.01
            Y = Y - self.learning_rate * (gains * grad)
            if k == 250:
                self.learning_rate /= 2.0

        return self.affinities, self.affinities_Y

    def KL_divergence(self, P, Q):
        return np.sum(P * np.log(P / Q))
