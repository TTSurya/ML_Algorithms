import numpy as np

# K-Means
class KMeans:
    def __init__(self, n_clusters=2, max_iter=100, tol=1e-4):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X):
        np.random.seed(42)
        self.centroids = X[np.random.choice(len(X), self.n_clusters, replace=False)]
        for _ in range(self.max_iter):
            labels = np.argmin(np.linalg.norm(X[:, None] - self.centroids, axis=2), axis=1)
            new_centroids = np.array([X[labels==i].mean(axis=0) for i in range(self.n_clusters)])
            if np.all(np.abs(new_centroids - self.centroids) < self.tol): break
            self.centroids = new_centroids
        self.labels = labels

    def predict(self, X):
        return np.argmin(np.linalg.norm(X[:, None] - self.centroids, axis=2), axis=1)


# Hierarchical Agglomerative
def agglomerative(X, n_clusters=2):
    clusters = [[i] for i in range(len(X))]
    n = len(X)
    distances = np.zeros((n,n))
    for i in range(n):
        for j in range(i+1,n):
            distances[i,j] = np.linalg.norm(X[i]-X[j])
            distances[j,i] = distances[i,j]

    while len(clusters) > n_clusters:
        min_dist, to_merge = float('inf'), (0,1)
        for i in range(len(clusters)):
            for j in range(i+1,len(clusters)):
                d = np.mean([distances[p,q] for p in clusters[i] for q in clusters[j]])
                if d < min_dist: min_dist, to_merge = d, (i,j)
        i,j = to_merge
        clusters[i].extend(clusters[j])
        clusters.pop(j)

    labels = np.zeros(n)
    for idx, c in enumerate(clusters):
        for i in c: labels[i] = idx
    return labels


# PCA
def pca(X, n_components=2):
    X_centered = X - X.mean(axis=0)
    U,S,Vt = np.linalg.svd(X_centered)
    return X_centered @ Vt[:n_components].T


# Example usage
X = np.random.rand(10,3)
km = KMeans(3); km.fit(X)
labels_km = km.labels
labels_agg = agglomerative(X,3)
X_pca = pca(X,2)

print("KMeans:", labels_km)
print("Agglomerative:", labels_agg)
print("PCA reduced:", X_pca)
