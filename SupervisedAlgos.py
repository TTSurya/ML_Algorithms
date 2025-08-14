import numpy as np
from collections import Counter

class GaussianNB:
    def fit(self, X, y):
        self.classes = np.unique(y)
        self.mean = {c: X[y == c].mean(axis=0) for c in self.classes}
        self.var = {c: X[y == c].var(axis=0) for c in self.classes}
        self.priors = {c: (y == c).mean() for c in self.classes}
    def predict(self, X):
        preds = []
        for x in X:
            probs = {c: np.log(self.priors[c]) - 0.5 * np.sum(np.log(2 * np.pi * self.var[c])) - 0.5 * np.sum(((x - self.mean[c]) ** 2) / self.var[c]) for c in self.classes}
            preds.append(max(probs, key=probs.get))
        return np.array(preds)

class MultinomialNB:
    def fit(self, X, y):
        self.classes = np.unique(y)
        self.class_count = {c: (y == c).sum() for c in self.classes}
        self.feature_count = {c: X[y == c].sum(axis=0) for c in self.classes}
        self.feature_prob = {c: (self.feature_count[c] + 1) / (self.feature_count[c].sum() + X.shape[1]) for c in self.classes}
    def predict(self, X):
        preds = []
        for x in X:
            log_probs = {c: np.log(self.class_count[c] / len(X)) + np.sum(x * np.log(self.feature_prob[c])) for c in self.classes}
            preds.append(max(log_probs, key=log_probs.get))
        return np.array(preds)

class KNN:
    def __init__(self, k=3):
        self.k = k
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
    def predict(self, X):
        preds = []
        for x in X:
            distances = np.linalg.norm(self.X_train - x, axis=1)
            k_idx = np.argsort(distances)[:self.k]
            k_labels = self.y_train[k_idx]
            preds.append(np.bincount(k_labels).argmax())
        return np.array(preds)

class LinearRegression:
    def fit(self, X, y):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        self.theta = np.linalg.pinv(X_b.T @ X_b) @ X_b.T @ y
    def predict(self, X):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        return X_b @ self.theta

class LogisticRegression:
    def __init__(self, lr=0.01, n_iter=1000):
        self.lr = lr
        self.n_iter = n_iter
    def fit(self, X, y):
        self.theta = np.zeros(X.shape[1])
        for _ in range(self.n_iter):
            z = X @ self.theta
            h = 1 / (1 + np.exp(-z))
            gradient = X.T @ (h - y) / y.size
            self.theta -= self.lr * gradient
    def predict(self, X):
        return (1 / (1 + np.exp(-(X @ self.theta))) >= 0.5).astype(int)

class DecisionTree:
    def __init__(self, max_depth=5, min_samples_split=2):
        self.max_depth, self.min_samples_split = max_depth, min_samples_split
    class Node:
        def __init__(self, gini, n_samples, n_classes, predicted_class):
            self.gini, self.n_samples, self.n_classes, self.predicted_class = gini, n_samples, n_classes, predicted_class
            self.feature_index = self.threshold = self.left = self.right = None
    def _gini(self, y):
        m = y.size
        if m == 0: return 0
        probs = np.bincount(y) / m
        return 1 - np.sum(probs ** 2)
    def _best_split(self, X, y):
        m, n = X.shape
        best_gini, best_idx, best_thr = 1, None, None
        for idx in range(n):
            thresholds, classes = zip(*sorted(zip(X[:, idx], y)))
            thresholds, classes = np.array(thresholds), np.array(classes)
            for i in range(1, m):
                if thresholds[i] == thresholds[i-1]:
                    continue
                gini = (i * self._gini(classes[:i]) + (m - i) * self._gini(classes[i:])) / m
                if gini < best_gini:
                    best_gini, best_idx, best_thr = gini, idx, (thresholds[i] + thresholds[i-1]) / 2
        return best_idx, best_thr
    def _grow_tree(self, X, y, depth=0):
        num_samples_per_class = [np.sum(y == i) for i in range(np.max(y) + 1)]
        predicted_class = np.argmax(num_samples_per_class)
        node = self.Node(self._gini(y), y.size, len(num_samples_per_class), predicted_class)
        if depth < self.max_depth and y.size >= self.min_samples_split and node.gini > 0:
            idx, thr = self._best_split(X, y)
            if idx is not None:
                left_idx = X[:, idx] <= thr
                node.feature_index, node.threshold = idx, thr
                node.left = self._grow_tree(X[left_idx], y[left_idx], depth + 1)
                node.right = self._grow_tree(X[~left_idx], y[~left_idx], depth + 1)
        return node
    def fit(self, X, y):
        self.tree_ = self._grow_tree(np.asarray(X), np.asarray(y).astype(int))
        return self
    def _predict_one(self, x):
        node = self.tree_
        while node.left:
            node = node.left if x[node.feature_index] <= node.threshold else node.right
        return node.predicted_class
    def predict(self, X):
        return np.array([self._predict_one(row) for row in np.asarray(X)])

class RandomForest:
    def __init__(self, n_estimators=10, max_depth=5, min_samples_split=2, max_features=None):
        self.n_estimators, self.max_depth, self.min_samples_split, self.max_features = n_estimators, max_depth, min_samples_split, max_features
    def fit(self, X, y):
        n_samples, n_features = X.shape
        max_feat = self.max_features or int(np.sqrt(n_features))
        self.trees = []
        for _ in range(self.n_estimators):
            idxs = np.random.choice(n_samples, n_samples, replace=True)
            feat_idxs = np.random.choice(n_features, max_feat, replace=False)
            tree = DecisionTree(self.max_depth, self.min_samples_split)
            tree._selected_features = feat_idxs
            tree.fit(X[idxs][:, feat_idxs], y[idxs])
            self.trees.append(tree)
        return self
    def predict(self, X):
        preds = np.array([t.predict(X[:, t._selected_features]) for t in self.trees])
        return np.apply_along_axis(lambda x: Counter(x).most_common(1)[0][0], axis=0, arr=preds)

class LinearSVM:
    def __init__(self, lr=0.01, epochs=1000, C=1.0):
        self.lr, self.epochs, self.C = lr, epochs, C
    def fit(self, X, y):
        X, y = np.asarray(X), np.where(np.asarray(y) <= 0, -1, 1)
        n, m = X.shape
        self.w, self.b = np.zeros(m), 0
        for _ in range(self.epochs):
            for i in range(n):
                if y[i] * (np.dot(self.w, X[i]) + self.b) < 1:
                    self.w += self.lr * (self.C * y[i] * X[i] - 2 * self.w)
                    self.b += self.lr * self.C * y[i]
                else:
                    self.w += self.lr * (-2 * self.w)
        return self
    def predict(self, X):
        return (np.asarray(X) @ self.w + self.b > 0).astype(int)

if __name__ == "__main__":
    X = np.array([[1.0, 2.1], [1.3, 1.9], [3.2, 4.1], [3.0, 3.9]])
    y_cls = np.array([0, 0, 1, 1])
    y_reg = np.array([1.0, 1.2, 2.9, 3.1])

    print("GNB:", GaussianNB().fit(X, y_cls) or GaussianNB().predict(X))
    print("MNB:", MultinomialNB().fit((X*10).astype(int), y_cls) or MultinomialNB().predict((X*10).astype(int)))
    print("KNN:", KNN().fit(X, y_cls) or KNN().predict(X))
    print("LinReg:", LinearRegression().fit(X, y_reg) or LinearRegression().predict(X))
    print("LogReg:", LogisticRegression().fit(X, y_cls) or LogisticRegression().predict(X))
