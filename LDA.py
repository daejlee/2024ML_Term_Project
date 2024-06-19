import numpy as np


class LDA:
    def __init__(self):
        self.means_ = None
        self.priors_ = None
        self.scalings_ = None
        self.intercept_ = None
        self.classes_ = None

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        n_features = X.shape[1]

        mean_overall = np.mean(X, axis=0)
        self.means_ = []
        Sw = np.zeros((n_features, n_features))
        Sb = np.zeros((n_features, n_features))

        for c in self.classes_:
            X_c = X[y == c]
            mean_c = np.mean(X_c, axis=0)
            self.means_.append(mean_c)
            Sw += (X_c - mean_c).T @ (X_c - mean_c)
            n_c = X_c.shape[0]
            mean_diff = (mean_c - mean_overall).reshape(n_features, 1)
            Sb += n_c * (mean_diff @ mean_diff.T)

        # Solve the eigenvalue problem for the matrix Sw^-1 * Sb
        eigvals, eigvecs = np.linalg.eig(np.linalg.inv(Sw) @ Sb)
        idx = np.argsort(abs(eigvals))[::-1]
        eigvecs = eigvecs[:, idx]
        self.scalings_ = eigvecs[:, : n_classes - 1]

        self.means_ = np.array(self.means_)
        self.priors_ = np.array([np.mean(y == c) for c in self.classes_])

        # Calculate intercept
        self.intercept_ = -0.5 * np.sum(self.means_ @ self.scalings_, axis=1)

    def transform(self, X):
        return X @ self.scalings_

    def predict(self, X):
        X_lda = self.transform(X)
        scores = [
            X_lda @ mean.T + intercept
            for mean, intercept in zip(self.means_ @ self.scalings_, self.intercept_)
        ]
        scores = np.array(scores).T
        return self.classes_[np.argmax(scores, axis=1)]
