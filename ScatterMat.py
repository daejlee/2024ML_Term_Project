import numpy as np


def ScatterMat(X, y):
    dim, _ = X.shape
    nclass = int(np.max(y)) + 1

    mean_X = np.mean(X, axis=1, keepdims=True)
    Sw = np.zeros((dim, dim))
    Sb = np.zeros((dim, dim))

    for i in range(nclass):
        inx_i = np.where(y == i)[0]
        X_i = X[:, inx_i]

        mean_Xi = np.mean(X_i, axis=1, keepdims=True)
        Sw += np.cov(X_i, rowvar=True, bias=True)
        Sb += len(inx_i) * (mean_Xi - mean_X) @ (mean_Xi - mean_X).T

    return Sw, Sb
