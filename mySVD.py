import numpy as np
from scipy.sparse import issparse


def mySVD(X, ReducedDim=0):
    MAX_MATRIX_SIZE = 1600  # 기계의 계산 능력에 따라 이 숫자를 변경할 수 있습니다
    EIGVECTOR_RATIO = 0.1  # 기계의 계산 능력에 따라 이 숫자를 변경할 수 있습니다

    nSmp, mFea = X.shape
    if mFea / nSmp > 1.0713:
        ddata = X @ X.T
        ddata = np.maximum(ddata, ddata.T)

        dimMatrix = ddata.shape[0]
        if (
            (ReducedDim > 0)
            and (dimMatrix > MAX_MATRIX_SIZE)
            and (ReducedDim < dimMatrix * EIGVECTOR_RATIO)
        ):
            eigvalue, U = eigs(ddata, k=ReducedDim, which="LM")
        else:
            if issparse(ddata):
                ddata = ddata.toarray()

            eigvalue, U = np.linalg.eigh(ddata)
            idx = np.argsort(-eigvalue)
            eigvalue = eigvalue[idx]
            U = U[:, idx]

        maxEigValue = np.max(np.abs(eigvalue))
        eigIdx = np.where(np.abs(eigvalue) / maxEigValue < 1e-10)[0]
        eigvalue = np.delete(eigvalue, eigIdx)
        U = np.delete(U, eigIdx, axis=1)

        if (ReducedDim > 0) and (ReducedDim < len(eigvalue)):
            eigvalue = eigvalue[:ReducedDim]
            U = U[:, :ReducedDim]

        eigvalue_Half = np.sqrt(eigvalue)
        S = np.diag(eigvalue_Half)

        if nSmp >= 3:
            eigvalue_MinusHalf = 1.0 / eigvalue_Half
            V = X.T @ (U * eigvalue_MinusHalf)
    else:
        ddata = X.T @ X
        ddata = np.maximum(ddata, ddata.T)

        dimMatrix = ddata.shape[0]
        if (
            (ReducedDim > 0)
            and (dimMatrix > MAX_MATRIX_SIZE)
            and (ReducedDim < dimMatrix * EIGVECTOR_RATIO)
        ):
            eigvalue, V = eigs(ddata, k=ReducedDim, which="LM")
        else:
            if issparse(ddata):
                ddata = ddata.toarray()

            eigvalue, V = np.linalg.eigh(ddata)
            idx = np.argsort(-eigvalue)
            eigvalue = eigvalue[idx]
            V = V[:, idx]

        maxEigValue = np.max(np.abs(eigvalue))
        eigIdx = np.where(np.abs(eigvalue) / maxEigValue < 1e-10)[0]
        eigvalue = np.delete(eigvalue, eigIdx)
        V = np.delete(V, eigIdx, axis=1)

        if (ReducedDim > 0) and (ReducedDim < len(eigvalue)):
            eigvalue = eigvalue[:ReducedDim]
            V = V[:, :ReducedDim]

        eigvalue_Half = np.sqrt(eigvalue)
        S = np.diag(eigvalue_Half)

        eigvalue_MinusHalf = 1.0 / eigvalue_Half
        U = X @ (V * eigvalue_MinusHalf)

    return U, S, V
