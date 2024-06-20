from mySVD import mySVD
import numpy as np
from scipy.sparse import issparse


def PCA1(data, options=None):
    if options is None:
        options = {}

    ReducedDim = options.get("ReducedDim", 0)

    nSmp, nFea = data.shape
    if (ReducedDim > nFea) or (ReducedDim <= 0):
        ReducedDim = nFea

    if issparse(data):
        data = data.toarray()

    sampleMean = np.mean(data, axis=0)
    data = data - sampleMean

    eigvector, eigvalue, _ = mySVD(data.T, ReducedDim)
    eigvalue = np.square(eigvalue)

    if "PCARatio" in options:
        sumEig = np.sum(eigvalue)
        sumEig *= options["PCARatio"]
        sumNow = 0
        for idx in range(len(eigvalue)):
            sumNow += eigvalue[idx]
            if sumNow >= sumEig:
                break
        eigvector = eigvector[:, : idx + 1]

    return eigvector, eigvalue
