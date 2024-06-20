from mySVD import mySVD
import numpy as np
from scipy.sparse import issparse


def PCA1(data, options=None):
    if options is None:
        options = {}

    # 축소할 차원을 설정, 기본값은 0 (축소하지 않음).
    ReducedDim = options.get("ReducedDim", 0)

    nSmp, nFea = data.shape
    # 축소 차원이 유효하지 않으면, 특징 수로 설정.
    if (ReducedDim > nFea) or (ReducedDim <= 0):
        ReducedDim = nFea

    # 희소 행렬인 경우 밀집 행렬로 변환.
    if issparse(data):
        data = data.toarray()

    # 데이터의 평균을 빼서 중심화.
    sampleMean = np.mean(data, axis=0)
    data = data - sampleMean

    # 전치된 데이터에 대해 SVD 수행.
    eigvector, eigvalue, _ = mySVD(data.T, ReducedDim)
    eigvalue = np.square(eigvalue)

    # PCARatio가 지정된 경우, 적절한 차원 수 선택.
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