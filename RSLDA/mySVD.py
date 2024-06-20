import numpy as np
from scipy.sparse import issparse
from scipy.sparse.linalg import eigs

def mySVD(X, ReducedDim=0):
    MAX_MATRIX_SIZE = 1600  # 기계의 계산 능력에 따라 이 숫자를 변경할 수 있습니다
    EIGVECTOR_RATIO = 0.1  # 기계의 계산 능력에 따라 이 숫자를 변경할 수 있습니다

    nSmp, mFea = X.shape  # 샘플 수와 특징 수
    if mFea / nSmp > 1.0713:  # 특징 수가 샘플 수보다 많은 경우
        ddata = X @ X.T  # X * X^T 계산
        ddata = np.maximum(ddata, ddata.T)  # 대칭 행렬로 변환

        dimMatrix = ddata.shape[0]  # 행렬의 차원
        if (
            (ReducedDim > 0)  # 축소 차원이 설정된 경우
            and (dimMatrix > MAX_MATRIX_SIZE)  # 행렬 크기가 최대 크기보다 큰 경우
            and (ReducedDim < dimMatrix * EIGVECTOR_RATIO)  # 축소 차원이 특정 비율보다 작은 경우
        ):
            eigvalue, U = eigs(ddata, k=ReducedDim, which="LM")  # 가장 큰 k개의 고유값 및 고유벡터 계산
        else:
            if issparse(ddata):  # 희소 행렬인 경우
                ddata = ddata.toarray()  # 밀집 행렬로 변환

            eigvalue, U = np.linalg.eigh(ddata)  # 고유값 분해
            idx = np.argsort(-eigvalue)  # 고유값을 내림차순으로 정렬하는 인덱스
            eigvalue = eigvalue[idx]  # 고유값 정렬
            U = U[:, idx]  # 고유벡터 정렬

        maxEigValue = np.max(np.abs(eigvalue))  # 고유값의 최대 절대값
        eigIdx = np.where(np.abs(eigvalue) / maxEigValue < 1e-10)[0]  # 작은 고유값 인덱스 찾기
        eigvalue = np.delete(eigvalue, eigIdx)  # 작은 고유값 제거
        U = np.delete(U, eigIdx, axis=1)  # 작은 고유값에 해당하는 고유벡터 제거

        if (ReducedDim > 0) and (ReducedDim < len(eigvalue)):  # 축소 차원이 설정된 경우
            eigvalue = eigvalue[:ReducedDim]  # 축소 차원까지의 고유값 선택
            U = U[:, :ReducedDim]  # 축소 차원까지의 고유벡터 선택

        eigvalue_Half = np.sqrt(eigvalue)  # 고유값의 제곱근
        S = np.diag(eigvalue_Half)  # 고유값의 제곱근을 대각 행렬로 변환

        if nSmp >= 3:  # 샘플 수가 3 이상인 경우
            eigvalue_MinusHalf = 1.0 / eigvalue_Half  # 고유값 제곱근의 역수
            V = X.T @ (U * eigvalue_MinusHalf)  # 고유벡터 계산
    else:  # 샘플 수가 특징 수보다 많은 경우
        ddata = X.T @ X  # X^T * X 계산
        ddata = np.maximum(ddata, ddata.T)  # 대칭 행렬로 변환

        dimMatrix = ddata.shape[0]  # 행렬의 차원
        if (
            (ReducedDim > 0)  # 축소 차원이 설정된 경우
            and (dimMatrix > MAX_MATRIX_SIZE)  # 행렬 크기가 최대 크기보다 큰 경우
            and (ReducedDim < dimMatrix * EIGVECTOR_RATIO)  # 축소 차원이 특정 비율보다 작은 경우
        ):
            eigvalue, V = eigs(ddata, k=ReducedDim, which="LM")  # 가장 큰 k개의 고유값 및 고유벡터 계산
        else:
            if issparse(ddata):  # 희소 행렬인 경우
                ddata = ddata.toarray()  # 밀집 행렬로 변환

            eigvalue, V = np.linalg.eigh(ddata)  # 고유값 분해
            idx = np.argsort(-eigvalue)  # 고유값을 내림차순으로 정렬하는 인덱스
            eigvalue = eigvalue[idx]  # 고유값 정렬
            V = V[:, idx]  # 고유벡터 정렬

        maxEigValue = np.max(np.abs(eigvalue))  # 고유값의 최대 절대값
        eigIdx = np.where(np.abs(eigvalue) / maxEigValue < 1e-10)[0]  # 작은 고유값 인덱스 찾기
        eigvalue = np.delete(eigvalue, eigIdx)  # 작은 고유값 제거
        V = np.delete(V, eigIdx, axis=1)  # 작은 고유값에 해당하는 고유벡터 제거

        if (ReducedDim > 0) and (ReducedDim < len(eigvalue)):  # 축소 차원이 설정된 경우
            eigvalue = eigvalue[:ReducedDim]  # 축소 차원까지의 고유값 선택
            V = V[:, :ReducedDim]  # 축소 차원까지의 고유벡터 선택

        eigvalue_Half = np.sqrt(eigvalue)  # 고유값의 제곱근
        S = np.diag(eigvalue_Half)  # 고유값의 제곱근을 대각 행렬로 변환

        eigvalue_MinusHalf = 1.0 / eigvalue_Half  # 고유값 제곱근의 역수
        U = X @ (V * eigvalue_MinusHalf)  # 고유벡터 계산

    return U, S, V  # 고유벡터, 고유값, 고유벡터 반환
