import numpy as np


def ScatterMat(X, y):
    dim, _ = X.shape
    nclass = int(np.max(y)) + 1  # 클래스 수

    mean_X = np.mean(X, axis=1, keepdims=True)  # 전체 데이터의 평균
    Sw = np.zeros((dim, dim))  # 클래스 내 산포 행렬 초기화
    Sb = np.zeros((dim, dim))  # 클래스 간 산포 행렬 초기화

    for i in range(nclass):
        inx_i = np.where(y == i)[0]  # 클래스 i에 해당하는 데이터 인덱스
        X_i = X[:, inx_i]  # 클래스 i에 속하는 데이터 포인트들

        mean_Xi = np.mean(X_i, axis=1, keepdims=True)  # 클래스 i 데이터의 평균
        Sw += np.cov(X_i, rowvar=True, bias=True)  # 클래스 i의 공분산 행렬을 Sw에 추가
        Sb += len(inx_i) * (mean_Xi - mean_X) @ (mean_Xi - mean_X).T  # 클래스 간 산포 행렬 Sb에 추가

    return Sw, Sb