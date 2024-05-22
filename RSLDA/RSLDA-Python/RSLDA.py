from PCA1 import PCA1
from ScatterMat import ScatterMat

import os
import pickle
import numpy as np

from tqdm.auto import tqdm


def RSLDA(
    X=None,  # 입력 데이터 행렬 (샘플 수 x 특징 수)
    label=None,  # 각 샘플 클래스 레이블
    lambda1=0.0002,  # 정규화 파라미터
    lambda2=0.001,  # "
    dim=100,  # 축소할 차원 수
    mu=0.1,  # ADMM 알고리즘 업데이트 파라미터 (μ)
    rho=1.01,  # "
    max_iter=100,  # 최대 반복 횟수
):
    print("RUNNING RSLDA!")
    m, n = X.shape  # 입력 데이터 행렬 행과 열의 수
    max_mu = 10**5  # μ의 최대값.

    # Initialization
    regu = 10**-5  # 작은 정규화 상수
    Sw, Sb = ScatterMat(X, label)  # 클래스 내 / 클래스 간 산포 행렬
    options = {}  # PCA1 옵션
    options["ReducedDim"] = dim
    P1, _ = PCA1(X.T, options)  # 초기 PCA 변환 행렬
    Q = np.ones((m, dim))  # 투영 행렬 초기화
    E = np.zeros((m, n))  # 잡음 행렬 초기화
    Y = np.zeros((m, n))  # 라그랑지 승수 초기화
    v = np.sqrt(
        np.sum(Q * Q, axis=1) + np.finfo(float).eps
    )  # 투영 행렬 행 벡터들의 l2-노름
    D = np.diag(1.0 / v)  # v의 역수로 대각 행렬 생성

    # Main loop
    for iter in tqdm(range(1, max_iter + 1), total=max_iter):

        # Update P
        if iter == 1:  # 첫 반복에서는 P를 초기 PCA 변환 행렬로 설정
            P = P1
        else:  # 그 외 반복에서는 M을 계산하여 SVD를 수행하여 P를 업데이트
            M = X - E + Y / mu
            U1, S1, V1 = np.linalg.svd(M @ X.T @ Q, full_matrices=False)
            P = U1 @ V1
            del M

        # Update Q
        M = X - E + Y / mu
        Q1 = 2 * (Sw - regu * Sb) + lambda1 * D + mu * X @ X.T
        Q2 = mu * X @ M.T @ P
        Q = np.linalg.solve(
            Q1, Q2
        )  # Q1과 Q2를 계산한 후 행렬 방정식을 풀어 Q를 업데이트
        v = np.sqrt(np.sum(Q * Q, axis=1) + np.finfo(float).eps)
        D = np.diag(1.0 / v)  # v, D를 업데이트하여 다음 반복에 사용할 준비

        # Update E: 잡음 행렬 업데이트
        eps1 = lambda2 / mu
        temp_E = X - P @ Q.T @ X + Y / mu
        E = np.maximum(0, temp_E - eps1) + np.minimum(0, temp_E + eps1)

        # Update Y, mu
        Y = Y + mu * (X - P @ Q.T @ X - E)  # 라그랑지 승수 행렬 업데이트
        mu = min(rho * mu, max_mu)  # mu 증가시키기 (max_mu보다 작아야 함)
        leq = X - P @ Q.T @ X - E  # 현 제약 조건 위반 정도 계산
        EE = np.sum(np.abs(E), axis=1)  # 잡음 행렬 E의 절대값 합 계산
        obj = (
            np.trace(Q.T @ (Sw - regu * Sb) @ Q)
            + lambda1 * np.sum(v)
            + lambda2 * np.sum(EE)
        )  # 목적 함수 계산하여 현 반복에서의 목적 함수 값 저장

        if (
            iter > 2
        ):  # 제약 조건 위반 정도와 목적 함수 값이 일정 수준 이하로 떨어지면 반복 중단
            if np.linalg.norm(leq, np.inf) < 10**-7 and abs(obj - obj_prev) < 0.00001:
                print(iter)
                break

        obj_prev = obj  # 이전 반복에서의 목적 함수 값 저장

    return (
        P,  # 학습된 변환 행렬 P
        Q,  # 투영 행렬 Q
        E,  # 잡음 행렬 E
        obj,  # 최종 목적 함수 값 반환
    )


# 특징 정렬 함수
def sort_power_of_features(Q):  # Q 각 행의 l2-노름을 계산하여 내림차순으로 정렬
    row_norm = np.linalg.norm(Q, axis=1)
    sorted_power = np.argsort(row_norm)[::-1]  # DESC
    return sorted_power


if __name__ == "__main__":
    # DATA_FOLDER = "P:/RESEARCH/DATA/CIFAR-100/CIFARDB/train/"

    features = pickle.load(open("./database/features_handcrafted_Cifar.pkl", "rb"))
    paths = pickle.load(open("./database/paths_handcrafted_Cifar.pkl", "rb"))

    # X = np.array(features)
    y = []  # 클래스 레이블 저장할 리스트

    # class_check = 'apple'
    # for i in range(len(paths)):
    #     path = paths[i]
    #     label =  0
    #     if class_check in path:
    #         label = 1
    #     y.append(label)

    X_train = []  # 훈련 데이터 샘플 저장할 리스트
    n_pos = 0  # 양성 및 음성 샘플 수 변수
    n_neg = 0  # "

    class_check = "apple"
    for i in range(len(paths)):  # 조건에 맞는 샘플을 X_train과 y에 저장
        if n_pos >= 10 and n_neg >= 100:
            break
        else:
            path = paths[i]
            if class_check in path and n_pos < 10:
                label = 1
                n_pos += 1
                y.append(label)
                X_train.append(features[i])
            elif class_check not in path and n_neg < 50:
                label = 0
                n_neg += 1
                y.append(label)
                X_train.append(features[i])

    X_train = np.array(X_train)
    X_train = X_train.T
    # X = X.T
    y = np.array(y)
    y = y.reshape(-1, 1)

    print(X_train.shape)
    print(y.shape)
    P, Q, E, obj = RSLDA(X=X_train, label=y)
    print("Shape of P = ", P.shape)
    print("Shape of Q = ", Q.shape)
    print("Shape of E = ", E.shape)

    # pickle.dump(Q, open("./database/Q.pkl", 'wb'))

    # TEST
    # Q = pickle.load(open("./database/Q.pkl", 'rb'))
    # print(Q.shape)
    # print(np.max(Q))
    # print(np.min(Q))
    # a = np.count_nonzero(Q)
    # print(a)
    # sort_power_of_features(Q=Q)
    # print(Q.shape)
