import numpy as np
from tqdm.auto import tqdm
from ScatterMat import ScatterMat
from PCA1 import PCA1


def RSLDA(
    X=None,  # 입력 데이터 행렬 (샘플 수 x 특징 수)
    label=None,  # 각 샘플 클래스 레이블
    lambda1=0.00001,  # 정규화 파라미터
    lambda2=0.0001,  # "
    dim=115,  # 축소할 차원 수
    mu=0.1,  # ADMM 알고리즘 업데이트 파라미터 (μ)
    rho=1.01,  # "
    max_iter=100,  # 최대 반복 횟수
    logger=None,
):
    print("RUNNING RSLDA (Serial)!")
    logger.info(
        f"Testing with params: lambda1={lambda1}, lambda2={lambda2}, dim={dim}, mu={mu}, rho={rho}"
    )
    m, n = X.shape  # 입력 데이터 행렬 행과 열의 수
    max_mu = 10**5  # μ의 최대값.

    # Initialization
    regu = 10**-5  # 작은 정규화 상수
    Sw, Sb = ScatterMat(X, label)  # 클래스 내 / 클래스 간 산포 행렬
    options = {"ReducedDim": dim}  # PCA1 옵션
    P1, _ = PCA1(X.T, options)  # 초기 PCA 변환 행렬
    Q = np.ones((m, dim))  # 투영 행렬 초기화
    E = np.zeros((m, n))  # 잡음 행렬 초기화
    Y = np.zeros((m, n))  # 라그랑지 승수 초기화
    v = np.sqrt(np.sum(Q * Q, axis=1) + np.finfo(float).eps)  # 투영 행렬 행 벡터들의 l2-노름
    D = np.diag(1.0 / v)  # v의 역수로 대각 행렬 생성

    obj = []

    for iter in tqdm(range(1, max_iter + 1), total=max_iter):
        # Update P
        if iter == 1:
            P = P1
        else:
            M = X - E + Y / mu
            U1, S1, V1 = np.linalg.svd(M @ X.T @ Q, full_matrices=False)
            P = U1 @ V1

        # Update Q
        M = X - E + Y / mu
        Q1 = 2 * (Sw - regu * Sb) + lambda1 * D + mu * X @ X.T
        Q2 = mu * X @ M.T @ P
        Q = np.linalg.solve(Q1, Q2)
        v = np.sqrt(np.sum(Q * Q, axis=1) + np.finfo(float).eps)
        D = np.diag(1.0 / v)

        # Update E: 잡음 행렬 업데이트
        eps1 = lambda2 / mu
        temp_E = X - P @ Q.T @ X + Y / mu
        E = np.maximum(0, temp_E - eps1) + np.minimum(0, temp_E + eps1)

        # Update Y, mu
        Y = Y + mu * (X - P @ Q.T @ X - E)  # 라그랑지 승수 행렬 업데이트
        mu = min(rho * mu, max_mu)  # mu 증가시키기 (max_mu보다 작아야 함)
        leq = X - P @ Q.T @ X - E  # 현 제약 조건 위반 정도 계산
        EE = np.sum(np.abs(E), axis=1)  # 잡음 행렬 E의 절대값 합 계산
        obj_value = (
            np.trace(Q.T @ (Sw - regu * Sb) @ Q)
            + lambda1 * np.sum(v)
            + lambda2 * np.sum(EE)
        )  # 목적 함수 계산하여 현 반복에서의 목적 함수 값 저장
        obj.append(obj_value)

        if iter > 2:  # 제약 조건 위반 정도와 목적 함수 값이 일정 수준 이하로 떨어지면 반복 중단
            if (
                np.linalg.norm(leq, np.inf) < 10**-7
                and abs(obj[-1] - obj[-2]) < 0.00001
            ):
                logger.info(f"Early stopping at iteration {iter}")
                break
    return (
        P,  # 학습된 변환 행렬 P
        Q,  # 투영 행렬 Q
        E,  # 잡음 행렬 E
        obj,  # 최종 목적 함수 값 반환
    )
