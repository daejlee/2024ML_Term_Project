import os
import cv2
import numpy as np
import random
import sys
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from tqdm.auto import tqdm
from scipy.sparse import issparse
from concurrent.futures import ThreadPoolExecutor
import logging


# 로거 설정
logger = logging.getLogger()
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler(sys.stdout)
file_handler = logging.FileHandler("output.log")
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)
logger.addHandler(console_handler)
logger.addHandler(file_handler)


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


def RSLDA(
    X=None,  # 입력 데이터 행렬 (샘플 수 x 특징 수)
    label=None,  # 각 샘플 클래스 레이블
    lambda1=0.00001,  # 정규화 파라미터
    lambda2=0.0001,  # "
    dim=115,  # 축소할 차원 수
    mu=0.1,  # ADMM 알고리즘 업데이트 파라미터 (μ)
    rho=1.01,  # "
    max_iter=100,  # 최대 반복 횟수
):
    print("RUNNING RSLDA (Serial)!")
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


# 데이터 로드 및 전처리
def load_and_preprocess_data(pizza_dir, not_pizza_dir, img_size=64, sele_num=15):
    images = []
    labels = []

    # 피자 이미지 불러오기
    for filename in os.listdir(pizza_dir):
        if filename.endswith(".jpg"):
            img = cv2.imread(os.path.join(pizza_dir, filename))
            if img is not None:
                img = cv2.resize(img, (img_size, img_size)).flatten()
                images.append(img)
                labels.append(1)

    # 피자가 아닌 이미지 불러오기
    for filename in os.listdir(not_pizza_dir):
        if filename.endswith(".jpg"):
            img = cv2.imread(os.path.join(not_pizza_dir, filename))
            if img is not None:
                img = cv2.resize(img, (img_size, img_size)).flatten()
                images.append(img)
                labels.append(0)

    # 데이터를 numpy 배열로 변환
    images = np.array(images)
    labels = np.array(labels)

    # 데이터 정규화
    scaler = StandardScaler()
    images = scaler.fit_transform(images)

    # 학습 데이터와 테스트 데이터로 분리
    X_train, X_test, y_train, y_test = train_test_split(
        images, labels, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test


# 메인 함수
def main():
    pizza_dir = "./pizza_not_pizza/pizza"
    not_pizza_dir = "./pizza_not_pizza/not_pizza"
    X_train, X_test, y_train, y_test = load_and_preprocess_data(
        pizza_dir, not_pizza_dir, img_size=32, sele_num=15
    )

    # 하이퍼파라미터 설정
    lambda1_values = [0.00001, 0.0001, 0.001]
    lambda2_values = [0.0001, 0.001, 0.01]
    dim_values = [50, 100, 150]
    mu_values = [0.01, 0.1, 1]
    rho_values = [1.01, 1.05, 1.1]

    param_combinations = 50  # 임의로 50개의 하이퍼파라미터 조합을 선택
    early_stopping_rounds = 5  # 조기 중단 조건: 성능이 개선되지 않는 최대 반복 횟수
    best_accuracy = 0
    best_params = {}
    no_improvement_count = 0

    for _ in range(param_combinations):
        lambda1 = random.choice(lambda1_values)
        lambda2 = random.choice(lambda2_values)
        dim = random.choice(dim_values)
        mu = random.choice(mu_values)
        rho = random.choice(rho_values)

        logger.info(
            f"Testing with params: lambda1={lambda1}, lambda2={lambda2}, dim={dim}, mu={mu}, rho={rho}"
        )
        P, Q, E, obj = RSLDA(
            X=X_train.T,
            label=y_train,
            lambda1=lambda1,
            lambda2=lambda2,
            dim=dim,
            mu=mu,
            rho=rho,
        )
        X_train_transformed = np.dot(Q.T, X_train.T).T
        X_test_transformed = np.dot(Q.T, X_test.T).T
        lda = LinearDiscriminantAnalysis()
        lda.fit(X_train_transformed, y_train)
        y_pred = lda.predict(X_test_transformed)
        accuracy = accuracy_score(y_test, y_pred)
        logger.info(f"Accuracy: {accuracy}")

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_params = {
                "lambda1": lambda1,
                "lambda2": lambda2,
                "dim": dim,
                "mu": mu,
                "rho": rho,
            }
            no_improvement_count = 0
        else:
            no_improvement_count += 1

        if no_improvement_count >= early_stopping_rounds:
            logger.info(
                f"Early stopping triggered after {early_stopping_rounds} rounds with no improvement."
            )
            break

    logger.info(f"Best accuracy: {best_accuracy}")
    logger.info(f"Best parameters: {best_params}")


if __name__ == "__main__":
    main()
