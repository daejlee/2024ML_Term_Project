import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tqdm.auto import tqdm
from scipy.sparse import issparse
from concurrent.futures import ThreadPoolExecutor

# ScatterMat 함수 정의
def ScatterMat(X, y):
    dim, _ = X.shape
    nclass = int(np.max(y)) + 1
    
    mean_X = np.mean(X, axis=1)
    Sw = np.zeros((dim, dim))
    Sb = np.zeros((dim, dim))
    
    for i in range(nclass):
        inx_i = np.where(y == i)[0]
        X_i = X[:, inx_i]
        
        mean_Xi = np.mean(X_i, axis=1)
        Sw += np.cov(X_i, rowvar=True, bias=True)
        Sb += len(inx_i) * (mean_Xi - mean_X)[:, None] * (mean_Xi - mean_X)[None, :]
        
    return Sw, Sb

# PCA1 함수 정의
def mySVD(X, ReducedDim):
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    return U[:, :ReducedDim], S[:ReducedDim], Vt[:ReducedDim, :]

def PCA1(data, options=None):
    if options is None:
        options = {}

    ReducedDim = 0
    if 'ReducedDim' in options:
        ReducedDim = options['ReducedDim']

    nSmp, nFea = data.shape
    if (ReducedDim > nFea) or (ReducedDim <= 0):
        ReducedDim = nFea

    if issparse(data):
        data = data.toarray()
    sampleMean = np.mean(data, axis=0)
    data = (data - np.tile(sampleMean, (nSmp, 1)))

    eigvector, eigvalue, _ = mySVD(data.T, ReducedDim)
    eigvalue = np.square(eigvalue)

    if 'PCARatio' in options:
        sumEig = np.sum(eigvalue)
        sumEig *= options['PCARatio']
        sumNow = 0
        for idx in range(len(eigvalue)):
            sumNow += eigvalue[idx]
            if sumNow >= sumEig:
                break
        eigvector = eigvector[:, :idx]

    return eigvector, eigvalue

# RSLDA 함수 정의
def RSLDA(
    X=None,  # 입력 데이터 행렬 (샘플 수 x 특징 수)
    label=None,  # 각 샘플 클래스 레이블
    lambda1=0.0002,  # 정규화 파라미터
    lambda2=0.001,  # "
    dim=100,  # 축소할 차원 수
    mu=0.1,  # ADMM 알고리즘 업데이트 파라미터 (μ)
    rho=1.01,  # "
    max_iter=50,  # 최대 반복 횟수 (기존 100에서 50으로 감소)
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

    def update_P(iter, P, M):
        if iter == 1:
            return P1
        else:
            U1, S1, V1 = np.linalg.svd(M @ X.T @ Q, full_matrices=False)
            return U1 @ V1

    def update_Q(Q1, Q2):
        return np.linalg.solve(Q1, Q2)

    def update_E(temp_E, eps1):
        return np.maximum(0, temp_E - eps1) + np.minimum(0, temp_E + eps1)

    with ThreadPoolExecutor() as executor:
        for iter in tqdm(range(1, max_iter + 1), total=max_iter):
            M = X - E + Y / mu
            P_future = executor.submit(update_P, iter, P1, M)
            Q1 = 2 * (Sw - regu * Sb) + lambda1 * D + mu * X @ X.T
            Q2 = mu * X @ M.T @ P1
            Q_future = executor.submit(update_Q, Q1, Q2)
            P = P_future.result()
            Q = Q_future.result()
            v = np.sqrt(np.sum(Q * Q, axis=1) + np.finfo(float).eps)
            D = np.diag(1.0 / v)
            
            # Update E: 잡음 행렬 업데이트
            eps1 = lambda2 / mu
            temp_E = X - P @ Q.T @ X + Y / mu
            E_future = executor.submit(update_E, temp_E, eps1)
            E = E_future.result()
            
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

# 이미지 디렉토리 경로
pizza_dir = "./pizza_not_pizza/pizza"
not_pizza_dir = "./pizza_not_pizza/not_pizza"

# 이미지 데이터와 라벨을 저장할 리스트
images = []
labels = []

# 피자 이미지 불러오기
for filename in os.listdir(pizza_dir):
    if filename.endswith(".jpg"):
        img = cv2.imread(os.path.join(pizza_dir, filename))
        if img is not None:
            img = cv2.resize(img, (32, 32)).flatten()  # 이미지 크기 32x32로 줄임
            images.append(img)
            labels.append(1)

# 피자가 아닌 이미지 불러오기
for filename in os.listdir(not_pizza_dir):
    if filename.endswith(".jpg"):
        img = cv2.imread(os.path.join(not_pizza_dir, filename))
        if img is not None:
            img = cv2.resize(img, (32, 32)).flatten()  # 이미지 크기 32x32로 줄임
            images.append(img)
            labels.append(0)

# 데이터를 numpy 배열로 변환
images = np.array(images)
labels = np.array(labels)

# 학습 데이터와 테스트 데이터로 분리
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# RSLDA 실행
X_train_T = X_train.T  # X_train을 전치하여 RSLDA에 전달
P, Q, E, obj = RSLDA(X=X_train_T, label=y_train)

# 변환된 데이터 확인
print("Shape of P =", P.shape)
print("Shape of Q =", Q.shape)
print("Shape of E =", E.shape)

# 변환된 데이터로 모델 학습 및 평가
X_train_transformed = np.dot(Q.T, X_train_T).T  # 투영된 데이터
X_test_T = X_test.T
X_test_transformed = np.dot(Q.T, X_test_T).T  # 테스트 데이터 투영

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

lda = LinearDiscriminantAnalysis()
lda.fit(X_train_transformed, y_train)
y_pred = lda.predict(X_test_transformed)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
