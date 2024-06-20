import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from Preprocess import load_and_preprocess_data
from RSLDA import RSLDA
from LDA import LDA


# 데이터 로드 및 전처리
def load_and_preprocess_data(pizza_dir, not_pizza_dir, img_size=32, sele_num=15):
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

    return images, labels


# 메인 함수
def main():
    pizza_dir = "./pizza_not_pizza/pizza"
    not_pizza_dir = "./pizza_not_pizza/not_pizza"
    X, y = load_and_preprocess_data(pizza_dir, not_pizza_dir, img_size=32, sele_num=15)

    # 데이터셋을 학습 및 테스트로 분리
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    lambda1, lambda2, dim, mu, rho = 0.001, 0.0001, 50, 1, 1.05

    print(
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

    lda = LDA()
    lda.fit(X_train_transformed, y_train)
    y_pred = lda.predict(X_test_transformed)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}")


if __name__ == "__main__":
    main()
