import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


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
