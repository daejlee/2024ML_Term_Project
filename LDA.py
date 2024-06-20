import numpy as np


class LDA:
    def __init__(self):
        self.means_ = None  # 각 클래스의 평균 벡터
        self.priors_ = None  # 각 클래스의 사전 확률
        self.scalings_ = None  # 스케일링 벡터 (선형 변환 행렬)
        self.intercept_ = None  # 상수 항 (절편)
        self.classes_ = None  # 클래스 라벨

    def fit(self, X, y):
        self.classes_ = np.unique(y)  # 고유한 클래스 라벨 추출
        n_classes = len(self.classes_)  # 클래스의 수
        n_features = X.shape[1]  # 특징의 수

        mean_overall = np.mean(X, axis=0)  # 전체 데이터의 평균 계산
        self.means_ = []  # 각 클래스의 평균을 저장할 리스트
        Sw = np.zeros((n_features, n_features))  # 클래스 내 산포 행렬 초기화
        Sb = np.zeros((n_features, n_features))  # 클래스 간 산포 행렬 초기화

        for c in self.classes_:
            X_c = X[y == c]  # 클래스 c에 속하는 샘플 추출
            mean_c = np.mean(X_c, axis=0)  # 클래스 c의 평균 계산
            self.means_.append(mean_c)  # 클래스 c의 평균을 리스트에 추가
            Sw += (X_c - mean_c).T @ (X_c - mean_c)  # 클래스 내 산포 행렬 갱신
            n_c = X_c.shape[0]  # 클래스 c에 속하는 샘플의 수
            mean_diff = (mean_c - mean_overall).reshape(n_features, 1)  # 클래스 평균과 전체 평균의 차이
            Sb += n_c * (mean_diff @ mean_diff.T)  # 클래스 간 산포 행렬 갱신

        # Sw^-1 * Sb 행렬의 고유값 문제를 해결
        eigvals, eigvecs = np.linalg.eig(np.linalg.inv(Sw) @ Sb)
        idx = np.argsort(abs(eigvals))[::-1]  # 고유값을 내림차순으로 정렬하는 인덱스
        eigvecs = eigvecs[:, idx]  # 고유벡터를 고유값의 내림차순으로 정렬
        self.scalings_ = eigvecs[:, : n_classes - 1]  # 가장 큰 (n_classes - 1)개의 고유벡터 선택

        self.means_ = np.array(self.means_)  # 리스트를 numpy 배열로 변환
        self.priors_ = np.array([np.mean(y == c) for c in self.classes_])  # 각 클래스의 사전 확률 계산

        # 절편 계산
        self.intercept_ = -0.5 * np.sum(self.means_ @ self.scalings_, axis=1)

    def transform(self, X):
        return X @ self.scalings_  # 데이터를 LDA 변환

    def predict(self, X):
        X_lda = self.transform(X)  # 입력 데이터를 LDA 변환
        scores = [
            X_lda @ mean.T + intercept
            for mean, intercept in zip(self.means_ @ self.scalings_, self.intercept_)
        ]
        scores = np.array(scores).T  # 각 클래스에 대한 점수를 계산
        return self.classes_[np.argmax(scores, axis=1)]  # 가장 높은 점수를 받은 클래스를 예측
