import numpy as np
import random
import sys
from sklearn.metrics import accuracy_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from tqdm.auto import tqdm
import logging
from Preprocess import load_and_preprocess_data
from RSLDA import RSLDA
from LDA import LDA

# 로거 설정
logger = logging.getLogger()
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler(sys.stdout)
file_handler = logging.FileHandler("./log/output.log")
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)
logger.addHandler(console_handler)
logger.addHandler(file_handler)


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
            logger=logger,
        )
        X_train_transformed = np.dot(Q.T, X_train.T).T
        X_test_transformed = np.dot(Q.T, X_test.T).T
        lda = LDA()
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
