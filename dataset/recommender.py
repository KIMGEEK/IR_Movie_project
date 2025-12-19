import numpy as np
import pandas as pd
import sys
import os

# 데이터 로드 (탭 구분)
def load_data(path):
    df = pd.read_csv(path, sep='\t', header=None, names=['user','item','rating','timestamp'])
    df['user'] -= 1  # 0-base index
    df['item'] -= 1
    return df[['user','item','rating']]

# 평점 행렬 생성
def create_matrix(df, n_users, n_items):
    mat = np.zeros((n_users, n_items))
    for _, row in df.iterrows():
        mat[int(row["user"]), int(row["item"])] = row["rating"]
    return mat

# SVD 학습 (bias 포함)
def train_svd(train_matrix, K=50, lr=0.007, reg=0.02, epochs=20):
    n_users, n_items = train_matrix.shape
    global_mean = train_matrix[train_matrix > 0].mean()
    # 초기화
    P = np.random.normal(scale=1./K, size=(n_users, K))
    Q = np.random.normal(scale=1./K, size=(n_items, K))
    bu = np.zeros(n_users)
    bi = np.zeros(n_items)
    rows, cols = train_matrix.nonzero()
    idx = np.arange(len(rows))
    for _ in range(epochs):
        np.random.shuffle(idx) # 매 에폭마다 셔플
        for k in idx:
            u, i = rows[k], cols[k]
            r_ui = train_matrix[u, i]
            pred = global_mean + bu[u] + bi[i] + P[u].dot(Q[i])
            err  = r_ui - pred
            bu[u] += lr * (err - reg * bu[u])
            bi[i] += lr * (err - reg * bi[i])
            P[u]  += lr * (err * Q[i] - reg * P[u])
            Q[i]  += lr * (err * P[u] - reg * Q[i])
    return global_mean, bu, bi, P, Q

# 예측 및 RMSE 계산
def predict(u, i, model):
    mu, bu, bi, P, Q = model
    est = mu + bu[u] + bi[i] + P[u].dot(Q[i])
    return np.clip(est, 1.0, 5.0) # 예측값을 1~5로 클리핑

def rmse(test_df, model):
    se = 0.0; n = 0
    for _, row in test_df.iterrows():
        u, i, r = int(row["user"]), int(row["item"]), row["rating"]
        est = predict(u, i, model)
        se += (r - est)**2; n += 1
    return np.sqrt(se / n)

def main(train_path: str, test_path: str) -> None:
    train_df = load_data(train_path)
    test_df = load_data(test_path)
    n_users = max(train_df.user.max(), test_df.user.max()) + 1
    n_items = max(train_df.item.max(), test_df.item.max()) + 1
    train_matrix = create_matrix(train_df, n_users, n_items)
    model = train_svd(train_matrix, K=50, lr=0.005, reg=0.02, epochs=30)
    rmse_value = rmse(test_df, model)
    base_name = os.path.basename(train_path)
    output_file = f"{base_name}_prediction.txt"
    with open(output_file, 'w') as f:
        for _, row in test_df.iterrows():
            u_idx = int(row['user'])
            i_idx = int(row['item'])
            pre_rating = predict(u_idx, i_idx, model)
            original_user = u_idx + 1
            original_item = i_idx + 1
            f.write(f"{original_user}\t{original_item}\t{pre_rating:.5f}\n")

    print(f"RMSE:{rmse_value:.6f}")

if __name__ == '__main__':
    # Expect exactly two command-line arguments: training and test file names.
    # sys.argv[0] is the script name, so we check for length 3.
    if len(sys.argv) != 3:
        print("Usage: python recommender.py <train_file> <test_file>")
        sys.exit(1)
    _, train_file, test_file = sys.argv
    main(train_file, test_file)