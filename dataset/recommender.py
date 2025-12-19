import numpy as np
import pandas as pd

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

# 예시 사용
train_df = load_data('u1.base')
test_df  = load_data('u1.test')
n_users = max(train_df.user.max(), test_df.user.max()) + 1
n_items = max(train_df.item.max(), test_df.item.max()) + 1
train_matrix = create_matrix(train_df, n_users, n_items)
model = train_svd(train_matrix, K=50, lr=0.005, reg=0.02, epochs=30)
print("Fold1 RMSE:", rmse(test_df, model))
