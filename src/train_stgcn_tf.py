import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

class CrimeDataset(Dataset):
    def __init__(self, X, Y, A_crime, OD):
        """
        X: (num_samples, T, N, F)
        Y: (num_samples, N)
        A_crime: (num_samples, N, N)
        OD: (num_samples, N, 4)
        """
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)
        self.A_crime = torch.tensor(A_crime, dtype=torch.float32)
        self.OD = torch.tensor(OD, dtype=torch.float32)

    def __len__(self):
        return min(len(self.X), len(self.A_crime), len(self.Y))

    def __getitem__(self, idx):
        return (self.X[idx], self.A_crime[idx], self.OD[idx], self.Y[idx])

# ----------------------------
# 图融合自注意力模块
# ----------------------------
class GraphFusionAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.att = nn.Linear(hidden_dim, 1)

    def forward(self, x1, x2, x3):
        # x: (B,T,N,H)
        s1 = self.att(x1)
        s2 = self.att(x2)
        s3 = self.att(x3)

        score = torch.cat([s1, s2, s3], dim=-1)  # (B,T,N,3)
        weight = torch.softmax(score, dim=-1)

        w1 = weight[..., 0].unsqueeze(-1)
        w2 = weight[..., 1].unsqueeze(-1)
        w3 = weight[..., 2].unsqueeze(-1)

        out = w1 * x1 + w2 * x2 + w3 * x3
        return out

# ----------------------------
# STGCN 块
# ----------------------------
class STGCNBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.gcn = nn.Linear(in_dim, out_dim)
        self.temporal = nn.Conv2d(
            out_dim,
            out_dim,
            kernel_size=(3,1),
            padding=(1,0)
        )
    
    def forward(self, x, A):
        # print("after matmul", x.mean(), x.std())
        # x: (B,T,N,F)
        if A.dim() == 2:
            x = torch.einsum('ij,btjf->btif', A, x)
        elif A.dim() == 3:
            x = torch.einsum('bij,btjf->btif', A, x)

        # A = A.squeeze(1)
        # print("A shape:", A.shape)
        # print("X shape:", x.shape)

        # x = torch.matmul(A, x)                # (B,T,N,F)

        x = self.gcn(x)
        x = x.permute(0,3,1,2)                # (B,F,T,N)
        x = self.temporal(x)
        x = x.permute(0,2,3,1)                # (B,T,N,F)

        return x

# ----------------------------
# Temporal Attention
# ----------------------------
class TemporalAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.att = nn.Linear(channels,1)
    
    def forward(self, x):
        # x: (B,T,N,C)
        score = self.att(x)            # (B,T,N,1)
        weight = torch.softmax(score, dim=1)
        out = (x * weight).sum(dim=1)  # sum时间维 -> (B,N,C)
        return out

# ----------------------------
# Decoupled_HG-STGCN 模型
# ----------------------------
class Decoupled_STGCN_ZINB(nn.Module):
    def __init__(self, in_dim, static_idx, hidden_dim=64):
        super().__init__()
        # gate fusion 
        self.fusion_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )
        self.static_idx = static_idx
        self.alpha_static = nn.Parameter(torch.tensor(0.5))
        self.alpha_dynamic = nn.Parameter(torch.tensor(0.5))
        
        # --- 静态支路：处理环境背景 ---
        self.static_net = nn.Sequential(
            nn.Linear(static_idx, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # --- 动态支路：处理天气、时滞、动态传播 ---
        dynamic_in_dim = in_dim - static_idx
        self.dynamic_stgcn = STGCNBlock(dynamic_in_dim, hidden_dim)
        
        # --- 图融合与注意力 ---
        self.graph_fusion = GraphFusionAttention(hidden_dim)
        self.temporal_att = TemporalAttention(hidden_dim)
        
        # --- ZINB 输出层 ---
        # 融合后的维度是 hidden_dim (static) + hidden_dim (dynamic)
        combine_dim = hidden_dim
        self.fc_pi = nn.Sequential(nn.Linear(combine_dim, 1), nn.Sigmoid())     # 零值概率
        self.fc_mu = nn.Sequential(nn.Linear(combine_dim, 1), nn.Softplus())    # 均值
        self.fc_theta = nn.Sequential(nn.Linear(combine_dim, 1), nn.Softplus()) # 离散度
    def build_od_graph(self, od_feat, K=10):
        # od_feat: (B,N,4)
        norm = torch.norm(od_feat, dim=-1, keepdim=True)
        od_norm = od_feat / (norm + 1e-6)

        sim = torch.matmul(od_norm, od_norm.transpose(1,2))  # (B,N,N)

        topk_val, topk_idx = torch.topk(sim, K, dim=-1)

        A = torch.zeros_like(sim)
        A.scatter_(-1, topk_idx, topk_val)

        return A / (A.sum(dim=-1, keepdim=True) + 1e-6)
    
    def forward(self, X, A1, A2, A3, A_hg, OD):
        # X: (B, T, N, F)
        # OD: (B, N, 4)

        # 1. 解耦特征
        # 静态特征取序列最后时刻即可 (因为它们在时间维是重复的)
        x_static = X[:, -1, :, :self.static_idx]      # (B, N, F_static)
        x_dynamic = X[:, :, :, self.static_idx:]     # (B, T, N, F_dynamic)
        
        # 2. 静态支路 (融合超图)
        alpha_s = torch.sigmoid(self.alpha_static)
        A_static = alpha_s * A1 + (1 - alpha_s) * A2          # (B, N, H)
        
        h_static = self.static_net(x_static) # 特征编码
        h_static = torch.matmul(A_static, h_static) # 空间传播
        A_hg = A_hg / (A_hg.sum(dim=-1, keepdim=True) + 1e-6)
        h_static = torch.matmul(A_hg, h_static)
        
        # 3. 动态支路 (原有 STGCN 逻辑)
        
        # === 机制建模 ===
        # ====== OD gating ======
        od_strength = torch.mean(OD, dim=-1, keepdim=True)   # (B,N,1)
        # 控制范围（关键）
        od_gate = torch.sigmoid(od_strength)
        # ====== gating crime graph ======
        # print("OD gate matrix shape:", od_gate_matrix.shape)
        A_dynamic = A3 * (1 + od_gate)
        # 归一化（必须）
        A_dynamic = A_dynamic / (A_dynamic.sum(dim=-1, keepdim=True) + 1e-6)

        # STGCN
        # h_dynamic = self.dynamic_stgcn(x_dynamic, A_dynamic)
        # h_dynamic = self.temporal_att(h_dynamic)
        
        h_dyn1 = self.dynamic_stgcn(x_dynamic, A1)
        h_dyn2 = self.dynamic_stgcn(x_dynamic, A2)
        h_dyn3 = self.dynamic_stgcn(x_dynamic, A_dynamic)
        h_dynamic = self.graph_fusion(h_dyn1, h_dyn2, h_dyn3)
        h_dynamic = self.temporal_att(h_dynamic)      # (B, N, H)
        
        # 4. 特征拼接 (解耦融合)
        h_cat = torch.cat([h_static, h_dynamic], dim=-1)
        g = self.fusion_gate(h_cat)   # (B,N,H)
        h_final = g * h_static + (1 - g) * h_dynamic
        h_final = h_static + h_final # 残差连接，增强静态特征的稳定性 保证 静态特征信息不会被 Gate 完全压制
        
        # 5. ZINB 参数预测
        pi = self.fc_pi(h_final).squeeze(-1)
        mu = self.fc_mu(h_final).squeeze(-1)
        theta = self.fc_theta(h_final).squeeze(-1)
        # 加入解耦约束
        return pi, mu, theta, h_static, h_dynamic


# =============================
# 1.加载数据
# =============================
X = np.load("data/processed/X.npy")   # (T', N, F)
Y = np.load("data/processed/Y.npy")   # (T', N)
OD = np.load("data/processed/dynamic_od_flow_1246.npy")  # (T', N, 4)
OD = np.log1p(OD) # 标准化

A_spatial = np.load("data/processed/adj_adaptive.npy")
A_distance = np.load("data/processed/adj_distance.npy")
A_crime_dynamic = np.load("data/processed/adj_crime_dynamic_gaussian.npy")
A_hg = np.load("data/processed/adj_hypergraph.npy")

# print("A_spatial max", A_spatial.max())
# print("A_distance max", A_distance.max())
# print("A_crime max", A_crime_dynamic.max())
# print(np.isnan(A_crime_dynamic).sum())
# print(np.isinf(A_crime_dynamic).sum())

# -----------------------------
# 2.滑动窗口预测下一天
# -----------------------------
window = 30
crime_lag = 7
offset = window - crime_lag
# 每个样本时间维度 = window
# 对应标签 Y = 窗口末一天
X_window = []
Y_window = []

for i in range(len(X)-offset):
    X_window.append(X[i:i+offset])
    Y_window.append(Y[i+offset])   # 下一天
X_window = np.stack(X_window, axis=0).astype(np.float32)  # (num_samples, offset, N, F)
Y_window = np.stack(Y_window, axis=0).astype(np.float32)  # (num_samples, N)

print("X_window:", X_window.shape)
print("Y_window:", Y_window.shape)
print("A_crime_dynamic:", A_crime_dynamic.shape)

# -----------------------------
# 3.划分数据集
# -----------------------------
num_samples = X_window.shape[0]
train_ratio = 0.7
val_ratio = 0.15

train_end = int(num_samples * train_ratio)
val_end = int(num_samples * (train_ratio + val_ratio))

X_train = X_window[:train_end]
Y_train = Y_window[:train_end]

X_val = X_window[train_end:val_end]
Y_val = Y_window[train_end:val_end]

X_test = X_window[val_end:]
Y_test = Y_window[val_end:]

# 3.对应动态犯罪图也要切片
A_crime_train = A_crime_dynamic[window:window+train_end]
A_crime_val = A_crime_dynamic[window+train_end:window+val_end]
A_crime_test = A_crime_dynamic[window+val_end:window+num_samples]

# OD 流切片
OD_train = OD[window:window+train_end]
OD_val = OD[window+train_end:window+val_end]
OD_test = OD[window+val_end:window+num_samples]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = 4

train_dataset = CrimeDataset(X_train, Y_train, A_crime_train, OD_train)
val_dataset   = CrimeDataset(X_val, Y_val, A_crime_val, OD_val)
test_dataset  = CrimeDataset(X_test, Y_test, A_crime_test, OD_test)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# -----------------------------
# 4.将静态图与超图转 tensor 并发送到设备
# -----------------------------
A_spatial = torch.tensor(A_spatial).float().to(device)
A_distance = torch.tensor(A_distance).float().to(device)
A_hg = torch.tensor(A_hg).float().to(device)

# -----------------------------
# 5.初始化模型
# -----------------------------
static_idx = 24
model = Decoupled_STGCN_ZINB(X.shape[2], static_idx).to(device)


# =============================
# 6.训练循环模板（稳定版）
# =============================
# 在 train_model 内部改写：
def zinb_loss(y_true, pi, mu, theta):
    eps = 1e-8
    # clamp避免数值爆炸
    pi = torch.clamp(pi, eps, 1 - eps)
    mu = torch.clamp(mu, eps, 1e6)
    theta = torch.clamp(theta, eps, 1e6)

    # log NB likelihood
    t1 = torch.lgamma(theta + y_true)
    t2 = torch.lgamma(theta)
    t3 = torch.lgamma(y_true + 1)

    log_nb = (
        t1 - t2 - t3 +
        theta * (torch.log(theta) - torch.log(theta + mu)) +
        y_true * (torch.log(mu) - torch.log(theta + mu))
    )
    # zero case
    log_zero_nb = theta * (torch.log(theta) - torch.log(theta + mu))
    # log likelihood
    zero_case = torch.log(
        pi + (1 - pi) * torch.exp(log_zero_nb) + eps
    )
    non_zero_case = torch.log(1 - pi + eps) + log_nb
    result = torch.where(
        y_true < 1e-6,
        zero_case,
        non_zero_case
    )
    return -torch.mean(result)


def train_model(model, train_loader, val_loader, A_spatial, A_distance,A_hg,
                device, epochs=100, lr=1e-3, clip_norm=5.0):

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=5
    )
    

    for epoch in range(epochs):
        model.train()
        train_losses = []

        for X_batch, A_crime_batch,OD_batch, Y_batch in train_loader:
            X_batch = X_batch.to(device)
            Y_batch = Y_batch.to(device)
            A_crime_batch = A_crime_batch.to(device)
            OD_batch = OD_batch.to(device)
            # 检查 NaN / 极值
            if torch.isnan(A_crime_batch).any():
                A_crime_batch = torch.nan_to_num(A_crime_batch, nan=0.0)

            optimizer.zero_grad()
            # 训练步：
            pi, mu, theta,h_static, h_dynamic = model(X_batch, A_spatial, A_distance, A_crime_batch, A_hg, OD_batch)
            # 防止梯度爆炸 
            mu = torch.clamp(mu, max=100)
            theta = torch.clamp(theta, max=100)
            # reshape: (B,N,H) -> (B*N,H)
            h_s = h_static.reshape(-1, h_static.shape[-1])
            h_d = h_dynamic.reshape(-1, h_dynamic.shape[-1])
            loss_orth = torch.norm(torch.matmul(h_s.T, h_d), p='fro') / h_s.shape[0]
            loss = zinb_loss(Y_batch, pi, mu, theta) + 1e-3 * loss_orth
            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)

            optimizer.step()
            train_losses.append(loss.item())

        # 验证
        model.eval()
        val_losses = []
        with torch.no_grad():
            for X_batch, A_crime_batch, OD_batch, Y_batch in val_loader:
                X_batch = X_batch.to(device)
                Y_batch = Y_batch.to(device)
                A_crime_batch = A_crime_batch.to(device)
                OD_batch = OD_batch.to(device)

                if torch.isnan(A_crime_batch).any():
                    A_crime_batch = torch.nan_to_num(A_crime_batch, nan=0.0)
                pi, mu, theta, h_static, h_dynamic = model(X_batch, A_spatial, A_distance, A_crime_batch, A_hg, OD_batch)
                mu = torch.clamp(mu, max=100)
                theta = torch.clamp(theta, max=100)
                loss = zinb_loss(Y_batch, pi, mu, theta)
                val_losses.append(loss.item())

        mean_train = np.mean(train_losses)
        mean_val = np.mean(val_losses)

        print(f"Epoch {epoch} | Train Loss {mean_train:.4f} | Val Loss {mean_val:.4f}")

        # 调整学习率
        scheduler.step(mean_val)

    return model

# =============================
# 7.测试函数
# =============================
# 构建新的指标：Top-k、PAI、Jaccard
def calculate_advanced_metrics(y_true, y_pred, k_percent=0.1):
    """
    y_true/y_pred: (N,) or (samples, N)
    k_percent: 选取前百分之多少的网格作为热点 (例如 10%)
    """
    N = y_true.shape[-1]
    k = int(N * k_percent)
    
    # 1. Top-K Hit Rate
    # 找到预测值最高的前 k 个索引
    top_k_pred_idx = np.argsort(y_pred, axis=-1)[:, -k:]
    hits = 0
    total_crimes = np.sum(y_true > 0)
    
    for i in range(len(y_true)):
        # 实际发生犯罪的网格中，有多少在预测的前k里
        true_indices = np.where(y_true[i] > 0)[0]
        hits += len(np.intersect1d(top_k_pred_idx[i], true_indices))
    
    # hit_rate = hits / (total_crimes + 1e-6)
    hit_rate = hits / (len(y_true)*k)

    # 2. PAI (Predictive Accuracy Index)
    # PAI = (n/N) / (a/A) -> (捕获犯罪比例) / (热点面积比例)
    area_ratio = k / N
    captured_crime_ratio = 0
    for i in range(len(y_true)):
        total_sample_crimes = np.sum(y_true[i])
        captured_crimes = np.sum(y_true[i][top_k_pred_idx[i]])
        captured_crime_ratio += (captured_crimes / (total_sample_crimes + 1e-6))
    
    pai = (captured_crime_ratio / len(y_true)) / area_ratio

    # 3. Jaccard (热点重叠指数)
    # 将真实值前 k 也设为热点，计算交并比
    top_k_true_idx = np.argsort(y_true, axis=-1)[:, -k:]
    jaccard_sum = 0
    for i in range(len(y_true)):
        intersection = len(np.intersect1d(top_k_pred_idx[i], top_k_true_idx[i]))
        union = len(np.union1d(top_k_pred_idx[i], top_k_true_idx[i]))
        jaccard_sum += intersection / (union + 1e-6)
    
    return hit_rate, pai, jaccard_sum / len(y_true)



def test_model(model, test_loader, A_spatial, A_distance, A_hg, device):
    model.eval()
    # A_spatial = A_spatial.to(device)
    # A_distance = A_distance.to(device)
    preds, targets = [], []

    with torch.no_grad():
        for X_batch, A_crime_batch,OD_batch, Y_batch in test_loader:
            X_batch = X_batch.to(device)
            Y_batch = Y_batch.to(device)
            A_crime_batch = A_crime_batch.to(device)
            OD_batch = OD_batch.to(device)

            if torch.isnan(A_crime_batch).any():
                A_crime_batch = torch.nan_to_num(A_crime_batch, nan=0.0)

            pi, mu, theta,h_static, h_dynamic = model(X_batch, A_spatial, A_distance, A_crime_batch, A_hg,OD_batch)
            pred = torch.clamp((1 - pi) * mu, min=0)
            preds.append(pred.detach().cpu().numpy())
            targets.append(Y_batch.detach().cpu().numpy())

    pred_test = np.vstack(preds)
    Y_test_all = np.vstack(targets)

    rmse = np.sqrt(mean_squared_error(Y_test_all.flatten(), pred_test.flatten()))
    mae = mean_absolute_error(Y_test_all.flatten(), pred_test.flatten())
    hit_rate, pai, jaccard = calculate_advanced_metrics(Y_test_all, pred_test)

    print("Test RMSE:", rmse)
    print("Test MAE:", mae)
    print("Test Hit Rate:", hit_rate)
    print("Test PAI:", pai)
    print("Test Jaccard:", jaccard)

    return pred_test, Y_test_all

train_model(model=model,train_loader=train_loader,val_loader=val_loader,A_spatial=A_spatial,A_distance=A_distance,A_hg=A_hg,device=device)
test_model(model=model,test_loader=test_loader,A_spatial=A_spatial,A_distance=A_distance,A_hg=A_hg,device=device)
