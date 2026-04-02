"""
期刊论文 Baseline 模型实现
包含传统方法、深度时空学习方法
用于与主模型(SpatioTemporalTransformer)对比
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple


# ================================
# 1. 传统方法 (非深度学习)
# ================================

class HistoricalAverage:
    """
    历史平均基线
    用过去同一时间段的平均值作为预测
    """

    def __init__(self, window_size: int = 7):
        self.window_size = window_size
        self.history = None

    def fit(self, X_train: np.ndarray, Y_train: np.ndarray):
        """
        X_train: (T, N, F) - 这里只用Y_train
        Y_train: (T, N) - 历史犯罪数
        """
        self.history = Y_train
        self.mean_crime = Y_train.mean(axis=0)  # (N,)

    def predict(self, X_test: np.ndarray, return_expected: bool = True) -> np.ndarray:
        """
        返回与深度学习模型一致的格式 (pi, mu, theta)
        简化为确定性预测
        """
        T_test = X_test.shape[0]
        N = self.mean_crime.shape[0]

        # 使用历史同期平均值
        if self.history is not None and len(self.history) >= self.window_size:
            # 取最后window_size天的平均
            recent_mean = self.history[-self.window_size:].mean(axis=0)
            pred = np.tile(recent_mean, (T_test, 1))  # (T_test, N)
        else:
            pred = np.tile(self.mean_crime, (T_test, 1))

        if return_expected:
            # 返回ZINB格式的dummy值
            pi = np.zeros((T_test, N))
            mu = pred
            theta = np.ones((T_test, N)) * 10
            return pi, mu, theta

        return pred


class RandomForestPredictor:
    """
    随机森林基线
    将时空特征展平后使用RF回归
    """

    def __init__(self, n_estimators: int = 200, max_depth: int = 20):
        from sklearn.ensemble import RandomForestRegressor
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            n_jobs=-1,
            random_state=42
        )
        self.is_fitted = False

    def fit(self, X_train: np.ndarray, Y_train: np.ndarray):
        """
        X_train: (T, N, F) - 需要展平为 (T*N, F)
        Y_train: (T, N) - 展平为 (T*N,)
        """
        T, N, F = X_train.shape

        # 展平
        X_flat = X_train.reshape(-1, F)  # (T*N, F)
        Y_flat = Y_train.reshape(-1)      # (T*N,)

        self.model.fit(X_flat, Y_flat)
        self.is_fitted = True
        self.N = N

    def predict(self, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """返回ZINB格式"""
        T, N, F = X_test.shape
        X_flat = X_test.reshape(-1, F)

        pred = self.model.predict(X_flat).reshape(T, N)

        # 包装为ZINB格式
        pi = np.zeros((T, N))
        mu = np.maximum(pred, 0.01)  # 保证正值
        theta = np.ones((T, N)) * 5

        return pi, mu, theta


# ================================
# 2. 深度学习 Baselines
# ================================

class ConvLSTMCell(nn.Module):
    """ConvLSTM单元"""

    def __init__(self, input_dim: int, hidden_dim: int, kernel_size: int = 3):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2

        self.conv = nn.Conv2d(
            in_channels=input_dim + hidden_dim,
            out_channels=4 * hidden_dim,  # i, f, g, o
            kernel_size=kernel_size,
            padding=self.padding
        )

    def forward(self, x, h_prev, c_prev):
        """
        x: (B, C, H, W) - H=N, W=1 或展平的空间
        h_prev, c_prev: (B, hidden_dim, H, W)
        """
        combined = torch.cat([x, h_prev], dim=1)
        conv_output = self.conv(combined)
        cc_i, cc_f, cc_g, cc_o = torch.split(conv_output, self.hidden_dim, dim=1)

        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        g = torch.tanh(cc_g)
        o = torch.sigmoid(cc_o)

        c = f * c_prev + i * g
        h = o * torch.tanh(c)

        return h, c


class ConvLSTM(nn.Module):
    """
    ConvLSTM Baseline
    纯CNN-RNN，不使用图结构
    """

    def __init__(self, input_dim: int, hidden_dim: int = 64,
                 num_layers: int = 2, kernel_size: int = 3,
                 dropout: float = 0.1):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # 编码器
        self.encoder = nn.Sequential(
            nn.Conv2d(input_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(hidden_dim)
        )

        # ConvLSTM层
        self.lstm_cells = nn.ModuleList([
            ConvLSTMCell(hidden_dim, hidden_dim, kernel_size)
            for _ in range(num_layers)
        ])

        self.dropout = nn.Dropout(dropout)

        # ZINB输出头
        self.fc_pi = nn.Linear(hidden_dim, 1)
        self.fc_mu = nn.Linear(hidden_dim, 1)
        self.fc_theta = nn.Linear(hidden_dim, 1)

    def forward(self, X, return_hidden: bool = False):
        """
        X: (B, T, N, F) - 将N视为空间H，F视为通道C
        """
        B, T, N, F = X.shape

        # 调整维度: (B, T, N, F) -> (B, T, F, N, 1)
        X = X.permute(0, 1, 3, 2).unsqueeze(-1)  # (B, T, F, N, 1)

        # 初始化隐藏状态
        h_states = [torch.zeros(B, self.hidden_dim, N, 1, device=X.device)
                    for _ in range(self.num_layers)]
        c_states = [torch.zeros(B, self.hidden_dim, N, 1, device=X.device)
                    for _ in range(self.num_layers)]

        # 时间步迭代
        for t in range(T):
            x_t = X[:, t]  # (B, F, N, 1)
            x_t = self.encoder(x_t)  # (B, hidden_dim, N, 1)

            for layer_idx, lstm_cell in enumerate(self.lstm_cells):
                h_states[layer_idx], c_states[layer_idx] = lstm_cell(
                    x_t, h_states[layer_idx], c_states[layer_idx]
                )
                x_t = h_states[layer_idx]

        # 取最后时刻的隐藏状态
        h_final = h_states[-1].squeeze(-1).permute(0, 2, 1)  # (B, N, hidden_dim)
        h_final = self.dropout(h_final)

        # ZINB输出
        pi = torch.sigmoid(self.fc_pi(h_final)).squeeze(-1)
        mu = F.softplus(self.fc_mu(h_final)).squeeze(-1)
        theta = F.softplus(self.fc_theta(h_final)).squeeze(-1)

        if return_hidden:
            return pi, mu, theta, h_final
        return pi, mu, theta


class STGCN(nn.Module):
    """
    ST-GCN Baseline
    经典的时空图卷积网络 (基于Chebyshev图卷积)
    """

    def __init__(self, input_dim: int, hidden_dim: int = 64,
                 num_layers: int = 3, kernel_size: int = 3,
                 dropout: float = 0.1):
        super().__init__()

        self.hidden_dim = hidden_dim

        # 时间卷积 (1D Conv)
        self.temporal_convs = nn.ModuleList()
        self.spatial_convs = nn.ModuleList()
        self.layer_norms = nn.ModuleList()

        in_dim = input_dim
        for i in range(num_layers):
            # 时间卷积
            self.temporal_convs.append(
                nn.Conv2d(in_dim, hidden_dim, (kernel_size, 1), padding=(kernel_size//2, 0))
            )
            # 空间卷积 (简化为线性变换，实际应使用Chebyshev)
            self.spatial_convs.append(
                nn.Linear(hidden_dim, hidden_dim)
            )
            self.layer_norms.append(nn.LayerNorm(hidden_dim))
            in_dim = hidden_dim

        self.dropout = nn.Dropout(dropout)

        # ZINB输出
        self.fc_pi = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        self.fc_mu = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Softplus()
        )
        self.fc_theta = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Softplus()
        )

    def chebyshev_conv(self, x, adj, K: int = 3):
        """
        简化的Chebyshev图卷积
        x: (B, N, F)
        adj: (N, N) - 归一化邻接矩阵
        """
        B, N, F = x.shape

        # 多项式近似: K阶Chebyshev
        support = [x]  # T_0(x) = x
        if K >= 2:
            # T_1(x) = L @ x，其中 L 是归一化拉普拉斯
            x1 = torch.matmul(adj.unsqueeze(0), x)  # (B, N, F)
            support.append(x1)

        for k in range(2, K):
            # T_k(x) = 2 * L @ T_{k-1}(x) - T_{k-2}(x)
            xk = 2 * torch.matmul(adj.unsqueeze(0), support[-1]) - support[-2]
            support.append(xk)

        # 简化为1阶近似 (相当于GCN)
        output = torch.matmul(adj.unsqueeze(0), support[0])
        return output

    def forward(self, X, adj, return_hidden: bool = False):
        """
        X: (B, T, N, F)
        adj: (N, N) - 空间邻接矩阵
        """
        B, T, N, F = X.shape

        # 调整为 (B, F, T, N) 用于2D卷积
        h = X.permute(0, 3, 1, 2)  # (B, F, T, N)

        # 时空卷积块
        for temp_conv, spat_conv, norm in zip(
            self.temporal_convs, self.spatial_convs, self.layer_norms
        ):
            # 时间卷积
            h_temp = temp_conv(h)  # (B, hidden_dim, T, N)
            h_temp = F.relu(h_temp)
            h_temp = self.dropout(h_temp)

            # 转换为 (B, T, N, hidden_dim) 进行空间卷积
            h_spat = h_temp.permute(0, 2, 3, 1)  # (B, T, N, hidden_dim)

            # 图卷积 (对每个时间步)
            h_out = []
            for t in range(T):
                h_t = self.chebyshev_conv(h_spat[:, t], adj)  # (B, N, hidden_dim)
                h_t = spat_conv(h_t)
                h_out.append(h_t)

            h_spat = torch.stack(h_out, dim=1)  # (B, T, N, hidden_dim)
            h_spat = norm(h_spat)
            h_spat = F.relu(h_spat)

            # 残差连接
            if h.shape[1] == self.hidden_dim:
                h_spat = h_spat + h.permute(0, 2, 3, 1)

            h = h_spat.permute(0, 3, 1, 2)  # 转回 (B, hidden_dim, T, N)

        # 取最后时刻
        h_final = h[:, :, -1, :].permute(0, 2, 1)  # (B, N, hidden_dim)

        # ZINB输出
        pi = self.fc_pi(h_final).squeeze(-1)
        mu = self.fc_mu(h_final).squeeze(-1)
        theta = self.fc_theta(h_final).squeeze(-1)

        if return_hidden:
            return pi, mu, theta, h_final
        return pi, mu, theta


class DCRNN(nn.Module):
    """
    DCRNN Baseline (Diffusion Convolutional Recurrent Neural Network)
    使用扩散卷积替代标准图卷积
    """

    def __init__(self, input_dim: int, hidden_dim: int = 64,
                 num_layers: int = 2, max_diffusion_step: int = 2,
                 dropout: float = 0.1):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.max_diffusion_step = max_diffusion_step

        # GRU单元
        self.gru_cells = nn.ModuleList()
        for i in range(num_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            self.gru_cells.append(
                DiffusionGRUCell(in_dim, hidden_dim, max_diffusion_step)
            )

        self.dropout = nn.Dropout(dropout)

        # 输出层
        self.fc_pi = nn.Linear(hidden_dim, 1)
        self.fc_mu = nn.Linear(hidden_dim, 1)
        self.fc_theta = nn.Linear(hidden_dim, 1)

    def forward(self, X, adj, return_hidden: bool = False):
        """
        X: (B, T, N, F)
        adj: (N, N) - 支持有向图的转移矩阵
        """
        B, T, N, F = X.shape

        # 初始化隐藏状态
        h = [torch.zeros(B, N, self.hidden_dim, device=X.device)
             for _ in range(self.num_layers)]

        # 时间步迭代
        for t in range(T):
            x_t = X[:, t]  # (B, N, F)

            for layer_idx, gru_cell in enumerate(self.gru_cells):
                h[layer_idx] = gru_cell(x_t, h[layer_idx], adj)
                x_t = h[layer_idx]
                if layer_idx < self.num_layers - 1:
                    x_t = self.dropout(x_t)

        # 最终输出
        h_final = h[-1]

        pi = torch.sigmoid(self.fc_pi(h_final)).squeeze(-1)
        mu = F.softplus(self.fc_mu(h_final)).squeeze(-1)
        theta = F.softplus(self.fc_theta(h_final)).squeeze(-1)

        if return_hidden:
            return pi, mu, theta, h_final
        return pi, mu, theta


class DiffusionGRUCell(nn.Module):
    """扩散GRU单元"""

    def __init__(self, input_dim: int, hidden_dim: int, max_diffusion_step: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_diffusion_step = max_diffusion_step

        # 扩散卷积参数
        num_matrices = 2 * max_diffusion_step + 1  # 双向扩散

        self.gate_conv = nn.Linear(input_dim + hidden_dim * num_matrices, 2 * hidden_dim)
        self.candidate_conv = nn.Linear(input_dim + hidden_dim * num_matrices, hidden_dim)

    def forward(self, x, h_prev, adj):
        """
        x: (B, N, F)
        h_prev: (B, N, H)
        adj: (N, N) - 邻接矩阵
        """
        # 扩散卷积
        x_diffused = self._diffusion_conv(x, adj)
        h_diffused = self._diffusion_conv(h_prev, adj)

        # 拼接
        combined = torch.cat([x_diffused, h_diffused], dim=-1)

        # GRU门控
        gates = torch.sigmoid(self.gate_conv(combined))
        r, u = torch.split(gates, self.hidden_dim, dim=-1)

        # 候选状态
        combined_c = torch.cat([x_diffused, r * h_diffused], dim=-1)
        c = torch.tanh(self.candidate_conv(combined_c))

        # 更新
        h_new = u * h_prev + (1 - u) * c

        return h_new

    def _diffusion_conv(self, x, adj):
        """
        扩散卷积: 多阶邻域聚合
        """
        B, N, F = x.shape

        # 简化为直接邻居聚合
        # 实际DCRNN需要构建扩散转移矩阵
        x_diffused = [x]

        for k in range(1, self.max_diffusion_step + 1):
            x_k = torch.matmul(adj.unsqueeze(0), x_diffused[-1])
            x_diffused.append(x_k)

        return torch.cat(x_diffused, dim=-1)


class GraphWaveNet(nn.Module):
    """
    Graph WaveNet Baseline
    自适应图卷积 + 空洞因果卷积
    """

    def __init__(self, input_dim: int, hidden_dim: int = 64,
                 num_nodes: int = None, dropout: float = 0.1,
                 num_layers: int = 4):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_nodes = num_nodes

        # 自适应节点嵌入
        if num_nodes is not None:
            self.node_embedding = nn.Parameter(torch.randn(num_nodes, 10) * 0.01)
        else:
            self.node_embedding = None

        # 图卷积层
        self.gconv_layers = nn.ModuleList()
        self.temporal_convs = nn.ModuleList()

        in_dim = input_dim
        for i in range(num_layers):
            # 自适应图卷积
            self.gconv_layers.append(
                AdaptiveGraphConv(in_dim if i == 0 else hidden_dim, hidden_dim)
            )
            # 空洞因果卷积
            dilation = 2 ** i
            self.temporal_convs.append(
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=2,
                         dilation=dilation, padding=dilation)
            )

        self.dropout = nn.Dropout(dropout)

        # 输出层
        self.fc_pi = nn.Linear(hidden_dim, 1)
        self.fc_mu = nn.Linear(hidden_dim, 1)
        self.fc_theta = nn.Linear(hidden_dim, 1)

    def forward(self, X, adj_prior=None, return_hidden: bool = False):
        """
        X: (B, T, N, F)
        adj_prior: 可选的先验邻接矩阵
        """
        B, T, N, F = X.shape

        # 生成自适应邻接矩阵
        if self.node_embedding is not None:
            adj_adaptive = F.softmax(
                F.relu(torch.matmul(self.node_embedding, self.node_embedding.T)),
                dim=-1
            )
            if adj_prior is not None:
                adj = 0.5 * adj_prior + 0.5 * adj_adaptive
            else:
                adj = adj_adaptive
        else:
            adj = adj_prior if adj_prior is not None else torch.eye(N, device=X.device)

        # 调整维度: (B, N, F, T)
        h = X.permute(0, 2, 3, 1)  # (B, N, F, T)

        # 时空层
        for gconv, tconv in zip(self.gconv_layers, self.temporal_convs):
            # 图卷积
            h = gconv(h, adj)  # (B, N, hidden_dim, T)
            h = F.relu(h)

            # 时间卷积
            B, N, C, T_t = h.shape
            h_temp = h.reshape(B * N, C, T_t)
            h_temp = tconv(h_temp)
            h = h_temp.reshape(B, N, -1, h_temp.size(-1))

            h = self.dropout(h)

        # 取最后时刻
        h_final = h[:, :, :, -1]  # (B, N, hidden_dim)

        # ZINB输出
        pi = torch.sigmoid(self.fc_pi(h_final)).squeeze(-1)
        mu = F.softplus(self.fc_mu(h_final)).squeeze(-1)
        theta = F.softplus(self.fc_theta(h_final)).squeeze(-1)

        if return_hidden:
            return pi, mu, theta, h_final
        return pi, mu, theta


class AdaptiveGraphConv(nn.Module):
    """自适应图卷积层"""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_dim, out_dim) * 0.01)
        self.bias = nn.Parameter(torch.zeros(out_dim))

    def forward(self, x, adj):
        """
        x: (B, N, F, T)
        adj: (N, N)
        """
        B, N, F, T = x.shape

        # 对每个时间步进行图卷积
        outputs = []
        for t in range(T):
            x_t = x[:, :, :, t]  # (B, N, F)
            # 图卷积: adj @ x @ weight
            out = torch.matmul(adj.unsqueeze(0), x_t)  # (B, N, F)
            out = torch.matmul(out, self.weight) + self.bias  # (B, N, out_dim)
            outputs.append(out)

        return torch.stack(outputs, dim=-1)  # (B, N, out_dim, T)


# ================================
# 3. 消融变体模型
# ================================

class STTransformerNoSemantic(nn.Module):
    """消融：移除语义融合的主模型"""

    def __init__(self, base_model_class, **kwargs):
        super().__init__()
        kwargs['use_semantic_gate'] = False
        kwargs['semantic_dim'] = 0
        self.model = base_model_class(**kwargs)

    def forward(self, *args, **kwargs):
        kwargs['semantic_embed'] = None
        return self.model(*args, **kwargs)


class STTransformerNoNearRepeat(nn.Module):
    """消融：移除近重复效应的主模型"""

    def __init__(self, base_model_class, **kwargs):
        super().__init__()
        kwargs['use_near_repeat'] = False
        self.model = base_model_class(**kwargs)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)


class STTransformerConcatSemantic(nn.Module):
    """消融：将语义门控改为直接拼接"""

    def __init__(self, base_model_class, **kwargs):
        super().__init__()
        # 强制禁用门控，使用拼接
        kwargs['use_semantic_gate'] = False
        self.model = base_model_class(**kwargs)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)


# ================================
# 4. 评估工具
# ================================

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                      k_percent: float = 0.1) -> dict:
    """
    计算所有评价指标

    Args:
        y_true: (T, N) 真实值
        y_pred: (T, N) 预测值 (期望值)
        k_percent: Top-K百分比

    Returns:
        metrics: 字典包含所有指标
    """
    from sklearn.metrics import mean_squared_error, mean_absolute_error

    T, N = y_true.shape
    k = int(N * k_percent)

    # 基础指标
    rmse = np.sqrt(mean_squared_error(y_true.flatten(), y_pred.flatten()))
    mae = mean_absolute_error(y_true.flatten(), y_pred.flatten())
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100

    # Hit Rate @ Top-K
    hits = 0
    for t in range(T):
        top_k_pred = np.argsort(y_pred[t])[-k:]
        true_crime_indices = np.where(y_true[t] > 0)[0]
        if len(true_crime_indices) > 0:
            hits += len(np.intersect1d(top_k_pred, true_crime_indices))

    hit_rate = hits / (T * k)

    # PAI (Prediction Accuracy Index)
    area_ratio = k / N
    pai_sum = 0
    for t in range(T):
        top_k_pred = np.argsort(y_pred[t])[-k:]
        total_crimes = y_true[t].sum()
        if total_crimes > 0:
            captured = y_true[t][top_k_pred].sum()
            pai_sum += (captured / total_crimes) / area_ratio

    pai = pai_sum / T

    # Jaccard Index
    jaccard_sum = 0
    for t in range(T):
        top_k_pred = set(np.argsort(y_pred[t])[-k:])
        top_k_true = set(np.argsort(y_true[t])[-k:])
        intersection = len(top_k_pred & top_k_true)
        union = len(top_k_pred | top_k_true)
        jaccard_sum += intersection / (union + 1e-8)

    jaccard = jaccard_sum / T

    return {
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape,
        f'HR@{int(k_percent*100)}%': hit_rate,
        'PAI': pai,
        'Jaccard': jaccard
    }


def run_baseline_comparison(X_train, Y_train, X_val, Y_val, X_test, Y_test,
                            adj_spatial, device='cuda'):
    """
    运行所有baseline并对比结果

    示例用法：
    ```python
    results = run_baseline_comparison(
        X_train, Y_train, X_val, Y_val, X_test, Y_test,
        adj_spatial=A_spatial
    )
    print(results['summary_table'])
    ```
    """
    results = {}

    # 1. Historical Average
    print("Running Historical Average...")
    ha = HistoricalAverage(window_size=7)
    ha.fit(X_train, Y_train)
    pi, mu, theta = ha.predict(X_test)
    results['HA'] = calculate_metrics(Y_test, mu)

    # 2. Random Forest
    print("Running Random Forest...")
    rf = RandomForestPredictor(n_estimators=200)
    rf.fit(X_train, Y_train)
    pi, mu, theta = rf.predict(X_test)
    results['RF'] = calculate_metrics(Y_test, mu)

    # 3. ConvLSTM
    print("Running ConvLSTM...")
    # 需要实现训练循环

    # 4. ST-GCN
    print("Running ST-GCN...")
    # 需要实现训练循环

    # ... 其他模型

    return results


if __name__ == "__main__":
    print("Baseline Models for Crime Prediction")
    print("=" * 50)
    print("Available models:")
    print("- HistoricalAverage: Statistical baseline")
    print("- RandomForestPredictor: ML baseline")
    print("- ConvLSTM: Deep learning (no graph)")
    print("- STGCN: Spatio-temporal graph CNN")
    print("- DCRNN: Diffusion convolutional RNN")
    print("- GraphWaveNet: Adaptive graph waveNet")
    print("=" * 50)
