"""
带语义相似度约束的双犯罪类型预测训练 (v2)
改进点:
1. KG模块可训练，支持预训练+微调两阶段
2. 环境邻接矩阵作为额外输入
3. 更灵活的架构设计
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from pathlib import Path
import sys
import yaml
from types import SimpleNamespace
from torch.utils.data import DataLoader, Dataset
import os
import json

sys.path.insert(0, str(Path(__file__).parent))

# 加载配置
def load_config(config_path="src/config/var_conf.yml"):
    with open(config_path, "r", encoding='utf-8') as f:
        cfg_dict = yaml.safe_load(f)
    def dict2namespace(d):
        ns = SimpleNamespace()
        for k, v in d.items():
            if isinstance(v, dict):
                setattr(ns, k, dict2namespace(v))
            else:
                setattr(ns, k, v)
        return ns
    return dict2namespace(cfg_dict)

cfg = load_config()

# 导入模型和损失
from train_no_cpted_no_sem import (
    SpatioTemporalTransformer, CrimeDataset,
    GatingFusion, TemporalTransformerBlock
)
from semantic_loss import SemanticSimilarityLoss
from build_heterogeneous_kg import HeterogeneousKG, build_predefined_knowledge_graph


class EnvironmentAugmentedModel(nn.Module):
    """
    环境增强的双犯罪类型预测模型

    特点:
    1. 融合空间邻接矩阵(A_spatial)和环境邻接矩阵(A_env)
    2. 可学习的KG模块
    3. 双犯罪类型输出
    """
    def __init__(self, base_model, kg_model, hidden_dim=64, use_env_adj=True, env_adj_weight=0.3):
        super().__init__()
        self.base_model = base_model
        self.kg_model = kg_model
        self.hidden_dim = hidden_dim
        self.use_env_adj = use_env_adj
        self.env_adj_weight = env_adj_weight

        # 环境邻接矩阵融合权重 (可学习)
        if use_env_adj:
            self.env_fusion_weight = nn.Parameter(torch.tensor(env_adj_weight))

        # 暴力犯罪投影层 (主任务)
        self.violent_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        # 财产犯罪投影层 (仅用于语义约束，不预测)
        self.property_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        # 输出头 (仅主任务-暴力犯罪)
        self.violent_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 3)
        )

    def fuse_adjacency(self, A_spatial, A_env):
        """
        融合空间邻接矩阵和环境邻接矩阵

        Args:
            A_spatial: (N, N) 空间邻接
            A_env: (N, N) 环境相似度邻接
        Returns:
            A_fused: (N, N) 融合后的邻接矩阵
        """
        if not self.use_env_adj or A_env is None:
            return A_spatial

        # 归一化
        A_spatial_norm = F.normalize(A_spatial, p=1, dim=1)
        A_env_norm = F.normalize(A_env, p=1, dim=1)

        # 可学习的融合
        w = torch.sigmoid(self.env_fusion_weight)
        A_fused = (1 - w) * A_spatial_norm + w * A_env_norm

        return A_fused

    def forward(self, X, A_spatial, A_distance, A_crime, A_hypergraph,
                A_env=None, OD=None, kg_embed=None, crime_history=None,
                return_kg_similarity=False):
        """
        前向传播

        Args:
            X: 输入特征
            A_spatial: 空间邻接
            A_env: 环境邻接 (可选)
            return_kg_similarity: 是否返回KG计算的相似度
        """
        B, T, N, F = X.shape
        device = X.device
        
        # 调试输出
        print(f"[DEBUG forward] X.shape: {X.shape}")
        print(f"[DEBUG forward] base_model.static_dim: {self.base_model.static_dim}")
        print(f"[DEBUG forward] F - static_dim: {F - self.base_model.static_dim}")

        # 融合邻接矩阵
        if self.use_env_adj and A_env is not None:
            A_spatial_fused = self.fuse_adjacency(A_spatial, A_env)
        else:
            A_spatial_fused = A_spatial

        # 基础模型前向
        static_idx = self.base_model.static_dim
        X_static = X[:, -1, :, :static_idx]
        X_dynamic = X[:, :, :, static_idx:]

        # 静态支路
        H_static_fused, _ = self.base_model.gating_fusion(X_static, kg_embed, None)
        H_static = self.base_model.static_encoder(H_static_fused)
        H_static = self.base_model.spatial_layers[0](H_static, A=A_spatial_fused)
        H_static = self.base_model.hypergraph_attn(H_static, A_hypergraph)

        # 动态支路
        H_dynamic = self.base_model.dynamic_encoder(X_dynamic)
        H_dynamic = self.base_model.time_pos(H_dynamic)

        for layer in self.base_model.temporal_layers:
            H_dynamic = layer(H_dynamic)

        H_dynamic_spatial = []
        for t in range(H_dynamic.size(1)):
            H_t = self.base_model.spatial_layers[1](H_dynamic[:, t, :, :])
            H_dynamic_spatial.append(H_t)
        H_dynamic = torch.stack(H_dynamic_spatial, dim=1)
        H_dynamic_agg = H_dynamic[:, -1, :, :]

        # 融合特征
        H_fused = H_static + H_dynamic_agg

        # 双类型特征投影 (用于语义约束)
        h_v = self.violent_proj(H_fused)
        h_p = self.property_proj(H_fused)  # 仅用于语义约束

        # 预测ZINB参数 (仅主任务-暴力犯罪)
        violent_params = self.violent_head(h_v)

        pi_v = torch.sigmoid(violent_params[:, :, 0:1])
        mu_v = torch.exp(violent_params[:, :, 1:2])
        theta_v = torch.exp(violent_params[:, :, 2:3])

        # 只返回暴力犯罪预测 + 两个隐藏特征(用于语义约束)
        outputs = ((pi_v, mu_v, theta_v), h_v, h_p)

        if return_kg_similarity:
            # 计算KG相似度矩阵
            with torch.set_grad_enabled(self.training):
                kg_similarity = self.kg_model.compute_similarity_matrix()
            return outputs, kg_similarity

        return outputs


def pretrain_kg(kg_model, num_epochs=30, lr=0.01):
    """
    预训练KG模块
    目标: 让KG的嵌入和关联矩阵稳定
    """
    print("\n" + "=" * 70)
    print(f"阶段1: 预训练KG模块 ({num_epochs} epochs)")
    print("=" * 70)

    optimizer = torch.optim.Adam(kg_model.parameters(), lr=lr)

    # 获取先验矩阵作为目标
    prior_matrix, _ = build_predefined_knowledge_graph()
    prior_matrix = prior_matrix.to(kg_model.prior_matrix.device)

    for epoch in range(num_epochs):
        kg_model.train()
        optimizer.zero_grad()

        # 前向传播
        crime_embed, poi_embed, final_adj = kg_model.forward()

        # 损失1: 关联矩阵与先验的接近度
        adj_loss = F.mse_loss(final_adj, prior_matrix)

        # 损失2: 嵌入的平滑性 (防止过拟合)
        embed_smoothness = torch.norm(crime_embed[:-1] - crime_embed[1:])

        # 总损失
        loss = adj_loss + 0.1 * embed_smoothness

        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0 or epoch == 0:
            similarity = kg_model.compute_similarity_matrix()
            print(f"  Epoch {epoch+1}/{num_epochs}: Loss={loss.item():.4f}, "
                  f"S_vp={similarity[0,1].item():.4f}")

    print("[OK] KG预训练完成")
    return kg_model


def train_main_model(model, kg_model, train_loader, criterion, optimizer,
                     scheduler, A_spatial_t, A_distance_t, A_hyper_t, A_env_t,
                     kg_embed_t, A_crime_t, num_epochs, device,
                     finetune_kg=False, kg_lr=0.0001):
    """
    主模型训练

    Args:
        finetune_kg: 是否微调KG
        kg_lr: KG微调学习率
    """
    print("\n" + "=" * 70)
    print(f"阶段2: 主模型训练 ({num_epochs} epochs)")
    if finetune_kg:
        print(f"  (KG微调启用, lr={kg_lr})")
    print("=" * 70)

    best_loss = float('inf')

    # 如果需要微调KG，创建单独的优化器
    if finetune_kg:
        kg_optimizer = torch.optim.Adam(kg_model.parameters(), lr=kg_lr)

    for epoch in range(num_epochs):
        model.train()
        kg_model.train() if finetune_kg else kg_model.eval()

        epoch_total = 0
        epoch_zinb_v = 0
        epoch_zinb_p = 0
        epoch_semantic = 0
        epoch_hit = 0
        epoch_jaccard = 0
        epoch_kg_sim = 0
        epoch_threshold = 0

        for batch_idx, (X_batch, _, OD_batch, Y_batch) in enumerate(train_loader):
            X_batch = X_batch.to(device)
            OD_batch = OD_batch.to(device)
            Y_batch = Y_batch.to(device)

            Y_v_batch = Y_batch[:, :, 0:1]
            Y_p_batch = Y_batch[:, :, 1:2]

            batch_size = X_batch.size(0)
            A_crime_batch = A_crime_t.unsqueeze(0).expand(batch_size, -1, -1)

            optimizer.zero_grad()
            if finetune_kg:
                kg_optimizer.zero_grad()

            # 前向传播
            outputs, kg_similarity = model(
                X_batch, A_spatial_t, A_distance_t, A_crime_batch,
                A_hyper_t, A_env_t, OD_batch, kg_embed_t, None,
                return_kg_similarity=True
            )

            outputs_v, h_v, h_p = outputs  # 只预测暴力犯罪，h_p用于语义约束

            # 使用KG计算的动态相似度
            S_vp_dynamic = kg_similarity[0, 1].detach()
            criterion.S_vp = S_vp_dynamic

            # 计算损失 (单任务+语义约束)
            # 注意：财产犯罪标签Y_p_batch只用于语义约束损失，不用于预测
            total_loss, loss_dict = criterion(
                outputs_v,
                Y_v_batch, Y_p_batch,
                h_v, h_p
            )

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            if finetune_kg:
                kg_optimizer.step()

            epoch_total += loss_dict['total']
            epoch_zinb_v += loss_dict['zinb_violent']
            epoch_zinb_p += loss_dict['zinb_property']
            epoch_semantic += loss_dict['semantic']
            epoch_hit += loss_dict['hit_rate']
            epoch_jaccard += loss_dict['jaccard']
            epoch_kg_sim += S_vp_dynamic.item()
            epoch_threshold += loss_dict['threshold']

        n_batches = len(train_loader)
        avg_total = epoch_total / n_batches
        avg_zinb_v = epoch_zinb_v / n_batches
        avg_zinb_p = epoch_zinb_p / n_batches
        avg_semantic = epoch_semantic / n_batches
        avg_hit = epoch_hit / n_batches
        avg_jaccard = epoch_jaccard / n_batches
        avg_kg_sim = epoch_kg_sim / n_batches
        avg_threshold = epoch_threshold / n_batches

        scheduler.step(avg_total)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1}/{num_epochs}: "
                  f"Total={avg_total:.4f} (ZINB_V={avg_zinb_v:.4f}, "
                  f"ZINB_P={avg_zinb_p:.4f}, Sem={avg_semantic:.4f}, "
                  f"Hit={avg_hit:.4f}, Jac={avg_jaccard:.4f}, "
                  f"Thresh={avg_threshold:.4f}, S_vp={avg_kg_sim:.4f})")

        if avg_total < best_loss:
            best_loss = avg_total
            save_path = f'checkpoints/model_semantic_v2_best.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'kg_model_state_dict': kg_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
            }, save_path)

    print(f"\n[OK] 训练完成！最佳模型: {save_path}")
    return save_path


def load_data_with_env():
    """加载数据，包括环境邻接矩阵"""
    print("\n[1] 加载数据...")

    X = np.load("data/processed/X_with_econ.npy")
    Y = np.load("data/processed/Y.npy")
    OD = np.load("data/processed/dynamic_od_flow.npy")
    OD = np.log1p(OD)

    A_spatial = np.load("data/processed/adj_adaptive.npy")
    A_distance = np.load("data/processed/adj_distance.npy")
    # 加载两种犯罪的邻接矩阵
    A_crime_violent = np.load("data/processed/adj_crime_violent.npy")

    # 默认使用暴力犯罪矩阵（兼容单任务训练）
    A_crime = A_crime_violent

    A_hypergraph = np.load("data/processed/adj_hypergraph.npy")

    # 加载环境邻接矩阵
    env_adj_path = "data/processed/grid_environment_similarity.npy"
    if os.path.exists(env_adj_path):
        A_env = np.load(env_adj_path)
        print(f"  [OK] 加载环境邻接矩阵: {A_env.shape}")
    else:
        print(f"  [WARN] 环境邻接矩阵不存在: {env_adj_path}")
        A_env = None

    kg_embed_path = "data/processed/kg_embeddings.npy"
    if os.path.exists(kg_embed_path):
        kg_embed = np.load(kg_embed_path)
    else:
        kg_embed = np.zeros((Y.shape[1], 32), dtype=np.float32)

    print(f"  X shape: {X.shape}")
    print(f"  Y shape: {Y.shape}")


    return X, Y, OD, A_spatial, A_distance, A_crime, A_hypergraph, A_env, kg_embed


def create_windows(X, Y, OD, window=14, crime_lag=7):
    """创建时间窗口"""
    print("\n[2] 创建时间窗口...")

    X_window, Y_window, OD_window = [], [], []
    for i in range(len(X) - window):
        X_window.append(X[i:i+window])
        Y_window.append(Y[i+window])
        OD_window.append(OD[i+window])

    X_window = np.array(X_window)
    Y_window = np.array(Y_window)
    OD_window = np.array(OD_window)

    offset = window - crime_lag
    X_input = X_window[:, :offset, :, :]

    print(f"  X_input: {X_input.shape}, Y: {Y_window.shape}")

    return X_input, Y_window, OD_window


def train_full_pipeline(
    lambda_semantic=0.1,
    lambda_hit=0.5,
    lambda_jaccard=0.3,
    num_epochs_kg=30,
    num_epochs_main=100,
    lr=0.001,
    lr_kg=0.01,
    lr_kg_finetune=0.0001,
    use_env_adj=True,
    finetune_kg=True
):
    """
    完整训练流程

    Args:
        lambda_semantic: 语义损失权重
        num_epochs_kg: KG预训练轮数
        num_epochs_main: 主模型训练轮数
        lr: 主模型学习率
        lr_kg: KG预训练学习率
        lr_kg_finetune: KG微调学习率
        use_env_adj: 是否使用环境邻接矩阵
        finetune_kg: 是否微调KG
    """
    print("=" * 70)
    print("语义相似度约束训练 v2 - 两阶段训练")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n设备: {device}")

    # 加载数据
    X, Y, OD, A_spatial, A_distance, A_crime, A_hypergraph, A_env, kg_embed = load_data_with_env()

    # 创建窗口
    # 创建时间窗口（X, Y, OD）
    X_win, Y_win, OD_win = create_windows(X, Y, OD)
    

    test_size = 12
    train_size = len(X_win) - test_size

    X_train, X_test = X_win[:train_size], X_win[train_size:]
    Y_train, Y_test = Y_win[:train_size], Y_win[train_size:]
    OD_train, OD_test = OD_win[:train_size], OD_win[train_size:]

    print(f"\n[3] 数据集: 训练{len(X_train)}, 测试{len(X_test)}")

    # 数据加载器
    train_dataset = CrimeDataset(X_train, Y_train, A_crime, OD_train)  # 单任务：不使用A_property
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

    # 模型参数
    # 从配置读取维度，确保与数据一致
    static_dim = cfg.dataset.static_dim  # 38 (环境24 + 经济7 + POI 7)
    dynamic_dim = cfg.dataset.dynamic_dim  # 33
    
    # 调试输出
    print(f"[DEBUG] cfg.dataset.static_dim: {cfg.dataset.static_dim}")
    print(f"[DEBUG] cfg.dataset.dynamic_dim: {cfg.dataset.dynamic_dim}")
    print(f"[DEBUG] static_dim variable: {static_dim}")
    print(f"[DEBUG] dynamic_dim variable: {dynamic_dim}")
    num_nodes = Y_train.shape[1]
    kg_dim = kg_embed.shape[-1] if kg_embed is not None else 32

    # 初始化KG模型
    prior_matrix, poi_names = build_predefined_knowledge_graph()
    kg_model = HeterogeneousKG(
        crime_types=2,
        poi_types=len(poi_names),
        hidden_dim=32,
        prior_matrix=prior_matrix.numpy()
    ).to(device)

    # 尝试加载已保存的KG状态
    kg_state_path = "data/processed/kg_model_state.pt"
    if os.path.exists(kg_state_path):
        kg_model.load_state_dict(torch.load(kg_state_path, map_location=device, weights_only=True))
        print(f"\n[OK] 加载预训练KG: {kg_state_path}")

    # ========== 阶段1: KG预训练 ==========
    if num_epochs_kg > 0:
        kg_model = pretrain_kg(kg_model, num_epochs=num_epochs_kg, lr=lr_kg)
        # 保存预训练后的KG
        torch.save(kg_model.state_dict(), "data/processed/kg_model_pretrained.pt")

    # 获取KG计算的初始相似度
    kg_model.eval()
    with torch.no_grad():
        initial_similarity = kg_model.compute_similarity_matrix()
    print(f"\n[KG] 初始Violent-Property相似度: {initial_similarity[0,1].item():.4f}")

    # ========== 阶段2: 主模型训练 ==========
    # 初始化基础模型
    base_model = SpatioTemporalTransformer(
        static_dim=static_dim,
        dynamic_dim=dynamic_dim,
        kg_dim=kg_dim,
        hidden_dim=64,
        num_heads=4,
        num_temporal_layers=3,
        num_spatial_layers=2,
        dropout=0.1,
        num_nodes=num_nodes,
        use_multitype_nr=True,
        distance_matrix=A_distance,
        predict_property=False,
        predict_public=False
    ).to(device)

    # 包装为环境增强模型
    model = EnvironmentAugmentedModel(
        base_model, kg_model, hidden_dim=64,
        use_env_adj=use_env_adj
    ).to(device)
    
    # 调试输出
    print(f"[DEBUG] base_model.static_dim: {base_model.static_dim}")
    print(f"[DEBUG] base_model.dynamic_dim: {base_model.dynamic_dim}")
    print(f"[DEBUG] X_train shape: {X_train.shape}")

    # 语义损失 (v2: 增加Hit Rate和Jaccard优化)
    criterion = SemanticSimilarityLoss(
        similarity_matrix=initial_similarity.cpu().numpy(),
        lambda_semantic=lambda_semantic,
        lambda_hit=lambda_hit,
        lambda_jaccard=lambda_jaccard,
        top_k_percent=0.1
    ).to(device)

    # 优化器 (只优化主模型参数，不优化KG)
    main_params = list(model.base_model.parameters()) + \
                  list(model.violent_proj.parameters()) + \
                  list(model.property_proj.parameters()) + \
                  list(model.violent_head.parameters())

    if use_env_adj:
        main_params += [model.env_fusion_weight]

    optimizer = torch.optim.Adam(main_params, lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=10, factor=0.5
    )

    # 准备图数据
    A_spatial_t = torch.tensor(A_spatial, dtype=torch.float32).to(device)
    A_distance_t = torch.tensor(A_distance, dtype=torch.float32).to(device)
    A_hyper_t = torch.tensor(A_hypergraph, dtype=torch.float32).to(device)
    kg_embed_t = torch.tensor(kg_embed, dtype=torch.float32).to(device)
    A_crime_t = torch.tensor(A_crime[-1] if A_crime.ndim == 3 else A_crime,
                             dtype=torch.float32).to(device)

    # 环境邻接矩阵
    if use_env_adj and A_env is not None:
        A_env_t = torch.tensor(A_env, dtype=torch.float32).to(device)
        print(f"[OK] 环境邻接矩阵已加载")
    else:
        A_env_t = None
        print(f"[INFO] 不使用环境邻接矩阵")

    # 训练主模型
    best_model_path = train_main_model(
        model, kg_model, train_loader, criterion, optimizer, scheduler,
        A_spatial_t, A_distance_t, A_hyper_t, A_env_t,
        kg_embed_t, A_crime_t, num_epochs_main, device,
        finetune_kg=finetune_kg, kg_lr=lr_kg_finetune
    )

    # 保存最终配置
    config = {
        'lambda_semantic': lambda_semantic,
        'num_epochs_kg': num_epochs_kg,
        'num_epochs_main': num_epochs_main,
        'use_env_adj': use_env_adj,
        'finetune_kg': finetune_kg,
        'final_kg_similarity': kg_model.compute_similarity_matrix().tolist()
    }
    with open('checkpoints/training_config_v2.json', 'w') as f:
        json.dump(config, f, indent=2)

    print("\n" + "=" * 70)
    print("训练完成!")
    print(f"最佳模型: {best_model_path}")
    print(f"配置: checkpoints/training_config_v2.json")
    print("=" * 70)

    return best_model_path


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='语义约束训练v2')
    parser.add_argument('--lambda_semantic', type=float, default=0.1)
    parser.add_argument('--lambda_hit', type=float, default=0.5, help='Hit Rate损失权重')
    parser.add_argument('--lambda_jaccard', type=float, default=0.3, help='Jaccard损失权重')
    parser.add_argument('--epochs_kg', type=int, default=30, help='KG预训练轮数')
    parser.add_argument('--epochs_main', type=int, default=100, help='主模型训练轮数')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lr_kg', type=float, default=0.01, help='KG预训练学习率')
    parser.add_argument('--lr_kg_ft', type=float, default=0.0001, help='KG微调学习率')
    parser.add_argument('--no_env_adj', action='store_true', help='不使用环境邻接矩阵')
    parser.add_argument('--no_kg_finetune', action='store_true', help='不微调KG')

    args = parser.parse_args()

    train_full_pipeline(
        lambda_semantic=args.lambda_semantic,
        lambda_hit=args.lambda_hit,
        lambda_jaccard=args.lambda_jaccard,
        num_epochs_kg=args.epochs_kg,
        num_epochs_main=args.epochs_main,
        lr=args.lr,
        lr_kg=args.lr_kg,
        lr_kg_finetune=args.lr_kg_ft,
        use_env_adj=not args.no_env_adj,
        finetune_kg=not args.no_kg_finetune
    )
