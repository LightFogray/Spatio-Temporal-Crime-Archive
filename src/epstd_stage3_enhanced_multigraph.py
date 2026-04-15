"""
EP-STD Stage 3 Multi-Graph Training
=====================================
多图模型训练流程（基于 epstd_stage3_enhanced）
"""

import os
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import pickle

from epstd_stage1 import EnvironmentEncoder
from epstd_stage2 import PrototypeLibrary
from epstd_stage3_multigraph import (
    MultiGraphConditionalDiffusion,
    DualTaskMultiGraphDiffusion,
    AdaptiveGraphFusion,
    CrossCrimeGate
)
from epstd_stage3_enhanced import (
    LogicGuidedDiffusionScheduler,
    LogicConstraintCalculator
)


# ==================== 多图模型训练流程 ====================

def train_multigraph_diffusion(
    env_encoder,
    prototype_library,
    X_train,
    Y_train,
    adj_list,
    epochs=100,
    batch_size=16,
    lr=1e-4,
    device='cuda',
    use_logic_guidance=True,
    logic_warmup_epochs=20
):
    """
    训练多图条件扩散模型
    """
    print("="*70)
    print("Training Multi-Graph Diffusion Model")
    print("="*70)
    print(f"Graphs: spatial, distance, crime_violent, crime_property, od")
    print(f"Adaptive Fusion: w = MLP(E_env)")
    print(f"Cross-Crime Gate: Enabled")
    print("="*70)

    # 冻结环境编码器
    env_encoder.eval()
    for param in env_encoder.parameters():
        param.requires_grad = False

    # 创建多图模型
    num_nodes = Y_train.shape[1]
    model = MultiGraphConditionalDiffusion(
        num_nodes=num_nodes,
        hidden_dim=128,
        num_layers=4,
        time_dim=64,
        env_dim=64,
        num_prototypes=prototype_library.n_prototypes
    ).to(device)

    # 加载预训练的基础模型权重
    if os.path.exists('checkpoints/epstd_diffusion_best.pt'):
        print("Loading pre-trained base model weights...")
        try:
            base_state = torch.load('checkpoints/epstd_diffusion_best.pt', map_location=device)
            model_dict = model.state_dict()
            shared_weights = {k: v for k, v in base_state.items() if k in model_dict}
            model_dict.update(shared_weights)
            model.load_state_dict(model_dict, strict=False)
        except Exception as e:
            print(f"  Warning: Could not load pre-trained weights: {e}")

    # 调度器和逻辑计算器
    scheduler = LogicGuidedDiffusionScheduler(num_timesteps=1000, device=device)
    logic_calculator = LogicConstraintCalculator(static_feature_dim=24).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    # 准备数据
    # 提取静态特征部分（前24维）用于环境编码器
    # X.shape = (samples, N, 50)，其中前24维是静态空间特征
    X_static = X_train[:, :, :24]  # 只取静态特征部分
    static_features = torch.tensor(X_static, dtype=torch.float32).to(device)

    Y_tensor = torch.tensor(Y_train, dtype=torch.float32).to(device)
    prototype_labels = np.load('data/processed/prototype_labels.npy')
    prototype_ids = torch.tensor(prototype_labels, dtype=torch.long).to(device)

    # 邻接矩阵
    adj_tensors = [torch.tensor(adj, dtype=torch.float32).to(device) for adj in adj_list]

    num_samples = len(X_train)
    best_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        epoch_losses = []

        # 逻辑引导强度调度
        if epoch < logic_warmup_epochs:
            guidance_scale = 0.0
        else:
            guidance_scale = min(1.0, (epoch - logic_warmup_epochs) / 20.0)

        pbar = tqdm(range(0, num_samples, batch_size), desc=f"Epoch {epoch+1}")
        for i in pbar:
            batch_idx = list(range(i, min(i + batch_size, num_samples)))
            if len(batch_idx) < 2:
                continue

            x_0 = Y_tensor[batch_idx]
            env_emb_batch = env_encoder(static_features[batch_idx])
            proto_ids_batch = prototype_ids.unsqueeze(0).expand(len(batch_idx), -1)

            # 前向扩散
            t = scheduler.sample_timesteps(len(batch_idx))
            noise = torch.randn_like(x_0)
            x_t = scheduler.add_noise(x_0, t, noise)

            # 预测噪声（多图输入）
            noise_pred, pi, graph_weights, crime_gates = model(
                x_t, t, env_emb_batch, proto_ids_batch, adj_tensors
            )

            # 扩散损失
            loss_diffusion = F.mse_loss(noise_pred, noise)

            # 零膨胀损失
            zero_mask = (x_0 == 0).float()
            loss_zi = F.binary_cross_entropy(pi, zero_mask)

            # 总损失
            loss = loss_diffusion + 0.1 * loss_zi

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_losses.append(loss.item())
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_loss = np.mean(epoch_losses)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Guidance: {guidance_scale:.2f}")

            # 打印图权重统计
            if graph_weights:
                gw = torch.stack(graph_weights).mean(dim=(0, 1, 2))
                print(f"  Graph weights: Spatial={gw[0]:.3f}, Distance={gw[1]:.3f}, "
                      f"CrimeV={gw[2]:.3f}, CrimeP={gw[3]:.3f}, OD={gw[4]:.3f}")

            if crime_gates:
                cg = torch.stack(crime_gates).mean(dim=(0, 1, 2))
                print(f"  Crime gate: Violent={cg[0]:.3f}, Property={cg[1]:.3f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), 'checkpoints/multigraph_diffusion_best.pt')

    print("\nTraining completed!")
    return model, scheduler, logic_calculator


# ==================== 双任务多图训练 ====================

def train_dual_multigraph_diffusion(
    env_encoder,
    prototype_library,
    X_train,
    Y_violent,
    Y_property,
    adj_list,
    epochs=100,
    batch_size=16,
    lr=1e-4,
    device='cuda',
    task_weights=None
):
    """
    双任务多图扩散模型训练
    """
    print("="*70)
    print("Training Dual-Task Multi-Graph Diffusion")
    print("="*70)

    if task_weights is None:
        task_weights = {'violent': 1.0, 'property': 0.5}

    # 冻结环境编码器
    env_encoder.eval()
    for param in env_encoder.parameters():
        param.requires_grad = False

    # 创建基础模型
    num_nodes = Y_violent.shape[1]
    base_model = MultiGraphConditionalDiffusion(
        num_nodes=num_nodes,
        hidden_dim=128,
        num_layers=4,
        time_dim=64,
        env_dim=64,
        num_prototypes=prototype_library.n_prototypes
    ).to(device)

    # 包装为双任务模型
    model = DualTaskMultiGraphDiffusion(base_model).to(device)

    scheduler = LogicGuidedDiffusionScheduler(num_timesteps=1000, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    # 准备数据
    static_features = torch.tensor(X_train, dtype=torch.float32).to(device)
    Y_v_tensor = torch.tensor(Y_violent, dtype=torch.float32).to(device)
    Y_p_tensor = torch.tensor(Y_property, dtype=torch.float32).to(device)
    prototype_labels = np.load('data/processed/prototype_labels.npy')
    prototype_ids = torch.tensor(prototype_labels, dtype=torch.long).to(device)
    adj_tensors = [torch.tensor(adj, dtype=torch.float32).to(device) for adj in adj_list]

    num_samples = len(X_train)
    best_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        epoch_losses = []

        for i in tqdm(range(0, num_samples, batch_size), desc=f"Epoch {epoch+1}"):
            batch_idx = list(range(i, min(i + batch_size, num_samples)))
            if len(batch_idx) < 2:
                continue

            x_v_0 = Y_v_tensor[batch_idx]
            x_p_0 = Y_p_tensor[batch_idx]

            env_emb_batch = env_encoder(static_features[batch_idx])
            proto_ids_batch = prototype_ids.unsqueeze(0).expand(len(batch_idx), -1)

            # 使用暴力犯罪数据计算crime_stats（用于门控）
            crime_stats = torch.stack([
                x_v_0.mean(dim=1, keepdim=True).expand(-1, x_v_0.shape[1]),
                x_p_0.mean(dim=1, keepdim=True).expand(-1, x_p_0.shape[1])
            ], dim=-1)  # (B, N, 2)

            # 前向扩散
            t = scheduler.sample_timesteps(len(batch_idx))
            noise_v = torch.randn_like(x_v_0)
            noise_p = torch.randn_like(x_p_0)
            x_v_t = scheduler.add_noise(x_v_0, t, noise_v)
            x_p_t = scheduler.add_noise(x_p_0, t, noise_p)

            # 平均噪声状态用于输入（共享底层）
            x_t = (x_v_t + x_p_t) / 2

            # 预测噪声
            (noise_v_pred, noise_p_pred), pi, graph_weights, crime_gates = model(
                x_t, t, env_emb_batch, proto_ids_batch, adj_tensors, crime_stats
            )

            # 任务特定损失
            loss_v = F.mse_loss(noise_v_pred, noise_v)
            loss_p = F.mse_loss(noise_p_pred, noise_p)

            # 不确定性加权
            precision_v = torch.exp(-2 * model.log_sigma_violent)
            precision_p = torch.exp(-2 * model.log_sigma_property)

            loss = (task_weights['violent'] * precision_v * loss_v + model.log_sigma_violent +
                    task_weights['property'] * precision_p * loss_p + model.log_sigma_property)

            # 零膨胀损失
            zero_mask_v = (x_v_0 == 0).float()
            zero_mask_p = (x_p_0 == 0).float()
            loss_zi = F.binary_cross_entropy(pi, (zero_mask_v + zero_mask_p) / 2)

            loss = loss + 0.1 * loss_zi

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_losses.append(loss.item())

        avg_loss = np.mean(epoch_losses)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
            print(f"  Task uncertainties: Violent={model.log_sigma_violent.item():.3f}, "
                  f"Property={model.log_sigma_property.item():.3f}")

            if graph_weights:
                gw = torch.stack(graph_weights).mean(dim=(0, 1, 2))
                print(f"  Graph weights: S={gw[0]:.2f}, D={gw[1]:.2f}, "
                      f"CV={gw[2]:.2f}, CP={gw[3]:.2f}, OD={gw[4]:.2f}")

            if crime_gates:
                cg = torch.stack(crime_gates).mean(dim=(0, 1, 2))
                print(f"  Crime gate: V={cg[0]:.2f}, P={cg[1]:.2f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), 'checkpoints/dual_multigraph_diffusion_best.pt')

    return model, scheduler


# ==================== 主流程 ====================

def main():
    """多图模型主流程"""
    print("="*70)
    print("L-EPSTD Stage 3: Multi-Graph Adaptive Fusion")
    print("="*70)
    print("Features:")
    print("  1. 5 Graphs: Spatial, Distance, CrimeV, CrimeP, OD")
    print("  2. Adaptive Fusion: w = MLP(E_env)")
    print("  3. Cross-Crime Gate: Auto-switch")
    print("="*70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # 加载数据
    X_full = np.load('data/processed/X.npy')  # (samples, N, 50)
    Y = np.load('data/processed/Y.npy')

    # Y 现在可能是 (samples, N, 2) 包含双犯罪类型，取暴力犯罪 (channel 0)
    if Y.ndim == 3:
        Y = Y[:, :, 0]  # 暴力犯罪

    # 提取静态特征部分（前24维）用于环境编码器
    # 完整的X包含: 静态特征(24) + 天气(8) + 犯罪滞后(14) + OD(4) = 50
    X = X_full[:, :, :24]  # 只取静态特征用于Stage 3的初始输入

    # 加载5张图
    print("\nLoading adjacency matrices...")
    adj_list = [
        np.load('data/processed/adj_adaptive.npy'),
        np.load('data/processed/adj_distance.npy'),
        np.load('data/processed/adj_crime_violent.npy'),
        np.load('data/processed/adj_crime_property.npy'),
        np.load('data/processed/adj_od.npy')
    ]

    # 如果是动态图，取平均
    for i in [2, 3]:  # crime graphs
        if len(adj_list[i].shape) == 3:
            adj_list[i] = adj_list[i].mean(axis=0)

    print(f"Graph shapes: {[a.shape for a in adj_list]}")

    # 加载预训练模型
    env_encoder = EnvironmentEncoder(input_dim=24, output_dim=64).to(device)
    env_encoder.load_state_dict(torch.load('checkpoints/env_encoder_best.pt', map_location=device))
    env_encoder.eval()

    with open('checkpoints/prototype_library.pkl', 'rb') as f:
        proto_data = pickle.load(f)

    prototype_library = PrototypeLibrary(n_prototypes=proto_data['n_prototypes'])
    prototype_library.prototypes = proto_data['prototypes']
    prototype_library.prototype_risks = proto_data['prototype_risks']

    # 划分数据
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    Y_train, Y_test = Y[:split_idx], Y[split_idx:]

    print(f"\nTraining data: {len(X_train)} samples")
    print(f"Test data: {len(X_test)} samples")

    # 训练
    model, scheduler, _ = train_multigraph_diffusion(
        env_encoder, prototype_library,
        X_train, Y_train, adj_list,
        epochs=100, batch_size=16, device=device
    )

    print("\n" + "="*70)
    print("Training Completed!")
    print("="*70)


if __name__ == "__main__":
    main()
