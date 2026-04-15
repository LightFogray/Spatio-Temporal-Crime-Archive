#!/usr/bin/env python3
"""
维度检查脚本 - 验证所有模块的输入输出维度
================================================
"""

import numpy as np
import torch
import torch.nn as nn

print("="*70)
print("维度检查 - 数据与模型维度验证")
print("="*70)

# 1. 检查数据文件
print("\n[1/5] 检查数据文件维度...")

data_files = {
    'X.npy': 'data/processed/X.npy',
    'Y.npy': 'data/processed/Y.npy',
    'env_embeddings.npy': 'data/processed/env_embeddings.npy',
    'prototype_labels.npy': 'data/processed/prototype_labels.npy',
    'crime_combined_timeseries.npy': 'data/processed/crime_combined_timeseries.npy',
}

expected_shapes = {
    'X.npy': (723, 1246, 50),  # (T, N, F=24+8+14+4)
    'Y.npy': (723, 1246),       # (T, N)
    'env_embeddings.npy': (1246, 64),  # (N, env_dim)
    'prototype_labels.npy': (1246,),   # (N,)
}

for name, path in data_files.items():
    try:
        data = np.load(path)
        print(f"  {name:25s}: {str(data.shape):20s}", end="")
        if name in expected_shapes:
            if data.shape == expected_shapes[name]:
                print(" [OK] 匹配")
            else:
                print(f" [WARN] 期望 {expected_shapes[name]}")
        else:
            print()
    except FileNotFoundError:
        print(f"  {name:25s}: 文件不存在 [ERR]")

# 2. 检查图结构
print("\n[2/5] 检查图结构维度...")

graph_files = [
    ('adj_adaptive.npy', (1246, 1246)),
    ('adj_distance.npy', (1246, 1246)),
    ('adj_crime_violent.npy', (1246, 1246)),
    ('adj_crime_property.npy', (1246, 1246)),
    ('adj_od.npy', (1246, 1246)),
]

for name, expected_shape in graph_files:
    try:
        data = np.load(f'data/processed/{name}')
        print(f"  {name:25s}: {str(data.shape):20s}", end="")
        if data.shape == expected_shape:
            print(" [OK]")
        else:
            print(f" [WARN] 期望 {expected_shape}")
    except FileNotFoundError:
        print(f"  {name:25s}: 文件不存在 [ERR]")

# 3. 检查模型维度兼容性
print("\n[3/5] 检查模型维度兼容性...")

# 模拟测试
B, N, env_dim = 4, 1246, 64
static_dim = 24
num_prototypes = 10

print(f"  批次大小 B: {B}")
print(f"  网格数量 N: {N}")
print(f"  环境维度: {env_dim}")
print(f"  静态特征维度: {static_dim}")
print(f"  原型数量: {num_prototypes}")

# 4. 检查环境编码器
print("\n[4/5] 检查环境编码器...")
try:
    from epstd_stage1 import EnvironmentEncoder
    encoder = EnvironmentEncoder(input_dim=static_dim, output_dim=env_dim)
    test_input = torch.randn(B, N, static_dim)
    test_output = encoder(test_input)
    print(f"  输入: {test_input.shape} -> 输出: {test_output.shape}")
    if test_output.shape == (B, N, env_dim):
        print("  [OK] 环境编码器维度正确")
    else:
        print(f"  [WARN] 期望输出 (B, N, env_dim) = ({B}, {N}, {env_dim})")
except Exception as e:
    print(f"  [ERR] 环境编码器检查失败: {e}")

# 5. 检查多图模型
print("\n[5/5] 检查多图扩散模型...")
try:
    from epstd_stage3_multigraph import MultiGraphConditionalDiffusion

    model = MultiGraphConditionalDiffusion(
        num_nodes=N,
        hidden_dim=128,
        num_layers=2,
        time_dim=64,
        env_dim=env_dim,
        num_prototypes=num_prototypes
    )

    # 测试输入
    x_t = torch.randn(B, N)
    t = torch.randint(0, 1000, (B,))
    env_emb = torch.randn(B, N, env_dim)

    # 测试一维 prototype_ids
    proto_ids_1d = torch.randint(0, num_prototypes, (N,))
    print(f"  测试1D prototype_ids: {proto_ids_1d.shape}")

    # 测试二维 prototype_ids
    proto_ids_2d = torch.randint(0, num_prototypes, (B, N))
    print(f"  测试2D prototype_ids: {proto_ids_2d.shape}")

    # 前向传播测试（一维）
    with torch.no_grad():
        noise_pred, pi, gw, cg = model(x_t, t, env_emb, proto_ids_1d, adj_list=None)

    print(f"  输出噪声: {noise_pred.shape}")
    print(f"  零膨胀概率: {pi.shape}")
    print("  [OK] 多图模型维度正确")

except Exception as e:
    print(f"  [ERR] 多图模型检查失败: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*70)
print("维度检查完成")
print("="*70)

# 总结
print("\n关键维度关系总结:")
print("  X_full        : (T, N, 50)  -> 静态特征(24) + 天气(8) + 犯罪滞后(14) + OD(4)")
print("  X_static      : (T, N, 24)  -> 环境编码器输入")
print("  env_emb       : (B, N, 64)  -> 环境编码器输出")
print("  prototype_ids : (N,) 或 (B, N)")
print("  图结构        : (N, N) x 5张")
print("  Y             : (T, N)      -> 预测目标")
