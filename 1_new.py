# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 13:14:27 2026

@author: Administrator
"""

# -*- coding: utf-8 -*-
"""
将 .mat 数据拆分为 train / validate 的 .npz
并额外保存 modis 第20列的近岸/大洋标签
"""

import h5py
import numpy as np
import os
from sklearn.model_selection import train_test_split

# ========== 用户配置 ==========
MAT_PATH = r"G:\fwq_data\PT\seawifs\modis_seawifs_open_coastal_atmos_HQ_0915.mat"
OUT_DIR = r"G:\fwq_data\PT\seawifs\dataset2"
TRAIN_DIR = os.path.join(OUT_DIR, "train")
VAL_DIR = os.path.join(OUT_DIR, "validate")
os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(VAL_DIR, exist_ok=True)

CHUNK_SIZE = 500_000
VAL_RATIO = 0.2
RANDOM_STATE = 42

# 注意：
# 你说的是“第20列”，Python索引通常应为19
DOMAIN_COL = 19

# ========== 1. 读取大矩阵 ==========
print(f"📥 Loading {MAT_PATH} ...")
with h5py.File(MAT_PATH, 'r') as f:
    seawifs = np.array(f['seawifs']).T
    modis = np.array(f['modis']).T

print(f"✅ seawifs shape = {seawifs.shape}")
print(f"✅ modis   shape = {modis.shape}")

# ========== 2. 构造输入输出 ==========
# 这里按你现在给的写法，实际维度是 12 + 3 = 15 维
# range(8,20) -> 8~19，共12列
# range(22,25) -> 22~24，共3列
X = seawifs[:, list(range(8, 20)) + list(range(22, 25))].astype(np.float32)

# MODIS 10维输出
Y = modis[:, 8:18].astype(np.float32)

# 第20列：1为近岸，非1为大洋
domain = (modis[:, DOMAIN_COL] == 1).astype(np.float32).reshape(-1, 1)

print(f"✅ 输入形状: {X.shape}")
print(f"✅ 输出形状: {Y.shape}")
print(f"✅ 域标签形状: {domain.shape}")
print(f"📊 近岸比例: {domain.mean():.6f}")
print(f"📊 大洋比例: {1.0 - domain.mean():.6f}")

# ========== 3. 划分训练/验证 ==========
# 用 stratify 保证近岸/大洋比例在 train/val 中尽量一致
X_train, X_val, Y_train, Y_val, D_train, D_val = train_test_split(
    X, Y, domain,
    test_size=VAL_RATIO,
    random_state=RANDOM_STATE,
    shuffle=True,
    stratify=domain[:, 0]
)

print(f"📊 训练集样本数: {X_train.shape[0]}")
print(f"📊 验证集样本数: {X_val.shape[0]}")
print(f"📊 训练集近岸比例: {D_train.mean():.6f}")
print(f"📊 验证集近岸比例: {D_val.mean():.6f}")

# ========== 4. 保存函数 ==========
def save_chunks(X, Y, D, out_dir, prefix):
    n_samples = X.shape[0]
    n_chunks = (n_samples + CHUNK_SIZE - 1) // CHUNK_SIZE

    for i in range(n_chunks):
        start = i * CHUNK_SIZE
        end = min((i + 1) * CHUNK_SIZE, n_samples)

        np.savez_compressed(
            os.path.join(out_dir, f"{prefix}_chunk{i:03d}.npz"),
            data_matrix=X[start:end, :],
            labels=Y[start:end, :],
            domain_label=D[start:end, :]
        )
        print(f"💾 Saved {prefix}_chunk{i:03d}.npz ({end - start} samples)")

# ========== 5. 保存训练/验证数据 ==========
save_chunks(X_train, Y_train, D_train, TRAIN_DIR, "train")
save_chunks(X_val, Y_val, D_val, VAL_DIR, "val")

print("🎉 数据拆分完成！")