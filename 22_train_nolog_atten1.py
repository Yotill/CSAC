# -*- coding: utf-8 -*-
"""
Created on Fri Nov 21 10:05:33 2025

@author: Administrator
"""
import os, glob, joblib
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torch.amp import autocast, GradScaler

# ============ 用户配置 ============ #
BATCH_SIZE = 512
EPOCHS = 500
INPUT_DIM = 16
OUTPUT_DIM = 10
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 1e-4
DROPOUT = 0.1
NUM_UNITS = 256
DEPTH = 6
GRAD_CLIP = 1.0
EARLY_STOPPING_PATIENCE = 30
RESUME_TRAINING = True

OUT_DIR = r"G:\fwq_data\PT\seawifs\output_saved_modis_seawifs"
MODEL_LAST = os.path.join(OUT_DIR, "nn_model_last_attn1_seawifs.pt")
MODEL_BEST = os.path.join(OUT_DIR, "nn_model_best_attn1_seawifs.pt")
SCALER_X_PATH = os.path.join(OUT_DIR, "scaler_X.pkl")
SCALER_Y_PATH = os.path.join(OUT_DIR, "scaler_Y.pkl")
LOSS_PLOT_PATH = os.path.join(OUT_DIR, "loss_curve_attn1_seawifs.png")

BAND_WEIGHTS = np.array([1.0,1.0,1.0,1.0,1.0,1.0,1.2,1.6,1.8,1.6], dtype=np.float32)
SMOOTH_LAMBDA = 5e-4

# ============ 模型结构 ============ #
class ResidualBlock(nn.Module):
    def __init__(self, dim, drop=0.0):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim * 2)
        self.ln1 = nn.LayerNorm(dim * 2)
        self.act = nn.GELU()
        self.drop = nn.Dropout(drop)
        self.fc2 = nn.Linear(dim * 2, dim)
        self.ln2 = nn.LayerNorm(dim)

    def forward(self, x):
        h = self.fc1(x); h = self.ln1(h); h = self.act(h)
        h = self.drop(h); h = self.fc2(h); h = self.ln2(h)
        return x + h

class SpectralAttentionHead(nn.Module):
    def __init__(self, hidden, n_bands, attn_dim=64, n_heads=4):
        super().__init__()
        self.to_seq = nn.Linear(hidden, n_bands * attn_dim)
        self.attn = nn.MultiheadAttention(embed_dim=attn_dim, num_heads=n_heads, batch_first=True)
        self.out = nn.Linear(attn_dim, 1)
        self.n_bands = n_bands
        self.attn_dim = attn_dim

    def forward(self, h):
        B = h.size(0)
        seq = self.to_seq(h).view(B, self.n_bands, self.attn_dim)
        out, _ = self.attn(seq, seq, seq)
        out = self.out(out).squeeze(-1)
        return out

class ResMLP_Attn(nn.Module):
    def __init__(self, input_dim, output_dim, hidden=NUM_UNITS, depth=DEPTH, drop=DROPOUT,
                 attn_dim=64, n_heads=4):
        super().__init__()
        self.inp = nn.Linear(input_dim, hidden)
        self.act = nn.GELU()
        self.blocks = nn.Sequential(*[ResidualBlock(hidden, drop) for _ in range(depth)])
        self.out_ln = nn.LayerNorm(hidden)
        self.head = SpectralAttentionHead(hidden, output_dim, attn_dim, n_heads)

    def forward(self, x):
        h = self.act(self.inp(x))
        h = self.blocks(h)
        h = self.out_ln(h)
        return self.head(h)

# ============ 数据预处理 ============ #
def load_npz_file(npz_path):
    with np.load(npz_path, allow_pickle=True) as data:
        X = data["data_matrix"].astype(np.float32)
        Y = data["labels"].astype(np.float32)
    return X, Y

def create_dataset(npz_files, scaler_X, scaler_Y):
    X_all, Y_all = [], []
    for f in npz_files:
        X, Y = load_npz_file(f)
        X = scaler_X.transform(X)
        Yz = scaler_Y.transform(Y)
        X_all.append(X); Y_all.append(Yz)
    return TensorDataset(torch.from_numpy(np.vstack(X_all)),
                         torch.from_numpy(np.vstack(Y_all)))

def plot_loss_curve(train_loss, val_loss, path):
    plt.figure(figsize=(8, 6))
    plt.plot(train_loss, label='Train')
    plt.plot(val_loss, label='Val')
    plt.title('Model Loss (raw space)', fontsize=14)
    plt.xlabel('Epoch'); plt.ylabel('Loss')
    plt.grid(True); plt.legend()
    plt.savefig(path, dpi=300, bbox_inches='tight'); plt.close()

# ============ 损失与正则 ============ #
class BandWeightedSmoothL1(nn.Module):
    def __init__(self, band_weights: torch.Tensor, beta=1.0):
        super().__init__()
        self.beta = beta
        self.band_weights = band_weights

    def forward(self, pred, target):
        diff = torch.abs(pred - target)
        loss = torch.where(diff < self.beta, 0.5 * diff**2 / self.beta, diff - 0.5 * self.beta)
        loss = loss * self.band_weights[None, :]
        return loss.mean()

def spectral_smoothness_regularizer(pred, lam=SMOOTH_LAMBDA):
    if pred.size(1) < 3 or lam <= 0:
        return pred.new_tensor(0.0)
    d2 = pred[:, 2:] - 2 * pred[:, 1:-1] + pred[:, :-2]
    return lam * (d2**2).mean()

# ============ Early Stopping ============ #
class EarlyStopping:
    def __init__(self, patience=20):
        self.patience = patience
        self.best = np.inf
        self.counter = 0
        self.should_stop = False
    def step(self, val_loss):
        if val_loss < self.best:
            self.best = val_loss; self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

# ============ 主训练函数 ============ #
def train_and_save(train_dir, val_dir):
    os.makedirs(OUT_DIR, exist_ok=True)

    train_files = sorted(glob.glob(os.path.join(train_dir, "*.npz")))
    val_files = sorted(glob.glob(os.path.join(val_dir, "*.npz")))
    assert train_files and val_files, "❌ 没有找到 .npz 文件"

    scaler_X = StandardScaler()
    scaler_Y = StandardScaler()
    for f in train_files:
        X, Y = load_npz_file(f)
        scaler_X.partial_fit(X)
        scaler_Y.partial_fit(Y)
    joblib.dump(scaler_X, SCALER_X_PATH)
    joblib.dump(scaler_Y, SCALER_Y_PATH)

    train_ds = create_dataset(train_files, scaler_X, scaler_Y)
    val_ds   = create_dataset(val_files, scaler_X, scaler_Y)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              pin_memory=True, num_workers=4, persistent_workers=True)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                              pin_memory=True, num_workers=4, persistent_workers=True)

    model = ResMLP_Attn(INPUT_DIM, OUTPUT_DIM).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=10)
    band_w = torch.from_numpy(BAND_WEIGHTS).to(device)
    criterion = BandWeightedSmoothL1(band_w, beta=1.0).to(device)
    scaler = GradScaler(device='cuda' if device.type=='cuda' else 'cpu')

    start_epoch, history_train, history_val, best_val = 0, [], [], np.inf
    if RESUME_TRAINING and os.path.exists(MODEL_LAST):
        ckpt = torch.load(MODEL_LAST, map_location=device)
        model.load_state_dict(ckpt['model'])
        opt.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        scaler.load_state_dict(ckpt['scaler'])
        history_train = ckpt['history_train']
        history_val   = ckpt['history_val']
        start_epoch   = ckpt['epoch'] + 1
        best_val      = ckpt['best_val']
        print(f"🔁 从第 {start_epoch} 轮继续训练 (best_val={best_val:.6f})")

    stopper = EarlyStopping(patience=EARLY_STOPPING_PATIENCE)
    for epoch in range(start_epoch, EPOCHS):
        model.train(); train_losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            with autocast(device_type='cuda' if device.type=='cuda' else 'cpu'):
                preds = model(xb)
                base_loss = criterion(preds, yb)
                smooth_reg = spectral_smoothness_regularizer(preds, lam=SMOOTH_LAMBDA)
                loss = base_loss + smooth_reg
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(opt); scaler.update()
            train_losses.append(loss.item())
        avg_train = float(np.mean(train_losses))
        history_train.append(avg_train)

        model.eval(); val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                with autocast(device_type='cuda' if device.type=='cuda' else 'cpu'):
                    preds = model(xb)
                    base_loss = criterion(preds, yb)
                    smooth_reg = spectral_smoothness_regularizer(preds, lam=SMOOTH_LAMBDA)
                    vloss = base_loss + smooth_reg
                val_losses.append(vloss.item())
        avg_val = float(np.mean(val_losses))
        history_val.append(avg_val)

        scheduler.step(avg_val)
        print(f"📘 Epoch {epoch+1}/{EPOCHS} | Train={avg_train:.6f} | Val={avg_val:.6f} | LR={opt.param_groups[0]['lr']:.2e}")

        torch.save({
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': opt.state_dict(),
            'scheduler': scheduler.state_dict(),
            'scaler': scaler.state_dict(),
            'history_train': history_train,
            'history_val': history_val,
            'best_val': best_val
        }, MODEL_LAST)

        if avg_val < best_val:
            best_val = avg_val
            torch.save({'model': model.state_dict()}, MODEL_BEST)

        stopper.step(avg_val)
        if stopper.should_stop:
            print(f"⏹️ 提前停止，最佳 Val={best_val:.6f}")
            break

    plot_loss_curve(history_train, history_val, LOSS_PLOT_PATH)
    print(f"✅ 训练完成，最佳模型保存至 {MODEL_BEST}")

# ============ main ============ #
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    print(f"✅ 使用设备: {device}")
    train_and_save(
        train_dir=r"G:\fwq_data\PT\seawifs\dataset\train",
        val_dir=r"G:\fwq_data\PT\seawifs\dataset\validate"
    )


