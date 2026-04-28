# -*- coding: utf-8 -*-
"""
plot_scatter_train_vali_resmlp_attn.py
--------------------------------------------------------
读取拆分后的 npz 文件，加载 ResMLP+SpectralAttention 模型，
绘制预测 Rrs vs MODIS Rrs 的散点图（分波段）。
--------------------------------------------------------
输出:
   figs/scatter_modis_viirs/<tag>/
      <tag>_pred_vs_modis.png
      <tag>_pred_vs_modis_stats.xlsx
"""
import os, glob, joblib, torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch import nn
from sklearn.metrics import mean_squared_error
from scipy.stats import linregress
from matplotlib.colors import LogNorm
from matplotlib.ticker import MaxNLocator, FuncFormatter

plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size'  : 14,
    'mathtext.fontset': 'custom',
    'mathtext.rm': 'Times New Roman',
    'mathtext.it': 'Times New Roman:italic',
    'mathtext.bf': 'Times New Roman:bold',
})

# ────────── 路径 / 参数 ──────────
OUT_DIR       = r"G:\fwq_data\PT\seawifs\output_saved_modis_seawifs"
FIG_DIR       = r"G:\fwq_data\PT\seawifs\figs"
MODEL_PATH    = os.path.join(OUT_DIR, "nn_model_last_log_attn1_seawifs.pt")
SCALER_X_PATH = os.path.join(OUT_DIR, "scaler_X.pkl")
SCALER_Y_PATH = os.path.join(OUT_DIR, "scaler_Y_log1.pkl")
PLOT_BASE_DIR = os.path.join(FIG_DIR, "scatter_modis_seawifs")

DATASET_DIRS = {
    "train": r"G:\fwq_data\PT\seawifs\dataset\train",
    "val"  : r"G:\fwq_data\PT\seawifs\dataset\validate",
}

BATCH_SIZE = 512
# 10 个 MODIS 波段
MODIS_BANDS = [412, 443, 469, 488, 531, 547, 555, 645, 667, 678]

# ────────── 模型结构（与训练保持一致） ──────────
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
        seq = self.to_seq(h).view(B, self.n_bands, self.attn_dim)  # [B, n_bands, attn_dim]
        out, _ = self.attn(seq, seq, seq)                          # [B, n_bands, attn_dim]
        out = self.out(out).squeeze(-1)                            # [B, n_bands]
        return out

class ResMLP_Attn(nn.Module):
    def __init__(self, input_dim, output_dim, hidden=256, depth=6, drop=0.25,
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

# ────────── 数据加载 ──────────
def load_folder(folder):
    Xs, Ys = [], []
    files = glob.glob(os.path.join(folder, "*.npz"))
    for f in files:
        with np.load(f, allow_pickle=True) as d:
            Xs.append(d["data_matrix"])
            Ys.append(d["labels"])
    return np.vstack(Xs), np.vstack(Ys)

def get_band_range(b):
    return {412:(0,0.03), 443:(0,0.03), 469:(0,0.03),
            488:(0,0.03), 531:(0,0.03), 547:(0,0.03),
            555:(0,0.03), 645:(0,0.012), 667:(0,0.003),
            678:(0,0.003)}.get(b, (0,0.03))

def fmt(x, _):
    return "0" if x==0 else f"{x:.3f}".rstrip('0').rstrip('.')

# ────────── 绘图函数 ──────────
def make_figure(y_pred, y_true, bands, out_png, out_xlsx):
    fig, axs = plt.subplots(2, 5, figsize=(14, 6))
    axs = axs.flatten()
    stats=[]
    for i, band in enumerate(bands):
        ax = axs[i]
        x, y = y_pred[:, i], y_true[:, i]
        mask = np.isfinite(x) & np.isfinite(y)
        x, y = x[mask], y[mask]

        vmin, vmax = get_band_range(band)
        ax.plot([vmin, vmax], [vmin, vmax], 'k-', lw=0.8)

        h = ax.hist2d(x, y, bins=400, cmin=1, cmap='rainbow',
                      norm=LogNorm(vmin=1, vmax=5e2),
                      range=[[vmin, vmax], [vmin, vmax]])

        slope, intercept, r, *_ = linregress(x, y)
        r2   = r*r
        rmse = np.sqrt(mean_squared_error(x, y))
        bias = np.mean(y - x)
        mare = np.mean(np.abs((y - x) / (x + 1e-12))) * 100

        ax.set_title(f"{band} nm", fontsize=12, weight='bold')
        ax.set_xlim(vmin, vmax); ax.set_ylim(vmin, vmax); ax.set_aspect('equal')
        ax.xaxis.set_major_locator(MaxNLocator(nbins=3))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
        ax.xaxis.set_major_formatter(FuncFormatter(fmt))
        ax.yaxis.set_major_formatter(FuncFormatter(fmt))
        ax.text(0.05, 0.95,
                f"$R^2$={r2:.2f}\nRMSE={rmse:.4f}\nBias={bias:.1e}\nMARE={mare:.1f}%",
                transform=ax.transAxes, va='top', fontsize=10)

        stats.append({'band':band,'R2':r2,'RMSE':rmse,'Bias':bias,'MARE':mare,'N':x.size})

    fig.text(0.49,0.05,"Predicted $R_{rs}$ (sr$^{-1}$)",ha='center',fontsize=14)
    fig.text(0.025,0.55,"MODIS $R_{rs}$ (sr$^{-1}$)",va='center',rotation='vertical',fontsize=14)
    plt.tight_layout(rect=[0.03,0.1,0.95,0.99])
    cax = fig.add_axes([0.93,0.23,0.01,0.64])
    fig.colorbar(h[3], cax=cax).ax.set_title('Pixels',weight='bold',fontsize=12,pad=6)
    fig.savefig(out_png,dpi=300); plt.close(fig)
    pd.DataFrame(stats).to_excel(out_xlsx,index=False)

# ────────── 主流程 ──────────
def plot_scatter(tag, folder):
    print(f"\n📂 处理 {tag} → {folder}")
    out_dir = os.path.join(PLOT_BASE_DIR, tag)
    os.makedirs(out_dir, exist_ok=True)

    X_raw, Y_true = load_folder(folder)
    input_dim = X_raw.shape[1]
    output_dim = Y_true.shape[1]
    print(f"🔎 输入维度={input_dim}, 输出维度={output_dim}")

    scaler_X = joblib.load(SCALER_X_PATH)
    scaler_Y = joblib.load(SCALER_Y_PATH)
    X_proc = scaler_X.transform(X_raw).astype(np.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = ResMLP_Attn(input_dim, output_dim).to(device)
    ckpt   = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model']); model.eval()

    preds = []
    with torch.no_grad():
        for i in range(0, X_proc.shape[0], BATCH_SIZE):
            batch = torch.from_numpy(X_proc[i:i+BATCH_SIZE]).to(device)
            preds.append(model(batch).cpu().numpy())
    y_pred = scaler_Y.inverse_transform(np.vstack(preds))

    make_figure(y_pred, Y_true, MODIS_BANDS,
                out_png=os.path.join(out_dir, f"{tag}_pred_vs_modis_attn.png"),
                out_xlsx=os.path.join(out_dir, f"{tag}_pred_vs_modis_stats_attn.xlsx"))

    print(f"✅ {tag.upper()} 完成 → {out_dir}")

# ────────── main ──────────
if __name__ == "__main__":
    for name, folder in DATASET_DIRS.items():
        if os.path.isdir(folder):
            plot_scatter(name, folder)
        else:
            print(f"⚠️ 目录不存在，跳过 {name}: {folder}")