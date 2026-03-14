# -*- coding: utf-8 -*-
import os, time
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# =========================
# Device & Backend
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if device.type == "cuda":
    torch.backends.cudnn.benchmark = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

# =========================
# Hyperparams & Paths
# =========================
TRAINING_DATA_DIR = r"C:\mmWave\mmWave\radar-gesture-recognition-chore-update-20250815\src\data\processed_data\train6_dataset.npz"
VAL_DATA_DIR      = r"C:\mmWave\mmWave\radar-gesture-recognition-chore-update-20250815\src\data\processed_data\val5_dataset.npz"

WINDOW_SIZE   = 55
STEP_SIZE     = 1
VAL_STEP_SIZE = 1
BATCH_SIZE    = 32
EPOCHS        = 100
LEARNING_RATE = 1e-4
NUM_CLASSES   = None  # 會在 main() 裡由資料集自動設定

# Regularization / saving
SAVE_MIN_DELTA = 5e-3  # 只用來判斷「是否更好就存檔」

# Label tricks
LABEL_SMOOTH = 0.05    # 0.05~0.1；只對 hard label 生效
LABEL_OFFSET = 0       # 標籤偏移：0 / +2 / +4 做 A/B
MID_JITTER   = 0       # 只在訓練集啟用的 ±2 幀抖動；(0/1/2/3)

# Output dir
timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time()))
MODEL_SAVE_PATH = os.path.join("output", "models", timestamp)
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

# =========================
# Sliding window Dataset (mmap)
# =========================
class SlidingWindowDataset(Dataset):
    """
    從 .npz (features: (N,C,H,W,F)) 讀取，產生 (C, WINDOW, H, W) 窗；
    標籤取：窗口中心 + 偏移 (+訓練抖動) 的幀。
    """
    def __init__(self, npz_path, window_size, step_size,
                 is_train=False, label_offset=0, mid_jitter=0):
        self.path = npz_path
        self.npz = np.load(self.path, allow_pickle=True, mmap_mode='r')

        feats = self.npz["features"]  # (N,C,H,W,F) 或 (C,H,W,F)
        if feats.ndim == 4:
            feats = feats[np.newaxis, ...]
        if feats.ndim != 5:
            raise ValueError(f"'features' 維度應為 5 或 4，現在是 {feats.shape}")
        self.feats = feats
        self.N, self.C, self.H, self.W, self.F = self.feats.shape
        if self.C != 2:
            print(f"[警告] features 的通道數 C={self.C}，而模型 in_channels=2")

        self.has_gts = "ground_truths" in self.npz
        if self.has_gts:
            gts = self.npz["ground_truths"]  # (N,F,Ccls) 或 (F,Ccls)
            if gts.ndim == 2:
                gts = gts[np.newaxis, ...]
            if gts.ndim != 3 or gts.shape[0] != self.N or gts.shape[1] != self.F:
                raise ValueError(f"ground_truths 尺寸不合：{gts.shape}")
            self.gts = gts
            self.num_classes = int(self.gts.shape[2])
        else:
            labels = self.npz["labels"]      # (N,F) 或 (F,)
            if labels.ndim == 1:
                labels = labels[np.newaxis, ...]
            if labels.ndim != 2 or labels.shape[0] != self.N or labels.shape[1] != self.F:
                raise ValueError(f"labels 尺寸不合：{labels.shape}")
            classes = self.npz["classes"] if "classes" in self.npz else None
            self.num_classes = int(len(classes)) if classes is not None else 4
            self.labels = labels

        self.window = int(window_size)
        self.step   = int(step_size)
        self.is_train     = bool(is_train)
        self.label_offset = int(label_offset)
        self.mid_jitter   = int(mid_jitter)

        # 可用窗數 & 累積
        self.windows_per_seq = []
        for _ in range(self.N):
            n_win = (self.F - self.window) // self.step + 1 if self.F >= self.window else 0
            self.windows_per_seq.append(max(0, n_win))
        self.cum_windows = np.cumsum(self.windows_per_seq)
        self.total = int(self.cum_windows[-1]) if len(self.cum_windows) > 0 else 0
        if self.total == 0:
            raise RuntimeError("資料長度不足以產生任何滑窗，請調小 WINDOW_SIZE 或確認 F")

    def __len__(self):
        return self.total

    def _index_to_seq(self, idx):
        seq = int(np.searchsorted(self.cum_windows, idx, side='right'))
        start_offset = idx - (self.cum_windows[seq-1] if seq > 0 else 0)
        start = start_offset * self.step
        return seq, start

    def __getitem__(self, idx):
        seq, start = self._index_to_seq(idx)
        end = start + self.window

        # 中心 + 偏移 + (訓練)抖動
        mid = start + self.window // 2 + self.label_offset
        if self.is_train and self.mid_jitter > 0:
            mid += np.random.randint(-self.mid_jitter, self.mid_jitter + 1)
        mid = max(start, min(end - 1, mid))  # 邊界保護

        # 取窗：(C,H,W,WINDOW) -> (C,WINDOW,H,W)
        win = self.feats[seq, :, :, :, start:end]
        win = np.transpose(win, (0, 3, 1, 2))
        x = torch.from_numpy(win.astype(np.float32))

        # 標籤：有 gts 用 gts；否則 one-hot + smoothing
        if self.has_gts:
            y_np = self.gts[seq, mid]
        else:
            hard = int(self.labels[seq, mid])
            y_np = np.full((self.num_classes,), LABEL_SMOOTH / self.num_classes, dtype=np.float32)
            y_np[np.clip(hard, 0, self.num_classes-1)] = 1.0 - LABEL_SMOOTH + (LABEL_SMOOTH / self.num_classes)
        y = torch.from_numpy(y_np.astype(np.float32))
        return x, y

# =========================
# 3D CNN Model
# =========================
class Gesture3DCNN(nn.Module):
    def __init__(self, num_classes=4):
        super(Gesture3DCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv3d(2, 32, 3), nn.ReLU(), nn.MaxPool3d(2), nn.BatchNorm3d(32),
            nn.Conv3d(32, 64, 3), nn.ReLU(), nn.MaxPool3d(2), nn.BatchNorm3d(64),
            nn.Conv3d(64, 128, 3), nn.ReLU(), nn.MaxPool3d(2), nn.BatchNorm3d(128),
        )
        self.dropout3d = nn.Dropout3d(p=0.2)
        self.global_avg_pool = nn.AdaptiveAvgPool3d((1,1,1))
        self.classifier = nn.Sequential(
            nn.Linear(128, 128), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.dropout3d(x)
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)  # logits

# =========================
# AMP 相容處理
# =========================
try:
    from torch.amp import autocast as amp_autocast, GradScaler as AmpGradScaler
    _USE_NEW_AMP = True
except Exception:
    from torch.cuda.amp import autocast as amp_autocast, GradScaler as AmpGradScaler  # type: ignore
    _USE_NEW_AMP = False

def _make_scaler():
    if device.type == 'cuda':
        return AmpGradScaler('cuda') if _USE_NEW_AMP else AmpGradScaler(enabled=True)
    return AmpGradScaler(enabled=False)

# ========= 訓練（固定跑滿 epoch；不含早停） =========
def train_model(train_loader, val_loader, num_classes, num_epochs=None):
    epochs_to_run = num_epochs if num_epochs is not None else EPOCHS

    model = Gesture3DCNN(num_classes=num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=2e-4)
    criterion = nn.KLDivLoss(reduction="batchmean")
    scaler = _make_scaler()

    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    best_val_loss = float('inf')

    for epoch in range(epochs_to_run):
        # ---- train ----
        model.train()
        running_loss = 0.0; correct = 0; total = 0
        pbar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{epochs_to_run}] (Training)", unit="batch")

        for batch_x, batch_y in pbar:
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)

            optimizer.zero_grad()
            with amp_autocast(device_type='cuda', enabled=(device.type == "cuda")):
                logits = model(batch_x)
                log_probs = torch.log_softmax(logits, dim=1)
                loss = criterion(log_probs, batch_y)

            scaler.scale(loss).backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # 可選
            scaler.step(optimizer); scaler.update()

            running_loss += loss.item() * batch_x.size(0)
            preds   = torch.argmax(logits, dim=1)
            targets = torch.argmax(batch_y, dim=1)
            correct += (preds == targets).sum().item()
            total   += batch_x.size(0)
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        epoch_loss = running_loss / max(1, len(train_loader.dataset))
        train_acc = correct / max(1, total)
        history['train_loss'].append(epoch_loss)
        history['train_acc'].append(train_acc)

        # ---- val ----
        model.eval()
        val_running_loss = 0.0; v_correct = 0; v_total = 0
        with torch.no_grad():
            for val_x, val_y in val_loader:
                val_x = val_x.to(device, non_blocking=True)
                val_y = val_y.to(device, non_blocking=True)
                with amp_autocast(device_type='cuda', enabled=(device.type == "cuda")):
                    logits = model(val_x)
                    log_probs = torch.log_softmax(logits, dim=1)
                    val_loss = criterion(log_probs, val_y)

                val_running_loss += val_loss.item() * val_x.size(0)
                v_preds   = torch.argmax(logits, dim=1)
                v_targets = torch.argmax(val_y, dim=1)
                v_correct += (v_preds == v_targets).sum().item()
                v_total   += val_x.size(0)

        val_epoch_loss = val_running_loss / max(1, len(val_loader.dataset))
        val_acc = v_correct / max(1, v_total)
        history['val_loss'].append(val_epoch_loss)
        history['val_acc'].append(val_acc)

        print(f"[Epoch {epoch+1}/{epochs_to_run}] "
              f"Train Loss: {epoch_loss:.4f} || Val Loss: {val_epoch_loss:.4f} || "
              f"Train Acc: {train_acc:.4f} || Val Acc: {val_acc:.4f}")

        # 存最佳（不早停）
        if val_epoch_loss < best_val_loss - SAVE_MIN_DELTA:
            best_val_loss = val_epoch_loss
            best_model_path = os.path.join(MODEL_SAVE_PATH, f"epoch_{epoch+1}_valLoss_{val_epoch_loss:.4f}.pth")
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved: {best_model_path}")

    final_model_path = os.path.join(MODEL_SAVE_PATH, "3d_cnn_model_amp.pth")
    torch.save(model.state_dict(), final_model_path)
    print(f"Training complete. Final model saved at {final_model_path}")
    return model, history

# =========================
# Plot curves
# =========================
def plot_history(history):
    epochs_range = range(1, len(history['train_loss']) + 1)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, history['train_loss'], label='Training Loss')
    plt.plot(epochs_range, history['val_loss'], label='Validation Loss')
    plt.title('Loss Curves'); plt.xlabel('Epochs'); plt.ylabel('Loss'); plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, history['train_acc'], label='Training Accuracy')
    plt.plot(epochs_range, history['val_acc'], label='Validation Accuracy')
    plt.title('Accuracy Curves'); plt.xlabel('Epochs'); plt.ylabel('Accuracy'); plt.legend()
    plt.tight_layout(); plt.show()

# =========================
# Main
# =========================
def main():
    # 訓練資料：維持 STEP_SIZE=1
    train_ds = SlidingWindowDataset(
        TRAINING_DATA_DIR, WINDOW_SIZE, STEP_SIZE,
        is_train=True,  label_offset=LABEL_OFFSET, mid_jitter=MID_JITTER
    )
    # 驗證資料：改用較大的步進，降低樣本相關性
    val_ds   = SlidingWindowDataset(
        VAL_DATA_DIR,      WINDOW_SIZE, VAL_STEP_SIZE,   # ★ 這裡用 VAL_STEP_SIZE
        is_train=False, label_offset=0, mid_jitter=0
    )
    ...


    global NUM_CLASSES
    NUM_CLASSES = train_ds.num_classes
    print("NUM_CLASSES set to:", NUM_CLASSES)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=0, pin_memory=(device.type == "cuda"), persistent_workers=False
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=0, pin_memory=(device.type == "cuda"), persistent_workers=False
    )

    model, history = train_model(train_loader, val_loader, NUM_CLASSES, num_epochs=EPOCHS)
    plot_history(history)

if __name__ == "__main__":
    main()
