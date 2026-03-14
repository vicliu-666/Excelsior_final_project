import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix, classification_report

# ---------------------------------------------------
# Parameters
# ---------------------------------------------------
TEST_DATA_FILE = r"C:\mmWave\mmWave\radar-gesture-recognition-chore-update-20250815\src\data\processed_data\val5_dataset.npz"
MODEL_PATH     = r"C:\mmWave\mmWave\radar-gesture-recognition-chore-update-20250815\src\output\models\window_55_0_9\3d_cnn_model_background_0_9.pth"

WINDOW_SIZE = 55
HIGH_TH     = 0.5
LOW_TH      = 0.1

CLASS_NAMES = ['background', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
NUM_CLASSES = len(CLASS_NAMES)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ---------------------------------------------------
# 1. Model definition
# ---------------------------------------------------
class Gesture3DCNN(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv3d(2, 32, 3), nn.ReLU(), nn.MaxPool3d(2), nn.BatchNorm3d(32),
            nn.Conv3d(32, 64, 3), nn.ReLU(), nn.MaxPool3d(2), nn.BatchNorm3d(64),
            nn.Conv3d(64, 128, 3), nn.ReLU(), nn.MaxPool3d(2), nn.BatchNorm3d(128),
        )
        self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(128, 128), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        logits = self.classifier(x)
        return torch.softmax(logits, dim=1)

# ---------------------------------------------------
# 2. Load data
# ---------------------------------------------------
data = np.load(TEST_DATA_FILE, allow_pickle=True)
features = data['features']   # (N, 2, 32, 32, 90)
labels   = data['labels']     # (N, 90)
classes  = data['classes']

print("features shape:", features.shape)
print("labels shape  :", labels.shape)
print("classes       :", classes)

# ---------------------------------------------------
# 3. Load model
# ---------------------------------------------------
model = Gesture3DCNN().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
model.eval()

# ---------------------------------------------------
# Helpers
# ---------------------------------------------------
def extract_window(clip_feat, center_frame, window_size=WINDOW_SIZE):
    """
    clip_feat: (2, 32, 32, frames)
    return  : (1, 2, window, 32, 32)
    """
    total = clip_feat.shape[-1]
    half = window_size // 2

    if total < window_size:
        raise ValueError(f"frames={total} 小於 WINDOW_SIZE={window_size}")

    start = max(0, min(center_frame - half, total - window_size))
    end = start + window_size

    win = clip_feat[..., start:end]         # (2, 32, 32, window)
    win = np.transpose(win, (0, 3, 1, 2))   # (2, window, 32, 32)
    return np.expand_dims(win, 0)           # (1, 2, window, 32, 32)

def get_true_clip_label(frame_labels):
    """
    frame_labels: (90,)
    0 = background, 1~10 = gestures
    """
    frame_labels = np.asarray(frame_labels).astype(int)
    non_bg = frame_labels[frame_labels != 0]

    if len(non_bg) == 0:
        return 0

    return np.bincount(non_bg, minlength=NUM_CLASSES).argmax()

def predict_clip_label(model, clip_feat):
    """
    clip_feat: (2, 32, 32, 90)
    """
    frames = clip_feat.shape[-1]
    probs = np.zeros((frames, NUM_CLASSES), dtype=np.float32)

    with torch.no_grad():
        for t in range(frames):
            win = extract_window(clip_feat, t)
            inp = torch.from_numpy(win).float().to(device)
            out = model(inp).cpu().numpy().squeeze()   # (11,)
            probs[t] = out

    # dual-threshold sequence
    pred_seq = np.zeros(frames, dtype=int)
    current = 0

    for t in range(frames):
        non_bg = probs[t, 1:]                  # classes 1~10
        i_max = np.argmax(non_bg) + 1
        p_max = non_bg[i_max - 1]

        if current == 0:
            if p_max >= HIGH_TH:
                current = i_max
        else:
            if p_max < LOW_TH:
                current = 0
            else:
                current = i_max

        pred_seq[t] = current

    # clip-level prediction
    non_bg_preds = pred_seq[pred_seq != 0]
    if len(non_bg_preds) == 0:
        pred_lbl = 0
    else:
        pred_lbl = np.bincount(non_bg_preds, minlength=NUM_CLASSES).argmax()

    return int(pred_lbl), pred_seq, probs

# ---------------------------------------------------
# 4. Inference
# ---------------------------------------------------
true_clip_labels = []
pred_clip_labels = []

N = len(features)
print(f"Total clips: {N}")

for idx in range(N):
    clip_feat = features[idx]     # (2,32,32,90)
    frame_lbl = labels[idx]       # (90,)

    true_lbl = get_true_clip_label(frame_lbl)
    pred_lbl, pred_seq, probs = predict_clip_label(model, clip_feat)

    true_clip_labels.append(int(true_lbl))
    pred_clip_labels.append(int(pred_lbl))

    print(f"[{idx+1:03d}/{N}] True={CLASS_NAMES[true_lbl]:<10s} Pred={CLASS_NAMES[pred_lbl]:<10s}")

# ---------------------------------------------------
# 5. Confusion matrix
# ---------------------------------------------------
cm = confusion_matrix(
    true_clip_labels,
    pred_clip_labels,
    labels=np.arange(NUM_CLASSES)
)

print("\nConfusion Matrix:")
print(cm)

report = classification_report(
    true_clip_labels,
    pred_clip_labels,
    labels=np.arange(NUM_CLASSES),
    target_names=CLASS_NAMES,
    digits=4,
    zero_division=0
)
print("\nClassification Report:")
print(report)

# ---------------------------------------------------
# 6. Plot
# ---------------------------------------------------
plt.figure(figsize=(12, 10))
ax = sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=CLASS_NAMES,
    yticklabels=CLASS_NAMES,
    annot_kws={"size": 10}
)

ax.set_xticklabels(CLASS_NAMES, rotation=0, ha='center', fontsize=10)
ax.set_yticklabels(CLASS_NAMES, rotation=0, va='center', fontsize=10)

plt.xlabel("Predicted", fontsize=14)
plt.ylabel("True", fontsize=14)
plt.title("Clip-level Confusion Matrix (0-9 + background)", fontsize=16)
plt.tight_layout()
plt.show()