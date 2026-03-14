# -*- coding: utf-8 -*-
import os
import json
import h5py
import numpy as np
from pathlib import Path

# ===================== 路徑設定 =====================
DATA_DIR = r"C:\mmWave\mmWave\radar-gesture-recognition-chore-update-20250815\src\data\train6"

OUT_DIR = Path("data") / "processed_data"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_NAME = "train6_dataset.npz"
OUT_PATH = OUT_DIR / OUT_NAME
CLASSES_JSON = OUT_DIR / "classes.json"


# ===================== 小工具 =====================
def quick_check(data_dir):
    root = Path(data_dir)
    print("[check] DATA_DIR:", root)
    if not root.exists():
        print("[X] 目錄不存在")
        return
    subs = [d for d in root.iterdir() if d.is_dir()]
    print("[check] 子資料夾：", [d.name for d in subs])
    total = 0
    for d in subs:
        h5s = [f for f in d.iterdir() if f.suffix.lower() == ".h5"]
        print(f"  - {d.name}: {len(h5s)} 個 .h5")
        total += len(h5s)
    print("[check] 第一層 .h5 總數:", total)


def load_h5_file(file_path: str):
    """
    只讀 DS1，不再要求 LABEL。
    DS1 須為 (2, 32, 32, F)
    """
    with h5py.File(file_path, "r") as f:
        if "DS1" not in f:
            raise KeyError(f"{file_path} 缺少 'DS1'")
        ds1 = np.array(f["DS1"], dtype=np.float32)

    if ds1.ndim != 4:
        raise ValueError(f"{file_path} DS1 應為 4D，shape={ds1.shape}")
    if ds1.shape[0] != 2:
        raise ValueError(f"{file_path} DS1 第一維(通道)=2，shape={ds1.shape}")

    return ds1


# ===================== 主流程 =====================
def process_data():

    data_root = Path(DATA_DIR)
    if not data_root.exists():
        raise FileNotFoundError(f"DATA_DIR 不存在：{DATA_DIR}")

    # 類別順序：按字典序，沒有背景也沒關係
    # ====== 讓 background 永遠排最前 ======
    all_dirs = [d.name for d in data_root.iterdir() if d.is_dir()]

    if "background" in all_dirs:
        gesture_types = ["background"] + sorted([g for g in all_dirs if g != "background"])
    else:
        gesture_types = sorted(all_dirs)

    gesture_to_label = {g: i for i, g in enumerate(gesture_types)}

    print("Gesture mapping:", gesture_to_label)

    # 掃描全部資料
    samples = []  # (ds1, class_idx)
    max_length = 0

    for gesture in gesture_types:
        gdir = data_root / gesture
        class_idx = gesture_to_label[gesture]

        for name in os.listdir(gdir):
            if not name.lower().endswith(".h5"):
                continue
            fp = str(gdir / name)
            try:
                ds1 = load_h5_file(fp)
            except Exception as e:
                print(f"[讀取失敗] {fp} -> {e}")
                continue

            F = ds1.shape[-1]
            max_length = max(max_length, F)
            samples.append((ds1, class_idx))

    if not samples:
        raise RuntimeError("沒有讀到任何 .h5 樣本，請檢查資料夾層級與檔案內容")

    total_classes = len(gesture_types)
    print(f"[info] 樣本: {len(samples)}, 最大時間長度 Fmax={max_length}, 類別數={total_classes}")

    features_padded = []
    labels_framewise = []

    # ===================== 補長度 =====================
    for ds1, class_idx in samples:
        F = ds1.shape[-1]

        # ---- feature padding ----
        if F < max_length:
            pad = ((0, 0), (0, 0), (0, 0), (0, max_length - F))
            ds1_pad = np.pad(ds1, pad, mode="constant")
        else:
            ds1_pad = ds1[:, :, :, :max_length]

        # ---- label: 每一 frame 都是該類別（硬標籤） ----
        lbl = np.full((max_length,), class_idx, dtype=np.int32)

        features_padded.append(ds1_pad.astype(np.float32))
        labels_framewise.append(lbl)

    # 打包
    features_padded = np.stack(features_padded, axis=0)   # (N, 2, 32, 32, Fmax)
    labels_framewise = np.stack(labels_framewise, axis=0) # (N, Fmax)

    # 儲存
    np.savez(
        OUT_PATH,
        features=features_padded,
        labels=labels_framewise,
        classes=np.array(gesture_types, dtype=object),
    )
    with open(CLASSES_JSON, "w", encoding="utf-8") as f:
        json.dump(gesture_types, f, ensure_ascii=False, indent=2)

    print(f"[OK] 已輸出：{OUT_PATH}")
    print(f"  features: {features_padded.shape}")
    print(f"  labels:   {labels_framewise.shape}")
    print(f"  classes:  {gesture_types}")

    # 驗證避免空檔
    d = np.load(OUT_PATH, allow_pickle=True)
    print("\n[verify saved npz]")
    print("keys:", list(d.keys()))
    for k in d.keys():
        arr = d[k]
        print(f" - {k:12s} shape={arr.shape} dtype={arr.dtype} size={arr.size}")


# ===================== 執行 =====================
if __name__ == "__main__":
    quick_check(DATA_DIR)
    process_data()
