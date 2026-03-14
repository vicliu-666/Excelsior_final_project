# =============================================
# realtime_infer_with_gui.py
# - 0~9 手勢：先顯示圖片 3 秒
# - 3 秒後關掉圖片並執行電腦指令
# - 做完指令後暫停辨識，按 Q 再開始下一輪
# =============================================

import os
import sys
import subprocess
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PySide2 import QtWidgets, QtCore, QtGui

# ======== 路徑/參數（可直接改） ========
MODEL_PATH   = r"C:\Users\user\Desktop\mmWave\KKT_Module_Example_20240820\radar-gesture-recognition-chore-update-20250815\src\3d_cnn_model_background_0_9.pth"
SETTING_FILE = r"C:\Users\user\Desktop\mmWave\KKT_Module_Example_20240820\radar-gesture-recognition-chore-update-20250815\TempParam\K60168-Test-00256-008-v0.0.8-20230717_60cm"

WINDOW_SIZE  = 55
CLASS_NAMES  = ["Background"] + [str(i) for i in range(10)]
ENTER_TH     = 0.50                 # 進入閥值
EXIT_TH      = 0.20                 # 退出閥值
STREAM_TYPE  = "feature_map"        # 或 "raw_data"

# ======== 手勢圖片路徑 ========
# 改成你實際放圖片的資料夾，檔名假設為 0.png ~ 9.png
IMAGE_DIR = r"C:\Users\user\Desktop\mmWave\KKT_Module_Example_20240820\radar-gesture-recognition-chore-update-20250815\src\img"

IMAGE_PATHS = {
    "0": os.path.join(IMAGE_DIR, "0.png"),  # 音量上升
    "1": os.path.join(IMAGE_DIR, "1.png"),  # 音量下降
    "2": os.path.join(IMAGE_DIR, "2.png"),  # 放大
    "3": os.path.join(IMAGE_DIR, "3.png"),  # 縮小
    "4": os.path.join(IMAGE_DIR, "4.png"),  # 關閉視窗
    "5": os.path.join(IMAGE_DIR, "5.png"),  # 返回桌面
    "6": os.path.join(IMAGE_DIR, "6.png"),  # 開啟 Chrome
    "7": os.path.join(IMAGE_DIR, "7.png"),  # 開啟 Spotify
    "8": os.path.join(IMAGE_DIR, "8.png"),  # 開啟 Word
    "9": os.path.join(IMAGE_DIR, "9.png"),  # 開啟檔案總管
}

# ======== 你的 GUI 元件 ========
from gesture_gui_pyside_final1 import GestureGUI

# ======== Kaiku / KKT imports ========
from KKT_Module import kgl
from KKT_Module.DataReceive.Core import Results
from KKT_Module.DataReceive.DataReceiver import MultiResult4168BReceiver
from KKT_Module.FiniteReceiverMachine import FRM
from KKT_Module.SettingProcess.SettingConfig import SettingConfigs
from KKT_Module.SettingProcess.SettingProccess import SettingProc
from KKT_Module.GuiUpdater.GuiUpdater import Updater


# ------------------------------------------------
#  手勢 → 電腦指令 mapping
#  需要 nircmd.exe 在 PATH 或同資料夾
# ------------------------------------------------
def execute_pc_command(gesture: str):
    try:
        # 0：音量上升
        if gesture == "0":
            for _ in range(5):
                subprocess.Popen("nircmd.exe changesysvolume 2000")

        # 1：音量下降
        elif gesture == "1":
            for _ in range(5):
                subprocess.Popen("nircmd.exe changesysvolume -2000")

        # 2：放大（Ctrl + +）
        elif gesture == "2":
            subprocess.Popen("nircmd.exe sendkeypress ctrl+add")

        # 3：縮小（Ctrl + -）
        elif gesture == "3":
            subprocess.Popen("nircmd.exe sendkeypress ctrl+subtract")

        # 4：關閉視窗（Alt + F4）
        elif gesture == "4":
            subprocess.Popen("nircmd.exe sendkeypress alt+f4")

        # 5：返回桌面（Win + D）
        elif gesture == "5":
            subprocess.Popen("nircmd.exe sendkeypress lwin+d")

        # 6：開啟 Chrome
        elif gesture == "6":
            subprocess.Popen('start chrome', shell=True)

        # 7：開啟 Spotify
        elif gesture == "7":
            subprocess.Popen('start spotify', shell=True)

        # 8：開啟 Word
        elif gesture == "8":
            subprocess.Popen('start winword', shell=True)

        # 9：開啟檔案總管
        elif gesture == "9":
            subprocess.Popen('explorer', shell=True)

    except Exception as e:
        print(f"[CMD ERROR] {e}")


# 顯示在 console 的指令說明
COMMAND_DESC = {
    "0": "音量上升",
    "1": "音量下降",
    "2": "放大（Ctrl+＋）",
    "3": "縮小（Ctrl+－）",
    "4": "關閉視窗（Alt+F4）",
    "5": "返回桌面（Win+D）",
    "6": "開啟 Chrome",
    "7": "開啟 Spotify",
    "8": "開啟 Word",
    "9": "開啟檔案總管",
}


# ---------- Kaiku helpers ----------
def connect_device():
    try:
        device = kgl.ksoclib.connectDevice()
        if device == 'Unknow':
            ret = QtWidgets.QMessageBox.warning(
                None, 'Unknown Device', 'Please reconnect device and try again',
                QtWidgets.QMessageBox.Ok | QtWidgets.QMessageBox.Cancel
            )
            if ret == QtWidgets.QMessageBox.Ok:
                connect_device()
    except Exception:
        ret = QtWidgets.QMessageBox.warning(
            None, 'Connection Failed', 'Please reconnect device and try again',
            QtWidgets.QMessageBox.Ok | QtWidgets.QMessageBox.Cancel
        )
        if ret == QtWidgets.QMessageBox.Ok:
            connect_device()


def run_setting_script(setting_name: str):
    ksp = SettingProc()
    cfg = SettingConfigs()
    cfg.Chip_ID = kgl.ksoclib.getChipID().split(' ')[0]
    cfg.Processes = [
        'Reset Device',
        'Gen Process Script',
        'Gen Param Dict', 'Get Gesture Dict',
        'Set Script',
        'Run SIC',
        'Phase Calibration',
        'Modulation On'
    ]
    cfg.setScriptDir(f'{setting_name}')
    ksp.startUp(cfg)


def set_properties(obj: object, **kwargs):
    print(f"==== Set properties in {obj.__class__.__name__} ====")
    for k, v in kwargs.items():
        if not hasattr(obj, k):
            print(f'Attribute "{k}" not in {obj.__class__.__name__}.')
            continue
        setattr(obj, k, v)
        print(f'Attribute "{k}", set "{v}"')


# ---------- 3D CNN ----------
class Gesture3DCNN(nn.Module):
    def __init__(self, num_classes=11):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv3d(2, 32, 3), nn.ReLU(), nn.MaxPool3d(2), nn.BatchNorm3d(32),
            nn.Conv3d(32, 64, 3), nn.ReLU(), nn.MaxPool3d(2), nn.BatchNorm3d(64),
            nn.Conv3d(64,128, 3), nn.ReLU(), nn.MaxPool3d(2), nn.BatchNorm3d(128),
        )
        self.pool = nn.AdaptiveAvgPool3d((1,1,1))
        self.classifier = nn.Sequential(
            nn.Linear(128,128), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


def _maybe_remap_keys_to_classifier(state: dict) -> dict:
    if any(k.startswith("fc.") for k in state.keys()):
        new = {}
        for k, v in state.items():
            new["classifier." + k[3:]] = v if k.startswith("fc.") else v
        return new
    return state


# ---------- 推論核心 ----------
class OnlineInferenceContext:
    def __init__(self, model, device, window_size):
        self.model = model
        self.device = device
        self.window = window_size
        self.buffer = np.zeros((2, 32, 32, self.window), dtype=np.float32)
        self.collected = 0
        self.active = False
        self.last_pred = "Background"

    def reset_state(self):
        self.buffer[:] = 0
        self.collected = 0
        self.active = False
        self.last_pred = "Background"
        print("[CTX] reset_state")

    @staticmethod
    def to_frame(arr):
        x = np.asarray(arr)
        if x.shape == (2, 32, 32):
            pass
        elif x.shape == (32, 32, 2):
            x = np.transpose(x, (2, 0, 1))
        else:
            raise ValueError(f"Unexpected frame shape: {x.shape}")
        return x.astype(np.float32, copy=True)

    def push_and_infer(self, frame):
        self.buffer = np.roll(self.buffer, shift=-1, axis=-1)
        self.buffer[..., -1] = frame
        self.collected += 1
        if self.collected < self.window:
            return None

        win = np.expand_dims(self.buffer, axis=0)
        win = np.transpose(win, (0, 1, 4, 2, 3))
        x = torch.from_numpy(win).float().to(self.device)

        with torch.no_grad():
            logits = self.model(x)
            p = F.softmax(logits, dim=1).cpu().numpy()[0]
        return p

    def apply_double_threshold(self, probs):
        if probs is None or probs.ndim != 1:
            return "Background", False, probs

        nonbg_probs = probs[1:]
        if nonbg_probs.size == 0:
            return "Background", False, probs

        top_idx = int(nonbg_probs.argmax())
        top_prob = float(nonbg_probs[top_idx])
        top_name = CLASS_NAMES[top_idx + 1]

        if not self.active:
            if top_prob >= ENTER_TH:
                self.active = True
                current = top_name
            else:
                current = "Background"
        else:
            if np.all(nonbg_probs < EXIT_TH):
                self.active = False
                current = "Background"
            else:
                current = self.last_pred

        changed = (current != self.last_pred)
        self.last_pred = current
        return current, changed, probs


# ---------- Updater ----------
class InferenceUpdater(Updater):
    def __init__(self, ctx: OnlineInferenceContext, gesture_gui: GestureGUI, stream: str = "feature_map"):
        super().__init__()
        self.ctx = ctx
        self.gui = gesture_gui
        self.stream = stream

        self.paused = False          # True = 暫停辨識（等待 Q）
        self.pending_gesture = None  # 接收執行緒寫入、GUI 執行緒讀取
        self.processing = False      # GUI 是否正在處理當前手勢
        self.image_window = None     # 顯示手勢圖片用

    # --- 給 GUI thread 每 50 ms 呼叫 ---
    def gui_tick(self):
        if self.processing:
            return
        if self.pending_gesture is None:
            return

        gesture = self.pending_gesture
        self.pending_gesture = None
        self.processing = True

        print(f"[GUI] handling gesture {gesture}")
        self.show_gesture_image(gesture)

        # ★ 3 秒後執行指令（以及關掉圖片）
        QtCore.QTimer.singleShot(
            3000,
            lambda g=gesture: self._execute_after_delay(g)
        )

    def resume_recognition(self):
        print("[INFO] Q pressed → resume recognition (new round)")
        self.paused = False
        self.processing = False
        self.pending_gesture = None
        self.ctx.reset_state()

        # 把圖片關掉
        if self.image_window is not None:
            self.image_window.close()

    def show_gesture_image(self, gesture_label: str):
        path = IMAGE_PATHS.get(gesture_label)
        if not path or not os.path.exists(path):
            print(f"[WARN] image not found for gesture {gesture_label}: {path}")
            return

        if self.image_window is None:
            self.image_window = QtWidgets.QLabel()
            self.image_window.setWindowFlags(
                QtCore.Qt.Window |
                QtCore.Qt.WindowStaysOnTopHint |
                QtCore.Qt.Tool
            )
            self.image_window.setAlignment(QtCore.Qt.AlignCenter)

        pixmap = QtGui.QPixmap(path)
        if pixmap.isNull():
            print(f"[WARN] failed to load image: {path}")
            return

        # ★ 縮小圖片：寬度 600 px，等比例
        target_width = 600
        pixmap = pixmap.scaled(
            target_width,
            target_width,
            QtCore.Qt.KeepAspectRatio,
            QtCore.Qt.SmoothTransformation
        )

        self.image_window.setPixmap(pixmap)
        self.image_window.resize(pixmap.width(), pixmap.height())
        self.image_window.setWindowTitle(f"手勢 {gesture_label}")
        self.image_window.show()

    def _execute_after_delay(self, gesture_label: str):
        # 先關掉圖片
        if self.image_window is not None:
            self.image_window.close()

        # 顯示要做的指令（console）
        desc = COMMAND_DESC.get(gesture_label, "無對應指令")
        print(f"[ACTION] 手勢 {gesture_label}: {desc}")

        # 執行實際電腦指令
        execute_pc_command(gesture_label)
        # 保持 paused=True / processing=True，等你按 Q 再 resume

    def update(self, res: Results):
        try:
            if self.paused:
                return

            arr = res['raw_data'].data if self.stream == "raw_data" else res['feature_map'].data
            frame = self.ctx.to_frame(arr)
            probs = self.ctx.push_and_infer(frame)
            if probs is None:
                return

            current, changed, probs_out = self.ctx.apply_double_threshold(probs)

            # 更新 GUI 機率條
            try:
                self.gui.update_probabilities(probs_out, current)
            except Exception:
                pass

            if changed:
                print(f"[PRED] current={current}")
                if current != "Background" and not self.processing:
                    # 只在接收執行緒記錄手勢，不做 GUI 動作
                    self.pending_gesture = current
                    self.paused = True   # 暫停後續辨識

        except Exception as e:
            print("[ERROR] frame skipped:", e)
            pass


# ---------- Key event filter：偵測 Q ----------
class GestureKeyFilter(QtCore.QObject):
    def __init__(self, updater: InferenceUpdater):
        super().__init__()
        self.updater = updater

    def eventFilter(self, obj, event):
        if event.type() == QtCore.QEvent.KeyPress:
            if event.key() == QtCore.Qt.Key_Q:
                self.updater.resume_recognition()
                return True
        return False


# ---------- 主流程 ----------
def main():
    app = QtWidgets.QApplication(sys.argv)

    # 1) 啟動 GUI
    gui = GestureGUI()
    gui.show()

    # 2) 初始化雷達
    kgl.setLib()
    connect_device()
    run_setting_script(SETTING_FILE)

    if STREAM_TYPE == "raw_data":
        kgl.ksoclib.writeReg(0, 0x50000504, 5, 5, 0)
    else:
        kgl.ksoclib.writeReg(1, 0x50000504, 5, 5, 0)

    # 3) 模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Gesture3DCNN(num_classes=len(CLASS_NAMES)).to(device)

    state = torch.load(MODEL_PATH, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    state = _maybe_remap_keys_to_classifier(state)
    model.load_state_dict(state, strict=False)
    model.eval()
    print(f"[INFO] model loaded: {MODEL_PATH}  | device: {device}")
    print(f"[INFO] classes: {CLASS_NAMES}")

    # 4) 上線推論
    ctx = OnlineInferenceContext(model=model, device=device, window_size=WINDOW_SIZE)
    updater = InferenceUpdater(ctx, gesture_gui=gui, stream=STREAM_TYPE)

    # 5) Q 鍵 event filter
    key_filter = GestureKeyFilter(updater)
    gui.installEventFilter(key_filter)
    gui.key_filter = key_filter  # 防止被 GC

    # 6) GUI thread 的 timer：定期處理 pending_gesture
    gui_timer = QtCore.QTimer()
    gui_timer.timeout.connect(updater.gui_tick)
    gui_timer.start(50)  # 20 fps 檢查一次
    updater.gui_timer = gui_timer  # 防止被 GC

    # 7) Receiver + FRM
    receiver = MultiResult4168BReceiver()
    set_properties(receiver,
                   actions=1,
                   rbank_ch_enable=7,
                   read_interrupt=0,
                   clear_interrupt=0)
    FRM.setReceiver(receiver)
    FRM.setUpdater(updater)
    FRM.trigger()
    FRM.start()

    print("[INFO] Online inference with GUI started.")
    print("[INFO] 流程：辨識 → 顯示圖片 3 秒 → 關掉圖片 + 執行指令 → 暫停；按 Q 再開始下一輪。")

    try:
        sys.exit(app.exec_())
    except KeyboardInterrupt:
        pass
    finally:
        try:
            FRM.stop()
        except Exception:
            pass
        try:
            kgl.ksoclib.closeDevice()
        except Exception:
            pass
        print("[INFO] Stopped.")


if __name__ == "__main__":
    main()
