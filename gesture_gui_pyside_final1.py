import sys
from PySide2.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QLabel, QProgressBar,
    QHBoxLayout, QSpacerItem, QSizePolicy
)
from PySide2.QtCore import Qt, QTimer


class GestureGUI(QWidget):
    """
    PySide2 手勢辨識 GUI：
    - 顯示 Background + 0~9 共 11 類的機率條狀圖
    - 突顯當前辨識結果
    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Gesture Recognition (Background + 0~9)")
        self.resize(900, 450)

        # === 主要 Layout ===
        main_layout = QVBoxLayout()

        # === 當前手勢標籤 ===
        self.current_gesture_label = QLabel("Current gesture: Background")
        self.current_gesture_label.setAlignment(Qt.AlignCenter)
        self.current_gesture_label.setStyleSheet(
            "font-size: 20px; font-weight: bold; padding: 10px; "
            "background-color: lightgray; border-radius: 5px;"
        )
        main_layout.addWidget(self.current_gesture_label)

        # === 直條圖區域 ===
        self.hbox = QHBoxLayout()

        # 手勢名稱：Background + "0"~"9"
        self.gesture_names = ["Background"] + [str(i) for i in range(10)]
        self.bars = {}   # name -> QProgressBar

        # 直條圖寬度
        self.BAR_WIDTH = 20

        # 顏色設定：可以自己改喜歡的
        self.bar_colors = {
            "Background": "#7f8c8d",
            "0": "#1abc9c",
            "1": "#3498db",
            "2": "#9b59b6",
            "3": "#e67e22",
            "4": "#e74c3c",
            "5": "#2ecc71",
            "6": "#f1c40f",
            "7": "#16a085",
            "8": "#2980b9",
            "9": "#8e44ad",
        }
        self.gesture_colors = {
            "Background": "lightgray",
            "0": "#d1f2eb",
            "1": "#d6eaf8",
            "2": "#f5eef8",
            "3": "#fae5d3",
            "4": "#f9e5e5",
            "5": "#d5f5e3",
            "6": "#fcf3cf",
            "7": "#d0ece7",
            "8": "#d4e6f1",
            "9": "#e8daef",
        }

        # 兩側 Spacer 讓直條圖置中
        self.hbox.addSpacerItem(
            QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)
        )

        # 為每個手勢建立一組「直條 + 標籤」
        for name in self.gesture_names:
            v_layout = QVBoxLayout()

            bar = QProgressBar()
            bar.setOrientation(Qt.Vertical)
            bar.setRange(0, 100)
            bar.setValue(0)
            bar.setTextVisible(False)
            color = self.bar_colors.get(name, "#7f8c8d")
            bar.setStyleSheet(f"QProgressBar::chunk {{ background-color: {color}; }}")
            bar.setFixedWidth(self.BAR_WIDTH)

            v_layout.addWidget(bar, alignment=Qt.AlignBottom)

            label = QLabel(name)
            label.setAlignment(Qt.AlignCenter)
            v_layout.addWidget(label, alignment=Qt.AlignCenter)

            self.hbox.addLayout(v_layout)
            self.hbox.addSpacerItem(
                QSpacerItem(10, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)
            )

            self.bars[name] = bar

        self.hbox.addSpacerItem(
            QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)
        )

        main_layout.addLayout(self.hbox)
        self.setLayout(main_layout)

    def update_probabilities(self, probs, current_gesture: str):
        """
        更新 11 個手勢的機率條狀圖與辨識結果。

        :param probs: 1D array-like, 長度應該等於 len(self.gesture_names)
                      對應順序需與 self.gesture_names 一致
                      [Background, "0","1",...,"9"]
        :param current_gesture: str, 當前辨識出的手勢名稱
        """
        if probs is None:
            return

        # 避免長度不符時直接爆掉
        n = min(len(probs), len(self.gesture_names))
        for i in range(n):
            name = self.gesture_names[i]
            p = float(probs[i])
            p = max(0.0, min(1.0, p))  # clamp 到 [0,1]
            self.bars[name].setValue(int(p * 100))

        # 更新中央標籤
        if current_gesture not in self.gesture_names:
            current_gesture = "Background"
        bg_color = self.gesture_colors.get(current_gesture, "lightgray")

        self.current_gesture_label.setText(f"Current gesture: {current_gesture}")
        self.current_gesture_label.setStyleSheet(
            f"font-size: 20px; font-weight: bold; padding: 10px; "
            f"background-color: {bg_color}; border-radius: 5px;"
        )


# 測試用：單獨跑這個檔案時可以看到條狀圖效果
if __name__ == "__main__":
    import numpy as np

    app = QApplication(sys.argv)
    window = GestureGUI()
    window.show()

    def simulate_data():
        # 隨機產生一個機率分佈
        probs = np.random.rand(len(window.gesture_names))
        probs = probs / probs.sum()

        # 取非背景中最大那個當手勢
        nonbg = probs[1:]
        idx = int(nonbg.argmax())
        current = window.gesture_names[idx + 1] if nonbg[idx] > 0.4 else "Background"

        window.update_probabilities(probs, current)

    timer = QTimer()
    timer.timeout.connect(simulate_data)
    timer.start(500)

    sys.exit(app.exec_())
