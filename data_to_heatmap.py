import numpy as np
import matplotlib.pyplot as plt
import h5py
import os

# 檔案路徑
file_path = r"C:\mmWave\mmWave\radar-gesture-recognition-chore-update-20250815\src\data\groupkfoldtrain\ruuu\background\Background_0073_2025_12_03_16_27_27.h5"

# 根據文件擴展名決定加載方式
if file_path.endswith('.npy'):
    done = np.load(file_path)
elif file_path.endswith('.h5'):
    with h5py.File(file_path, 'r') as f:
        # 讀取 DS1 數據集
        done = f['DS1'][:]
        #labels = f['LABEL'][:].flatten()  # 假設有 LABEL 數據集

        # 過濾 LABEL 為 0 的數據
        #for i, label in enumerate(labels):
        #    if label == 0:
        #        done[:, :, :, i] = 0  # 清空對應數據
else:
    raise ValueError("Unsupported file format. Please provide a .npy or .h5 file.")

# 初始化變數
channel_sum = done.shape[0]
hmap_size = channel_sum * 2  # 每個通道兩種軸向
hmap = np.zeros(hmap_size, dtype=object)

# 計算所有熱圖
i = 0
for channel in range(channel_sum):
    channel_data = done[channel, :, :, :]
    for axis_num in range(0, 2):
        heatmap_data = np.sum(channel_data, axis=axis_num)
        hmap[i] = heatmap_data
        i += 1

# 要顯示的三張圖：hmap[0], hmap[1], hmap[2]
ylabel_map = ['Velocity', 'Range', 'Phase']  # hmap[0], hmap[1], hmap[2] 對應的 Y label
titles = ['Heatmap 0 (ch1-axis0)', 'Heatmap 1 (ch1-axis1)', 'Heatmap 2 (ch2-axis0)']

plt.figure(figsize=(9, 6))

for idx in range(3):
    plt.subplot(3, 1, idx + 1)
    heatmap = hmap[idx].astype(np.float32)
    plt.imshow(heatmap, cmap='hot', aspect='auto')
    plt.colorbar()
    plt.title(titles[idx])
    plt.xlabel('Frame')
    plt.ylabel(ylabel_map[idx])

plt.tight_layout()
plt.show()
