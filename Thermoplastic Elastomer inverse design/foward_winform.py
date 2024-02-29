import argparse
from demo import *
import physbo
import scipy
import itertools
import matplotlib.pyplot as plt
import os
import tensorflow as tf
import csv

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# 加载模型
#model.load_weights('C:/Users/B20466/Desktop/梓堂/Project/材料AI/Thermoplastic Elastomer inverse design/polymer2.hdf5')

# 获取脚本文件所在目录
script_dir = os.path.dirname(__file__)
# 构建相对路径
relative_path = 'polymer2.hdf5'
# 完整文件路径
hdf5_path = os.path.join(script_dir, relative_path)
# 加载模型
model.load_weights(hdf5_path)

def curve_analysis(N, fA, a):
    x = np.array([N, fA, a])
    curve = model.predict(x).reshape(101,)
    return curve

def save_curve_plot(N, fA, a, save_dir):
    target = curve_analysis(N, fA, a)
    label = f"Curve Label: N={N}, fA={fA}, a={a}"

    x_axis = np.linspace(0, 3, 101)
    x_label = "Strain"
    y_label = "Stress[εσ^-3]"

    plt.plot(x_axis, target, label=label)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title("SS Curve")
    plt.legend()
    plt.grid(True)

    # 生成文件名
    filename = f"curve_N{N}_fA{fA}_a{a}.png"
    #圖片存放路徑
    save_path = "forward_pic"
    # 合并路径
    save_path = os.path.join("forward_pic", filename)
    
    # 保存图表到指定路径
    plt.savefig(save_path)
    print(f"Saved curve plot to {save_path}")
    
    # 生成 CSV 檔案名稱
    csv_filename = f"curve_N{N}_fA{fA}_a{a}.csv"
    csv_save_path = os.path.join("forward_csv", csv_filename)
    # 保存 CSV
    with open(csv_save_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['X', 'Y'])  # 寫入標題
        for x, y in zip(x_axis, target):
            writer.writerow([x, y])  # 寫入數據

    print(f"Saved curve data to {csv_save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--N', type=int, default=48, help='Value for N')
    parser.add_argument('--fA', type=float, default=0.59, help='Value for fA')
    parser.add_argument('--a', type=float, default=0.12, help='Value for a')
    parser.add_argument('--save_dir', type=str, default='forward_pic', help='Directory to save the plots')
    parser.add_argument('--save_dir_csv', type=str, default='forward_csv', help='Directory to save the csv')
    args = parser.parse_args()

    save_curve_plot(args.N, args.fA, args.a, args.save_dir)
