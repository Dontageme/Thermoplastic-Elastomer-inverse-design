import argparse
import pandas as pd
from demo import *
import physbo  # need download VS build tool
import scipy
import itertools
import matplotlib.pyplot as plt
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import tensorflow as tf
import msvcrt
import sys
import csv

target = None  # 在全局范围内初始化 target 变量

# 获取脚本文件所在目录
script_dir = os.path.dirname(__file__)
# 构建相对路径
relative_path = 'polymer2.hdf5'
# 完整文件路径
hdf5_path = os.path.join(script_dir, relative_path)
model.load_weights(hdf5_path)  #polymer.hdf5路徑
#Design target
#target = np.linspace(0,2.0,101)
x_axis = np.linspace(0, 3, 101)

# 繪製曲線
x_label = "Strain"
y_label = "Stress[εσ^-3]"

###############

#Design space
X_0=np.linspace(40.0,140.0,101) #40~140抓101個data
X_1=np.linspace(0.1,0.6,51)
X_2=np.linspace(0.0,0.5,51)
candidate = []
for i in X_0:
  for j in X_1:
    for k in X_2:
      candidate.append(np.array([i,j,k]))  #約25w筆 input feature
X = np.array(candidate) #input 25w x 3
#print('c:',type(candidate))
#print('x:',type(X))



####fxn defind

def calculate_slope(x1, y1, x2, y2):
    return (y2 - y1) / (x2 - x1)

def load_coordinates_from_csv(csv_path):
    df = pd.read_csv(csv_path)                      #ex: csv_file\Inverse_input.csv
    coordinates = list(zip(df['X'], df['Y']))
    return coordinates



def generate_linspace_lines(coordinates_str):
    # 解析坐标字符串并初始化 coordinates 列表
    coordinates = [(0, 0)]
    pairs = coordinates_str.split(';')
    for pair in pairs:
        x, y = map(float, pair.split(','))
        coordinates.append((x, y))
    
    if coordinates[-1][0] < 3:
        slope = calculate_slope(coordinates[-1][0], coordinates[-1][1], coordinates[-2][0], coordinates[-2][1])
        y_ = coordinates[-1][1] + slope * (3 - coordinates[-1][0])
        coordinates.append((3, y_))

    if len(coordinates) < 3:
        print("At least 2 coordinates are required.")
        return

    # 计算总的x轴距离
    total_x_distance = sum(coordinates[i + 1][0] - coordinates[i][0] for i in range(len(coordinates) - 1))
    # 计算每个插值点之间的x轴距离
    step = total_x_distance / 100

    # 初始化目标target
    plt.figure()
    target_x = []
    target_y = []

    # 绘制连线并插值
    x_current, y_current = coordinates[0]
    for i in range(len(coordinates) - 1):
        x_next, y_next = coordinates[i + 1]
        # 计算两个点之间的线段的长度
        segment_length = x_next - x_current

        # 计算需要插值的点数
        num_points = int(segment_length / step)
        # 使用linspace在两点之间生成插值点
        x_values = np.linspace(x_current, x_next, num_points)
        y_values = np.linspace(y_current, y_next, num_points)
        target_x.extend(x_values)
        target_y.extend(y_values)

        x_current, y_current = x_next, y_next

    # 确保target总共有101个点
    while len(target_x) < 101:
        target_x.append(target_x[-1])
    while len(target_y) < 101:
        target_y.append(target_y[-1])

    plt.plot(target_x, target_y)

    # 设置图表标题和坐标轴标签
    plt.xlabel("Strain")  # 设置 X 轴标签
    plt.ylabel("Stress[εσ^-3]")  # 设置 Y 轴标签
    plt.title("SS Curve")  # 设置图表标题

    # 显示图表
    plt.grid(True)
   
    # 生成文件名
    filename = f"preview.png"
    #圖片存放路徑
    save_dir = "preview_pic"
    # 合并路径
    save_path = os.path.join(save_dir, filename)
    
    # 保存图表到指定路径
    plt.savefig(save_path)
    print(f"Saved curve plot to {save_path}")    
    #print(len(target_y))

    return target_y

def main():
    global target
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=str, help='Path to CSV file with coordinates')
    parser.add_argument('--coordinates', type=str, help='Input coordinates as a single string (e.g., "0.1,0.3;1.0,0.5;2.0,0.7")')
    args = parser.parse_args()

    if args.csv:
        coordinates = load_coordinates_from_csv(args.csv)
        coordinates_str = ";".join([f"{x},{y}" for x, y in coordinates])

    elif args.coordinates:
        # 一次性输入的坐标格式为 "x1,y1;x2,y2;x3,y3;..."
        coordinates_str = args.coordinates  # 直接使用输入的坐标字符串



    else:
        print("No coordinates provided. Use either --csv or --coordinates.")
        return

    target = generate_linspace_lines(coordinates_str)

if __name__ == "__main__":
    main()


#define the simulator class and find the minimum value of a one-dimensional function using PHYSBO
class simulator:
 def __call__(self,action):
  action_idx = action[0]
  x = X[action_idx]
  #objective function
  fx = np.sum((model.predict(x)-target)**2)/101  #loss
  fx_list.append(fx) #loss for all input
  x_list.append(X[action_idx])
  
  return -fx

#set policy
policy = physbo.search.discrete.policy(test_X=X)

# set seed
policy.set_seed(0)

fx_list=[]
x_list=[]

#need random data to vuild gauss distribution
res=policy.random_search(max_num_probes=10, simulator=simulator())

res = policy.bayes_search(max_num_probes=3, simulator=simulator(), score='TS',interval=0, num_rand_basis=500)

best_fxs, best_actions = policy.history.export_sequence_best_fx()
best_fx = best_fxs[-1]
best_X = X[best_actions[-1],:]
print(f"best_fx: {best_fx} at N={best_X[0]},f_A={best_X[1]},alpha={best_X[2]}")


best_sol = [best_X[0],best_X[1], best_X[2]]
# 将最佳配方保存到 CSV 文件
# 定义保存的文件夹和文件名的前缀
save_dir = "suggest_parameter"
file_prefix = "suggest_parameter"

# 获取文件夹中已存在的 CSV 文件数量
existing_files = [f for f in os.listdir(save_dir) if f.startswith(file_prefix)]
next_file_number = len(existing_files) + 1

# 构建新的文件名
csv_file_path = os.path.join(save_dir, f"{file_prefix}_{next_file_number}.csv")

# 打开 CSV 文件并写入数据
with open(csv_file_path, mode='w', newline='') as csv_file:
    fieldnames = ['N', 'f_A', 'alpha']
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

    # 写入标题行
    writer.writeheader()

    # 写入最佳配方数据
    writer.writerow({'N': best_sol[0], 'f_A': best_sol[1], 'alpha': best_sol[2]})

print(f"Saved CSV to {csv_file_path}")


real = model.predict(best_X).reshape(101,)

# 繪製曲線
x_label = "Strain"
y_label = "Stress[εσ^-3]"
#plt.plot(x_axis, target, label="Target Curve", color = 'blue')  # 添加曲線標籤
plt.plot(x_axis, real, label="Real Curve", color = 'red')  # 添加曲線標籤

plt.xlabel(x_label)  # 設定 X 軸標籤
plt.ylabel(y_label)  # 設定 Y 軸標籤
plt.title("SS Curve")  # 設定圖表標題
plt.legend()  # 顯示圖例
plt.grid(True)  # 顯示網格

# 生成文件名
filename2 = f"overlap.png"
#圖片存放路徑
save_dir = "overlap_pic"
# 合并路径
save_path2 = os.path.join(save_dir, filename2)

# 保存图表到指定路径
plt.savefig(save_path2)
print(f"Saved curve plot to {save_path2}")