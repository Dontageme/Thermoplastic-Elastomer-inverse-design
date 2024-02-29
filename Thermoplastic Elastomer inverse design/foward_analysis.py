from demo import *
import physbo  # need download VS build tool
import scipy
import itertools
import matplotlib.pyplot as plt
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import tensorflow as tf
model.load_weights('C:/Users/B20466/Desktop/梓堂/Project/材料AI/Thermoplastic Elastomer inverse design/polymer2.hdf5')  #polymer.hdf5 路徑

#Design target
#target = np.linspace(0,2.0,101)
x_axis = np.linspace(0, 3, 101)

def curve_analysis(N,fA,a):
    
    x = np.array([N,fA,a])
    curve = model.predict(x).reshape(101,)
    return curve


N=50
fA=0.59
a=0.12


target = curve_analysis(48, 0.59, 0.12)
#print(target.shape)
label = f"Curve Label: N={N}, fA={fA}, a={a}"


x_label = "Strain"
y_label = "Stress[εσ^-3]"
plt.plot(x_axis, target, label=label)  # 添加曲線標籤
plt.xlabel(x_label)  # 設定 X 軸標籤
plt.ylabel(y_label)  # 設定 Y 軸標籤
plt.title("SS Curve")  # 設定圖表標題
plt.legend()  # 顯示圖例
plt.grid(True)  # 顯示網格

# 顯示圖表
plt.show()
