from demo import *
import physbo  # need download VS build tool
import scipy
import itertools
import matplotlib.pyplot as plt
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import tensorflow as tf
model.load_weights('C:/Users/B20466/Desktop/梓堂/Project/材料AI/Thermoplastic Elastomer inverse design/polymer2.hdf5')  #polymer.hdf5路徑
#Design target
#target = np.linspace(0,2.0,101)
x_axis = np.linspace(0, 3, 101)

def settarget(x1,x2,y1,y2,y3):
  line1 = np.linspace(0,y1,x1)
  line2 = np.linspace(y1,y2,x2-x1) 
  line3 = np.linspace(y2,y3,101-x2) 
  combined_line = np.concatenate((line1, line2, line3))
  return combined_line
  
  
target = settarget(2,70,0.4,0.8,1.2)
# 繪製曲線
x_label = "Strain"
y_label = "Stress[εσ^-3]"
plt.plot(x_axis, target, label="Target Curve", color = 'blue')  # 添加曲線標籤

plt.xlabel(x_label)  # 設定 X 軸標籤
plt.ylabel(y_label)  # 設定 Y 軸標籤
plt.title("SS Curve")  # 設定圖表標題
plt.legend()  # 顯示圖例
plt.grid(True)  # 顯示網格

# 顯示圖表
plt.show()


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
print('c:',type(candidate))
print('x:',type(X))


###############

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
real = model.predict(best_X).reshape(101,)

# 繪製曲線
x_label = "Strain"
y_label = "Stress[εσ^-3]"
plt.plot(x_axis, target, label="Target Curve", color = 'blue')  # 添加曲線標籤
plt.plot(x_axis, real, label="Real Curve(N=125.0,f_A=0.54,alpha=0.31)", color = 'red')  # 添加曲線標籤

plt.xlabel(x_label)  # 設定 X 軸標籤
plt.ylabel(y_label)  # 設定 Y 軸標籤
plt.title("SS Curve")  # 設定圖表標題
plt.legend()  # 顯示圖例
plt.grid(True)  # 顯示網格

# 顯示圖表
plt.show()