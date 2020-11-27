from __future__ import absolute_import, division, print_function, unicode_literals

from tensorflow.keras import layers
from tensorflow import keras
import tensorflow as tf
import gym
import numpy as np
import random
import os
import time

gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#     # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
#     try:
#         tf.config.experimental.set_virtual_device_configuration(
#             gpus[0],
#             [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=5120)])
#         logical_gpus = tf.config.experimental.list_logical_devices('GPU')
#         print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
#     except RuntimeError as e:
#         # Virtual devices must be set before GPUs have been initialized
#         print(e)
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
from tensorflow.keras import layers





#函数rare event example
#用指数分布CE交叉熵最短路径P（S（x）>r）概率计算
def  unite_exp(x=[],list_v=[]):#返回联合指数联合概率分布，不使用
     i=0
     sum1=0.0
     sum2=1.0
     for e in list_v:
        sum1+=x[i]/e
        sum2*=1/e
        i+=1
     return np.exp(-1.0*sum1)*sum2

def  min_road(x=[]):#最短距离函数
     min_result=x[0]+x[2]+x[4]
     if x[0]+[3]<min_result:
        min_result=x[0]+[3]
     if x[1]+x[4]<min_result:
        min_result=x[1]+x[4]
     if x[1]+x[2]+x[3]<min_result:
        min_result=x[1]+x[2]+x[3]
     return min_result

def exponent_sampel_unite(beta):#均值为beta的指数分布的样本，exp（-x/beta）*（1/beta）
    u = np.random.rand();
    x= -beta * np.log(u);
    return(x);

def W_rate(x=[],f=[],g=[]):#计算W=f（x）/g（x），指数分布fdf函数，使用
    sum1=0.0
    sum2=1.0
    i=0
    for e in f:
        sum1=sum1+(1.0/e-1.0/g[i])*x[i]
        sum2*=g[i]/e
        i+=1
    return np.exp(-1.0*sum1)*sum2

class rate_event():
       def __init__(self):
           self.beta=[0.25,0.4,0.1,0.3,0.2]
           self.r=2.0
           self.p=0.1
           self.N=5000

       def cout_num(self,value,rt):#I（S（x）>rt）
           if value>rt:
              return 1.0
           else:
              return 0.0

       def train(self):
           beta_iter_u=self.beta
           beta_iter_v=self.beta
           rt=0
           index=0
           while True:
               index=index+1
               S_min=[]#记录min（x1+x3+x5,x1+x4,x2+x5,x2+x3+x4）
               S_x=[]#记录（x1，x2，x3，x4，x5）
               for e in range(self.N):
                   x=[]
                   for i in range(5):
                       x.append(exponent_sampel_unite(beta_iter_v[i]))
                   S_x.append(x)#记录（x1，x2，x3，x4，x5）
                   S_min.append(min_road(x))#从样本计算最短路径min（x1+x3+x5,x1+x4,x2+x5,x2+x3+x4）
               # print(S_x)
               S=list(zip(S_x,S_min))
               def by_name(t):#安装S_min最短距离排序
                   return(t[1])
               S=sorted(S,key=by_name)
               # print(S)
               rt=S[int(self.N*(1.0-self.p))][1]#当前1-p分位数上的tr是否大于预定值
               print("rt",rt)
               if rt>self.r:
                  print("v:=",beta_iter_v)
                  break
               #更新参数beta_iter_v
               I_W=[]
               for e in S:#计算N个I_W的权重
                   #e[0]（x1，x2，x3，x4，x5）
                   #e[1] min（x1+x3+x5,x1+x4,x2+x5,x2+x3+x4）
                   I_W.append(self.cout_num(e[1],rt)*W_rate(e[0],beta_iter_u,beta_iter_v)) #min
               #W*X
               i=0
               sum_I_W=0
               temp_v=np.zeros(5)
               for e in S:
                   temp_v+=np.array(e[0],dtype=float)*I_W[i]
                   sum_I_W+=I_W[i]
                   i=i+1
               beta_iter_v=temp_v/sum_I_W
               print("v:=",beta_iter_v)

           #计算概率S(X)>r=2.0
           S_min=[]
           S_x=[]
           for e in range(100000):
               x=[]
               for i in range(5):
                   x.append(exponent_sampel_unite(beta_iter_v[i]))
               S_x.append(x)
               S_min.append(min_road(x))#计算最短距离
           i=0
           sum=0
           for e  in S_min:
               sum=sum+self.cout_num(e,self.r)*W_rate(S_x[i],beta_iter_u,beta_iter_v)
               i=i+1
           print("P(S(x)>r)",sum/S_min.__len__())
a=rate_event()
a.train()

#二、组合优化问题
# Combinatorial Optimization Example
#vetor，y=（x1,x2.....xn）=(1,1,1,1,1,0,0,0,0,0)
#一开始并不知道y哪些是1哪些是0，我们通过样本进行评估y的概率P
class combin_Opt():
       def __init__(self):
           self.y=[1,1,1,1,1,0,0,0,0,0]#真实的y
           self.num=self.y.__len__()
           self.N=50
           self.p=0.1
           self.init_pro=[0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5]

       def S(self,x):#需要测算的随机变量的函数
          i=0
          sum=0
          for e in self.y:
              sum+=np.abs(e-x[i])
              i=i+1
          return self.num-sum

       def cout_num(self,value,rt):
           if value>=rt:
               return 1.0
           else:
               return 0.0

       def train(self):
           print("v1:",self.init_pro)
           rt_old=0
           while True:
               y_list=[]
               for e in range(self.N):#收集样本
                   y_sample=[]
                   for i in range(self.num):#生成样本y
                       y_sample.append(np.random.choice(a=[0,1],p=[1.0-self.init_pro[i],self.init_pro[i]]))
                   y_list.append(y_sample)

               S_list=[]#计算阈值rt
               for e in y_list:
                   S_list.append(self.S(e))
               S_list_o=sorted(S_list)
               rt_new=S_list_o[int((1.0-self.p)*self.N)]
               print("rt_new:=",rt_new)

               if rt_new==rt_old:
                  break

               I_list=[]#计算阈值rt
               for e in S_list:
                   I_list.append(self.cout_num(e,rt_new))

               sum=0
               sum_up=0
               i=0
               for e in I_list:#修正参数
                   sum+=e
                   sum_up+=e*np.array(y_list[i],dtype=float)
                   i=i+1
               self.init_pro=list(sum_up/sum)
               rt_old=rt_new
               print("v2:",self.init_pro)
           print("v3:",self.init_pro)
a=combin_Opt()
a.train()