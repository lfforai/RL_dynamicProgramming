import tensorflow as tf
import numpy as np
import random
import gym
from collections import deque

class DDPG:
    def __init__(self,filepath="/root/model2/",**kwargs):
        self.filepath=filepath
        # self.env=gym.make("MountainCarContinuous-v0")
        self.memory=deque([])
        self.memory_ALD=[]
        self.K_Matrix=np.matrix(np.zeros((1,1)))
        self.K_Matrix_inv=np.matrix(np.zeros((1,1)))
        self.k_t_1_o=0
        self.ct=0
        self.r=0.9
        self.var=3.0
        self.var_increase=0.95
        self.var_mini=0.001
        self.max_memory_len=5000
        self.increas_memory_len=200
        self.deincreas_memory_len=200
        self.N=100   #迭代次数
        self.env=gym.make('CartPole-v0') #
        self.action_num=2
        self.space_num=4
        self.var=0.4
        self.f_value=0.001
        self.At=0
        self.bt=0
        self.arg=0
        self.greed_max=1.0
        self.greed_min=0.05
        self.greed_des=0.95
        self.greed=self.greed_max

    def gym_info(self):
        print("action_space:",list(self.env.action_space))
        print("observation_space:",self.env.observation_space)

    def action_lr(self,state_now,p=0.1):
        if self.memory_ALD.__len__()==0:
           return random.choices([0,1],[0.5,0.5])[0]
        else:
           list_left=[]
           left_as=self.s_fun(0,state_now)
           for e in self.memory_ALD:
               list_left.append(self.K_func(self.s_fun(e[3],e[0]),left_as))
           left_action_value=np.dot(np.array(list_left),self.arg)

           list_right=[]
           right_as=self.s_fun(1,state_now)
           for e in self.memory_ALD:
               list_right.append(self.K_func(self.s_fun(e[3],e[0]),right_as))
           right_action_value=np.dot(np.array(list_right),self.arg)

           if random.random()>p:
             if right_action_value>left_action_value:
                return 1
             else:
                return 0
           else:
             return random.choices([0,1],[0.5,0.5])[0]

    def K_func(self,x,y):# gs
        result=np.exp(-1.0*np.dot(x-y,x-y)/(2.0*self.var))
        return result

    def s_fun(self,a,statenow):  #togther action and state [1. 2. 3. 4. 0. 0. 0. 0.]
        result=np.zeros(self.action_num*self.space_num)
        result[a*self.space_num:a*self.space_num+self.space_num]=statenow
        return np.array(result)

    def compute_K_Matrix(self):
        K_len=self.memory_ALD.__len__()
        self.K_Matrix=np.zeros((K_len,K_len))
        for i in range(K_len):
            for j in range(K_len):
                self.K_Matrix[i][j]=self.K_func(
                    self.s_fun(self.memory_ALD[i][3],self.memory_ALD[i][0]), \
                    self.s_fun(self.memory_ALD[j][3],self.memory_ALD[j][0]))
        # print(self.K_Matrix)
        self.K_Matrix=np.matrix(self.K_Matrix)

    def k_t_1(self,xt):
        self.k_t_1_o=np.zeros((len(self.memory_ALD),1),dtype=float)
        for i in range(len(self.memory_ALD)):
            self.k_t_1_o[i][0]=self.K_func(self.s_fun(self.memory_ALD[i][3],self.memory_ALD[i][0]),
                                           self.s_fun(xt[3],xt[0]))

    def if_in2ALD(self,xt): # 1.0=K(tt)
        if  1.0-tf.matmul(tf.transpose(self.k_t_1_o),self.ct)[0]>self.f_value:
            self.memory_ALD.append(xt)
        else:
            pass

    def ALD(self):
        self.memory_ALD=[]
        self.K_Matrix=np.matrix(np.zeros((1,1)))
        self.K_Matrix_inv=np.matrix(np.zeros((1,1)))
        for e in self.memory:
            if self.memory_ALD.__len__()==0:
               self.memory_ALD.append(e)
            elif self.memory_ALD.__len__()==1:
               self.memory_ALD.append(e)
               self.compute_K_Matrix()
            else:
               if self.K_Matrix_inv.shape[0]<self.memory_ALD.__len__():
                  self.compute_K_Matrix()
                  self.K_Matrix_inv=tf.linalg.inv(self.K_Matrix).numpy()
                  self.k_t_1(e)
                  self.ct=tf.matmul(self.K_Matrix_inv,self.k_t_1_o)
                  self.if_in2ALD(e)
               else:
                  self.k_t_1(e)
                  self.ct=tf.matmul(self.K_Matrix_inv,self.k_t_1_o)
                  self.if_in2ALD(e)

    def sample_show(self,num):
        index=0
        sum_reward=0
        for c in range(100000):
            state_now = self.env.reset()
            while True:
                self.env.render()
                action_now=self.action_lr(state_now,0)
                state_next,reward,done,info = self.env.step(action_now)
                x, x_dot, theta, theta_dot = state_next
                r1 = (0.5 - abs(x))/self.env.x_threshold - 0.8
                r2 = (self.env.theta_threshold_radians - abs(theta))/self.env.theta_threshold_radians - 0.5
                reward = r1 + r2
                if done:
                    index=index+1
                    sum_reward=sum_reward+reward
                    if index>num:
                        print("show over,reward:",sum_reward)
                        return num
                    break
                else:
                    index=index+1
                    sum_reward=sum_reward+reward
                    if index>num:
                        print("show over,reward:",sum_reward)
                        return num
                    state_now=state_next

    def sample_lr(self,num):
        num_increas=0
        for c in range(100000):
            if len(self.memory)>self.max_memory_len:
               for i in range(self.deincreas_memory_len):
                   self.memory.pop()
            action_next=-1
            state_now = self.env.reset()
            while True:
                self.env.render()
                action_now=self.action_lr(state_now,self.greed)
                state_next,reward,done,info = self.env.step(action_now)
                x, x_dot, theta, theta_dot = state_next
                r1 = (0.5 - abs(x))/self.env.x_threshold - 0.8
                r2 = (self.env.theta_threshold_radians - abs(theta))/self.env.theta_threshold_radians - 0.5
                reward = r1 + r2
                if done:
                    num_increas= num_increas+1
                    if self.memory.__len__()>0:
                        temp=self.memory[-1]
                        if temp[5]!=-1:
                            self.memory.pop()
                            self.memory.append((temp[0],temp[1],temp[2],temp[3],action_now,temp[5]))
                    self.memory.append((state_now,state_next,reward,action_now,action_next,-1)) #end no action_next
                    if num_increas>num:
                       if len(self.memory)>self.max_memory_len:
                           for i in range(self.deincreas_memory_len):
                               self.memory.pop()
                       return num
                    break
                else:
                    num_increas= num_increas+1
                    if self.memory.__len__()>0:
                       temp=self.memory[-1]
                       if temp[5]!=-1:
                          self.memory.pop()
                          self.memory.append((temp[0],temp[1],temp[2],temp[3],action_now,temp[5]))
                    self.memory.append((state_now,state_next,reward,action_now,action_next,1))
                    if num_increas>num:
                       self.memory.pop()
                       if len(self.memory)>self.max_memory_len:
                          for i in range(self.deincreas_memory_len):
                                self.memory.pop()
                       return num
                    state_now=state_next

    def ks(self,xt):
        result=np.zeros((len(self.memory_ALD),1),dtype=float)
        for i in range(len(self.memory_ALD)):
            result[i][0]=self.K_func(self.s_fun(self.memory_ALD[i][3],self.memory_ALD[i][0]),xt)
        return result

    def At_bt(self):
        N=self.memory_ALD.__len__()
        self.At=np.zeros((N,N))
        self.bt=np.zeros((N,1))
        for  e in  self.memory:
             if e[5]!=-1:
                 temp=self.ks(self.s_fun(e[3],e[0]))
                 temp2=self.ks(self.s_fun(e[4],e[1]))
                 # (state_now,state_next,reward,action_now,action_next,1)
                 self.At=self.At+tf.matmul(temp,np.transpose(temp)-self.r*np.transpose(temp2))
                 self.bt=self.bt+e[2]*temp
             else:
                 temp=self.ks(self.s_fun(e[3],e[0]))
                 self.At=self.At+tf.matmul(temp,np.transpose(temp))
                 self.bt=self.bt+e[2]*temp
        self.arg=tf.reshape(tf.matmul(tf.linalg.inv(self.At),self.bt),(-1,))

    def renew_ADL(self):
        if self.memory.__len__()>0:
            temp=deque([])
            for  e in self.memory:
                 if e[5]!=-1:
                    temp.append([e[0],e[1],e[2],e[3],self.action_lr(e[1]),e[5]])
                 else:
                    temp.append(e)
            self.memory=temp.copy()

    def train(self,N):
        for i in range(N):
            if i%5==0:
                print("测试开始,贪婪度.样本数量",self.greed,self.memory.__len__())
                self.sample_show(1000)
            if i%2==0 and i !=0: #基于贪婪
               if self.greed*self.greed_des>self.greed_min:
                  self.greed=self.greed*self.greed_des
               else:
                  self.greed=self.greed_min
            self.sample_lr(self.increas_memory_len)
            print("ADL_len:",self.memory_ALD.__len__())
            self.ALD()
            self.At_bt()
            # self.renew_ADL()#也可以不改动旧样本动作

o=DDPG()
# print(o.memory.__len__())
# for e in o.memory:
#     if e[4]==-1 and e[5]!=-1:
#        print(e)
o.train(2000)
exit()