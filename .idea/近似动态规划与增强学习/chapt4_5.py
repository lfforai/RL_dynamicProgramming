import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras
import numpy as np
import random
import gym
from collections import deque

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

class Linear(layers.Layer):
    def __init__(self, name="Linear",units=32,training=True,**kwargs):
        super(Linear, self).__init__(name=name,**kwargs)
        self.units = units
        self.training=training

    def build(self, input_shape):#只有在运行完call以后才自动运行
        self.w = self.add_weight(name="w",shape=(input_shape[-1], self.units),
                                 initializer='random_normal',
                                 trainable=self.training)

        self.b = self.add_weight(name="b",shape=(self.units,),
                                 initializer='random_normal',
                                 trainable=self.training)
        # super(Linear, self).build(input_shape)

    def call(self, inputs):#call 被写入 __call__调用
        return tf.matmul(inputs, self.w) + self.b

    def get_config(self):
        config = super(Linear, self).get_config()
        config.update({'units': self.units})
        return config

class Q_st_at(layers.Layer):
    def __init__(self,name="action1",training=True,**kwargs):
        super(Q_st_at, self).__init__(name=name,**kwargs)
        self.training=training
        self.action_1 = Linear(name="Q_st_at"+name,units=1225,training=training)
        self.action_11 = Linear(name="Q_st_at"+name,units=1225,training=training)
        self.action_111 = Linear(name="Q_st_at"+name,units=1225,training=training)
        self.action_1111 = Linear(name="Q_st_at"+name,units=1,training=training)

    def call(self, state):
        x=self.action_1(state)
        x=tf.nn.relu(x)
        x=self.action_11(x)
        x=tf.nn.relu(x)
        x=self.action_111(x)
        x=tf.nn.sigmoid(x)
        x=self.action_1111(x)
        return x

    def get_config(self):
        config = super(Q_st_at, self).get_config()
        return config

class P_s_a(layers.Layer):
    def __init__(self,Q1=Q_st_at(name="action1"),Q2=Q_st_at(name="action2"),\
                 T=1.0,name="critic",**kwargs):
        super(P_s_a, self).__init__(name=name,**kwargs)
        self.T=T
        self.action_1 = Q1
        self.action_2 = Q2

    def call(self, state):
        x1=self.action_1(state)
        x2=self.action_2(state)
        x=tf.concat([x1,x2],-1)
        x=x/self.T
        x=tf.nn.softmax(x)
        return x

    def get_config(self):
        config = super(P_s_a, self).get_config()
        return config


from tensorflow.keras import backend as K
def RGNP_loss(y_true, y_pred):
    return  tf.reduce_mean(tf.pow(y_pred,2))/2

class RGNP_Net(tf.keras.Model):
    def __init__(self,T=0.05,**kwargs):
        super(RGNP_Net, self).__init__(**kwargs)
        self.r=0.80
        self.memory=deque([])
        self.max_memory_len=5000
        self.increas_memory_len=250
        self.deincreas_memory_len=150
        self.env=gym.make('CartPole-v0') #
        self.Q_left=Q_st_at(name="left")
        self.Q_right=Q_st_at(name="right")
        self.P_s_a_net=P_s_a(self.Q_left,self.Q_right,T)
        self.greed_max=0.9
        self.greed_min=0.05
        self.greed_des=0.85
        self.greed=self.greed_max

    def action_lr(self,state_now,p=0.1):
        # if random.random()>p:
        #     Q_now_left=self.Q_left(tf.convert_to_tensor([state_now]))[0][0]
        #     Q_now_right=self.Q_right(tf.convert_to_tensor([state_now]))[0][0]
            [left,right]=self.P_s_a_net(tf.convert_to_tensor([state_now]))[0]
            # print(left,Q_now_left,right,Q_now_right)
            left=1.0-right
            result=random.choices([0,1],[left,right])[0]
            return result
        # else:
        #     return random.choices([0,1],[0.5,0.5])[0]

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

    def call(self,input):
        state_now=input[0]
        action_now=input[1]#(0,1)格式，表示当状态行动是左或者右
        state_next=input[2]
        action_next=input[3]#(0,1)格式，表示下状态行动是左或者右
        reward=input[4]
        r_o=input[5]

        #计算next步的跳转概率
        p_s_a_next=self.P_s_a_net(state_next)
        p_s_a_next=tf.reduce_sum(tf.multiply(p_s_a_next,action_next),-1)

        #计算Q（st,at）,Q（st+1，at+1）
        Q_next_left=self.Q_left(state_next)
        Q_next_right =self.Q_right(state_next)
        Q_now_left=self.Q_left(state_now)
        Q_now_right=self.Q_right(state_now)

        Q_next=tf.reduce_sum(tf.multiply(tf.concat([Q_next_left,Q_next_right],-1),action_next),-1)
        Q_now=tf.reduce_sum(tf.multiply(tf.concat([Q_now_left,Q_now_right],-1),action_now),-1)

        #计算差分
        difference =tf.multiply((reward+tf.multiply(r_o,Q_next)-Q_now),p_s_a_next)
        return difference

    def train(self,x_train,y_ture,size):
        self.size=size
        optimizer = tf.keras.optimizers.Adagrad()
        # optimizer = tf.keras.optimizers.RMSprop()
        # optimizer = tf.keras.optimizers.Adam(learning_rate=1.0e-3)
        self.compile(optimizer, loss=RGNP_loss)
        self.fit(x_train, y_ture, epochs=3,batch_size=size)

    def train_o(self):
        for i in range(100000):
            self.sample_lr(self.increas_memory_len)
            temp_memory=list(self.memory).copy()#进行采样
            random.shuffle(temp_memory)
            temp_memory=temp_memory[:int(temp_memory.__len__()*1.0/2.0)]

            #准备训练样本
            x_state_now=[]
            x_state_action_now=[]
            x_state_next=[]
            x_state_action_next=[]
            x_reward=[]
            x_r=[]
            for e  in temp_memory:
                x_state_now.append(list(e[0]))
                if e[3]==0:#left
                   x_state_action_now.append([1,0])
                else:
                   x_state_action_now.append([0,1])
                x_state_next.append(list(e[1]))
                if e[4]==0:#right
                    x_state_action_next.append([1,0])
                else:
                    x_state_action_next.append([0,1])
                x_reward.append(e[2])
                if e[5]!=-1:
                   x_r.append(self.r)
                else:
                   x_r.append(0.0)
            x_state_now=tf.convert_to_tensor(x_state_now,dtype=tf.float32)
            x_state_action_now=tf.convert_to_tensor(x_state_action_now,dtype=tf.float32)
            x_state_next=tf.convert_to_tensor(x_state_next,dtype=tf.float32)
            x_state_action_next=tf.convert_to_tensor(x_state_action_next,dtype=tf.float32)
            x_reward=tf.convert_to_tensor(x_reward,dtype=tf.float32)
            x_r=tf.convert_to_tensor(x_r,dtype=tf.float32)
            # print([x_state_now,x_state_action_now,x_state_next,x_state_action_next,x_reward,x_reward])
            x=[x_state_now,x_state_action_now,x_state_next,x_state_action_next,x_reward,x_r]
            y=tf.zeros(temp_memory.__len__())
            self.train(x,y,temp_memory.__len__())#tf.keras.Model可以接受的x只能是tensor或者list[tensor]

            if i%4==0:
               print("开始show，贪婪度",self.greed)
               self.sample_show(800)

            if i%2==0 and i !=0: #基于贪婪
                if self.greed*self.greed_des>self.greed_min:
                    self.greed=self.greed*self.greed_des
                else:
                    self.greed=self.greed_min

o=RGNP_Net()
o.train_o()