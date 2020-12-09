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

#not net
#构造神经网
class Linear(layers.Layer):
    def __init__(self, name="Linear",units=32,training=True,**kwargs):
        super(Linear, self).__init__(name=name,**kwargs)
        self.units = units
        self.training=training

    def build(self, input_shape):#只有在运行完call以后可以构造
        self.w = self.add_weight(name="w",shape=(input_shape[-1], self.units),
                                 initializer='random_normal',
                                 trainable=self.training)

        self.b = self.add_weight(name="b",shape=(self.units,),
                                 initializer='random_normal',
                                 trainable=self.training)
        # super(Linear, self).build(input_shape)

    @tf.function
    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b

    def get_config(self):
        config = super(Linear, self).get_config()
        config.update({'units': self.units})
        return config

class critic(tf.keras.Model):#计算V(S）
    def __init__(self, name="critic",training=True,**kwargs):
        super(critic, self).__init__(name=name,**kwargs)
        self.training=training
        self.block_1 = Linear(name="critic_linear1",units=32,training=training)
        self.block_2 = Linear(name="critic_linear2",units=32,training=training)
        self.block_3 = Linear(name="critic_linear3",units=1,training=training)

    @tf.function
    def call(self, inputs_state):
        x = self.block_1(inputs_state)
        x = tf.nn.relu(x)
        x = self.block_2(x)
        x = tf.nn.relu(x)
        x = self.block_3(x)
        return x

    def train(self,x_train,y_ture,size):
        optimizer = tf.keras.optimizers.Adam(learning_rate=2e-3)
        self.compile(optimizer, loss=tf.keras.losses.MeanSquaredError())
        self.fit(x_train, y_ture, epochs=5,batch_size=size,verbose=0)

    def get_config(self):
        config = super(critic, self).get_config()
        return config

class actor(tf.keras.Model):#计算Pi(at|st)
    def __init__(self, name="actor",training=True,**kwargs):
        super(actor, self).__init__(name=name,**kwargs)
        self.training=training
        self.block_1 = Linear(name="actor_linear1",units=32,training=training)
        self.block_2 = Linear(name="actor_linear2",units=32,training=training)
        self.block_3 = Linear(name="actor_linear1",units=2,training=training)

    @tf.function
    def call(self, inputs_state):
        x = self.block_1(inputs_state)
        x = tf.nn.relu(x)
        x = self.block_2(x)
        x=tf.nn.relu(x)
        x=self.block_3(x)
        x = tf.nn.softmax(x)
        return x

    def get_config(self):
        config = super(actor, self).get_config()
        return config

class natural_gradient():
     def __init__(self,filepath="/root/natural/"):
         self.pra_num=200
         self.w1_action_lr=np.random.normal(size=(self.pra_num+1)*2)
         self.filepath=filepath
         self.actor_net=actor()#计算actor=PI（at|st）
         self.critic_net=critic()#计算V（s）
         self.env=gym.make("CartPole-v1") #
         if not os.path.exists(self.filepath):
             print("存储参数")
             # os.makedirs(self.filepath)
             self.actor_net.build(input_shape=(None,self.env.observation_space.shape[0]))
             self.critic_net.build(input_shape=(None,self.env.observation_space.shape[0]))
         else:
             print("加载参数")
             self.actor_net.load_weights(self.filepath+"actor_1.0")
             self.critic_net.load_weights(self.filepath+"critic_1.0")

         if os.path.exists("./w1_action_lr.txt"):
            self.w1_action_lr=np.array(np.loadtxt("./w1_action_lr.txt"),dtype=float)

         if os.path.exists("./bf_function.txt"):
                import csv
                with open("./bf_function.txt",newline='',encoding='UTF-8') as csvfile:
                     rows=csv.reader(csvfile)
                     self.bf_function=np.array(list(rows),dtype=float)
                csvfile.close()
         else:
                temp_list=[]
                for g in range(int(self.pra_num)):
                    a2=np.random.random()*4.0-2.0 #平衡杆角速度
                    # a1=-0.25943951023931953+0.25943951023931953*2/(self.pra_num/2)*g
                    a1=-0.30943951023931953+np.random.random()*0.30943951023931953*2  #平衡杆偏离垂直的角度
                    b2=np.random.random()*3.0-1.5  #小车速度
                    # b1=-1.0+2.0/(self.pra_num/2)*g #小车位置
                    b1=-2.0+np.random.random()*2.0*2
                    temp_list.append([b1,b2,a1,a2])
                self.bf_function=np.array(temp_list,dtype=float)
                np.savetxt('./bf_function.txt',self.bf_function, delimiter=',')  # 数组x

         from collections import deque
         self.sample_critic=deque([])#有效利用历史样本V（s）
         self.sample_actor=[]#无法利用历史样本actor=PI（at|st）
         self.N_critic=500#buffer的样本数
         self.N_actor=500#buffer的样本数
         self.N_critic_remove=200#V（s）buffer每次更新的样本数量
         self.N_critic_batch=200
         self.N_actor_batch=200#梯度下降的batchsize
         self.r=0.90#折扣系数
         self.threshold=0.001#计算自然梯度的阙值 ,自适应梯度0.001，自然梯度0.00001
         self.optimizer = tf.keras.optimizers.SGD(learning_rate=1.0)#更新梯度优化器
         self.record_shape=[] #记录每个weight的shape 用于还原
         self.record_len=[]   #记录每个weight的长度  用于还原

     def chage_state(self,state,action):#基于高斯基的BF函数
         BFlist=[1.0]
         value=np.array(state,dtype=float)
         for e in list(self.bf_function):
             BFlist.append(np.exp(-np.dot((value-np.array(e,dtype=float)),(value-np.array(e,dtype=float)))/(2.0*0.25)))
         if action==0:
             result=np.array(list(BFlist)+[1.0]+list(np.zeros(self.pra_num,dtype=float)),dtype=float)
             return result
         else:
             result=np.array([1.0]+list(np.zeros(self.pra_num,dtype=float))+list(BFlist),dtype=float)
             return result

     def action(self,state):#行动函数，生成pi（at|st）
         state_left=self.chage_state(state,0)
         state_right=self.chage_state(state,1)
         left_pro  = tf.exp(tf.reduce_sum(state_left*self.w1_action_lr))/ \
                     (tf.exp(tf.reduce_sum(state_left*self.w1_action_lr))+tf.exp(tf.reduce_sum(state_right*self.w1_action_lr)))
         right_pro = tf.exp(tf.reduce_sum(state_right*self.w1_action_lr))/ \
                     (tf.exp(tf.reduce_sum(state_left*self.w1_action_lr))+tf.exp(tf.reduce_sum(state_right*self.w1_action_lr)))
         return np.random.choice(a=[0,1],p=[left_pro.numpy(),right_pro.numpy()])

     # def action(self,state):#[state],以Pi（at|st）选择
     #     state=np.array(state,dtype=float)
     #     left_pro,right_pro=self.actor_net(state)[0]
     #     # print(left_pro)
     #     # print(right_pro)
     #     left_pro=left_pro.numpy()
     #     right_pro=1.0-left_pro
     #     return np.random.choice(a=[0,1],p=[left_pro,right_pro])

     def partial_derivative_lr(self,state,action):#BF函数求偏导
         state_left=self.chage_state(state,0).reshape(-1)
         state_right=self.chage_state(state,1).reshape(-1)
         lr_value=tf.constant(self.w1_action_lr)
         if action==0:#left
             with tf.GradientTape() as t:
                 t.watch(lr_value)
                 z = tf.math.log(tf.exp(tf.reduce_sum(state_left*lr_value))/ \
                                 (tf.exp(tf.reduce_sum(state_left*lr_value))+tf.exp(tf.reduce_sum(state_right*lr_value))))
             dz_left = t.gradient(z,lr_value)
             del t
             return  dz_left.numpy()
         else:
             with tf.GradientTape() as t:
                 t.watch(lr_value)
                 z = tf.math.log(tf.exp(tf.reduce_sum(state_right*lr_value))/ \
                                 (tf.exp(tf.reduce_sum(state_left*lr_value))+tf.exp(tf.reduce_sum(state_right*lr_value))))
             dz_right = t.gradient(z,lr_value)
             del t
             return  dz_right.numpy()

     def coll_sample_critic(self):#由于critic计算V（s）可以利用历史采样样本
         d=self.sample_critic.__len__()
         if  d>=self.N_critic:#buffer满腾出空间，补充新样本
             print("sample_critic已满，扩充样本")
             for e in range(self.N_critic_remove):
                 self.sample_critic.popleft()
             d=self.sample_critic.__len__()
         while d<self.N_critic:
             state_now = self.env.reset()
             sum_loss=0
             while True:
                 self.env.render()
                 action_now=self.action(state_now)
                 state_next,reward,done,info = self.env.step(action_now)
                 x, x_dot, theta, theta_dot = state_next
                 r1 = (self.env.x_threshold - abs(x))/self.env.x_threshold - 0.8
                 r2 = (self.env.theta_threshold_radians - abs(theta))/self.env.theta_threshold_radians - 0.5
                 reward = r1 + r2
                 if  done:
                     d=d+1
                     self.sample_critic.append((state_now,state_next,reward,action_now,-1))
                     sum_loss+=reward
                     if d%5==0:
                         print("critic sample total reward:",sum_loss)
                     break
                 else:
                     d=d+1
                     self.sample_critic.append((state_now,state_next,reward,action_now,1))
                     sum_loss+=reward
                     state_now=state_next

     def coll_sample_actor(self):#由于actor计算Pi（at|st）
         self.sample_actor=[]#由于不可使用历史样本，每次必须清空
         d=0
         # d=self.sample_actor.__len__()
         # if  d>=self.N_actor:#buffer满腾出空间，补充新样本
         #     print("sample_actor已满，扩充样本")
         #     for e in range(self.N_critic_remove):
         #         self.sample_actor.popleft()
         #     d=self.sample_actor.__len__()
         while d<self.N_actor:
             state_now = self.env.reset()
             sum_loss=0
             index=0#用于记录当前reward需要折扣回的时间t
             while True:
                 self.env.render()
                 action_now=self.action(state_now)
                 state_next,reward,done,info = self.env.step(action_now)
                 x, x_dot, theta, theta_dot = state_next
                 r1 = (self.env.x_threshold - abs(x))/self.env.x_threshold - 0.8
                 r2 = (self.env.theta_threshold_radians - abs(theta))/self.env.theta_threshold_radians - 0.5
                 reward = r1 + r2
                 if  done:
                     index+=1;d=d+1
                     self.sample_actor.append((state_now,state_next,reward,action_now,index,-1))
                     sum_loss+=reward
                     if d%5==0:
                         print("actor sample total reward:",sum_loss)
                     break
                 else:
                     index+=1;d=d+1
                      #e[0]当前s，e[1]跳转s,e[2]reward，e[3]当前action,e[4]t折扣期,e[5]done
                     self.sample_actor.append((state_now,state_next,reward,action_now,index, 1))
                     sum_loss+=reward
                     state_now=state_next

     def V_value_train(self):#训练critic网络
         s_critic=list(self.sample_critic).copy()
         random.shuffle(s_critic)
         s_critic=s_critic[:int(self.N_critic_batch)]
         y=[]
         x=[]
         for e in s_critic:
             if e[4]>0:#非终止状态
                x.append(e[0])
                y.append(e[2]+self.r*self.critic_net(np.array([e[1]]))[0])
             else:#终止状态
                x.append(e[0])
                y.append(e[2])
         x=np.array(x,dtype=float)
         y=np.array(y,dtype=float)
         self.critic_net.train(x,y,x.__len__())

     #计算基于神经网的自然梯度的fisher信息矩阵
     # 需要提取net中weights变量list为vector，计算完成后变会dy_dweights修正梯度
     #把weights合并到vector里面
     def weight_gather(self,list_v=[]):
         result=[]
         for e in list_v:
             result.extend(list(np.reshape(e.numpy(),(-1))))
         return result#

     def record_weight_shape(self,list_v=[]):#记录神经网参数的shape,用于还原
         result=[]
         len=[]
         for e in list_v:
             temp=np.shape(e)
             l=1
             for v in list(temp):
                 l=l*v
             result.append(tuple(temp));len.append(l)
         return result,len#返回每个weight向量的shape和长度，用于还原

     def reshape_weight_shape(self,list_v=[]):#把参数向量还原会神经网weight的格式
         Dw=list(np.reshape(list_v,(-1)))#（-1）的shape
         result=[];i=0;start=0;end=0
         for len in self.record_len:
             end=end+len
             result.append(tf.reshape(tf.constant(Dw[start:end],dtype=tf.float32),self.record_shape[i]))
             start=end
             i+=1
         return result

     def actor_train_lr(self):#利用BF基函数训练模块
         s_actor=list(self.sample_actor).copy()
         random.shuffle(s_actor)
         s_actor=s_actor[:int(self.N_actor_batch)]
         for i in range(5):#计算5次梯度
             n=0
             #e[0]当前s，e[1]跳转s,e[2]reward，e[3]当前action,e[4]t折扣期,e[5]done
             for e in s_actor:#1轮梯度
                 #计算当前action
                 dz_dgrads=self.partial_derivative_lr(e[0],e[3])
                 #计算fisher-information matrix的其中一个元素
                 if n==0:
                     n=dz_dgrads.__len__();F=np.zeros((n,n));DJ=np.zeros((n,1))
                 dz_dgrads=tf.reshape(tf.constant(dz_dgrads,dtype=tf.float32),(n,1))#[[1],[2],[3],[3]]列
                 dz_dgrads_T=tf.reshape(dz_dgrads,(1,n))#[[1,2,3,4]]行
                 F=F+tf.matmul(dz_dgrads,dz_dgrads_T)#计算过fisher矩阵
                 #计算DJ的其中一个元素  #e[0]当前s，e[1]跳转s,e[2]reward，e[3]当前action,e[4]t折扣期,e[5]done
                 if e[5]>0:#非结束
                     A=(e[2]+self.r*self.critic_net(np.array([e[1]]))[0]-self.critic_net(np.array([e[0]]))[0])*np.power(self.r,e[4])
                     A=A*dz_dgrads;DJ=DJ+A
                 else:
                     A=(e[2]-self.critic_net(np.array([e[0]]))[0])*np.power(self.r,e[4])
                     A=A*dz_dgrads;DJ=DJ+A

             #计算参数F和DJ
             DJ=DJ/s_actor.__len__();F=F/s_actor.__len__()#自然梯度1
             DJ_row=tf.reshape(DJ,(1,n));DJ_col=tf.reshape(DJ,(n,1))

             deta=np.power(2.0*self.threshold/(tf.matmul(DJ_row,tf.matmul(F,DJ_col))[0][0]),0.5)#自然梯度1
             # deta=np.power(self.threshold/tf.matmul(DJ_row,DJ_col)[0][0],0.5) #自适应参数2

             #更新参数
             Dw=np.reshape(deta*tf.matmul(np.linalg.inv(F+np.eye(n)*0.000001),DJ_col).numpy(),(-1))#自然梯度1
             # Dw=deta*DJ_col #自适应参数2

             print("deta:",deta)
             self.w1_action_lr=np.reshape(Dw.numpy(),(-1))+self.w1_action_lr
         np.savetxt('./w1_action_lr.txt',self.w1_action_lr, delimiter=',')  # 数组x
         # print(self.w1_action_lr)

         #还原参数到神经网的shape

     def actor_train(self):#利用神经网训练actor网路
         s_actor=list(self.sample_actor).copy()
         random.shuffle(s_actor)
         s_actor=s_actor[:int(self.N_actor_batch)]
         n=0
         #e[0]当前s，e[1]跳转s,e[2]reward，e[3]当前action,e[4]t折扣期,e[5]done
         for e in s_actor:
             #计算当前action
             if e[3]==0:#left
                action_v=tf.constant([[1.0,0.0]],dtype=tf.float32)
             else:#right
                action_v=tf.constant([[0.0,1.0]],dtype=tf.float32)
             with tf.GradientTape() as tape:
                  y=self.actor_net([np.array(e[0])])
                  #actor神经网一次输出2个动作概率，但是只计算当前action的Dlog（action|st）
                  loss=tf.reduce_sum(tf.multiply(tf.math.log(y),action_v))
             #计算Dlog（action|st）
             grads=tape.gradient(loss,self.actor_net.trainable_weights)
             del tape
             dz_dgrads=self.weight_gather(grads)#把所有训练参数组合为一个向量,*神经网使用
             if n==0:
                 n=dz_dgrads.__len__();F=np.zeros((n,n));DJ=np.zeros(n)
                 self.record_shape,self.record_len=self.record_weight_shape(grads)#记录参数shape
             dz_dgrads=tf.reshape(tf.constant(dz_dgrads,dtype=tf.float32),(n,1))#[[1],[2],[3],[3]]列
             dz_dgrads_T=tf.reshape(dz_dgrads,(1,n))#[[1,2,3,4]]行
             #计算fisher-information matrix的其中一个元素
             F=F+tf.matmul(dz_dgrads,dz_dgrads_T).numpy()
             #计算DJ的其中一个元素  #e[0]当前s，e[1]跳转s,e[2]reward，e[3]当前action,e[4]t折扣期,e[5]done
             if e[5]>0:#非结束
                A=(e[2]+self.r*self.critic_net(np.array([e[1]]))[0]-self.critic_net(np.array([e[0]]))[0])*np.power(self.r,e[4])
                A=A*np.array(dz_dgrads,dtype=float)
                DJ=DJ+A
             else:
                A=(e[2]-self.critic_net(np.array([e[0]]))[0])*np.power(self.r,e[4])
                A=A*np.array(dz_dgrads,dtype=float)
                DJ=DJ+A

         #计算参数F和DJ
         F=tf.constant(F,dtype=tf.float32)
         DJ=tf.constant(DJ,dtype=tf.float32)
         DJ=DJ/s_actor.__len__()
         F=F/s_actor.__len__()#自然梯度1
         DJ_row=tf.reshape(DJ,(1,n))
         DJ_col=tf.reshape(DJ,(n,1))

         deta=np.power(2.0*self.threshold/(tf.matmul(DJ_row,tf.matmul(F,DJ_col))[0][0]),0.5)#自然梯度1
         # deta=np.power(self.threshold/tf.matmul(DJ_row,DJ_col)[0][0],0.5) #自适应参数2

         #更新参数
         Dw=tf.constant(-1.0)*deta*tf.matmul(np.linalg.inv(F+np.eye(n)*0.000001),DJ_col)#自然梯度1
         # Dw=tf.constant(-1.0)*deta*DJ_col #自适应参数2

         print("deta:",deta)
         grads_Dw=self.reshape_weight_shape(Dw,(-1))
         print(Dw)

         #还原参数到神经网的shape
         self.optimizer.apply_gradients(zip(grads_Dw,self.actor_net.trainable_weights))
         self.actor_net.save_weights(os.path.join(self.filepath,'actor_{}'.format(1.0)))
         self.critic_net.save_weights(os.path.join(self.filepath,'critic_{}'.format(1.0)))

     def train(self):
         for e  in range(100000):
             # if e<5:
             #    self.threshold=0.000000001
             # else:
             #    self.threshold=0.001
             # if e%5==0 and e!=0:
             #    if self.threshold<0.0000001:
             #       self.threshold=self.threshold*1.05
             #    else:
             #       self.threshold=0.0000001
             self.coll_sample_critic()
             self.V_value_train()
             self.coll_sample_actor()
             self.actor_train_lr()

a=natural_gradient()
a.train()


