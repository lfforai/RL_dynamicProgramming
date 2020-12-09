#r more using history sample in RL study ,wildy useing important sample
#this shows how to use  important sampel in police RL

from __future__ import absolute_import, division, print_function, unicode_literals

from tensorflow.keras import layers
from tensorflow import keras
import tensorflow as tf
import gym
import numpy as np
import random
import os
import sys
np.set_printoptions(precision=4)

# sys.path.append(r'/root/IdeaProjects/RL_dynamicProgramming/.idea/gym_class/')
sys.path.append(r'../gym_class/')
from gym_class import *

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

#base on net
#V(state)
class critic(tf.keras.Model):
    def __init__(self, name="critic",training=True,filename="",**kwargs):
        super(critic, self).__init__(name=name,**kwargs)
        self.training=training
        self.filename=filename
        self.block_1 = Linear(name="critic_linear1",units=64,training=training)
        self.block_2 = Linear(name="critic_linear2",units=64,training=training)
        self.block_3 = Linear(name="critic_linear3",units=1,training=training)
        import os.path
        if os.path.isdir(filename):
            if os.listdir(filename):
                print("init load prameter from "+filename)
                self.load_weights(filename+"/prameter")
            else:
                print("init prameter from tf.random 1")
        else:
            print("init prameter from tf.random 2")

    #Q(state,action),input_state=[none,[a1,a2,a3,a4]],input_action=[none,[0,1]]
    #if action==1  [zeors,[a1,a2,a3,a4]]  else action==0  [[a1,a2,a3,a4],zeros]

    @tf.function
    def call(self,inputs):
        #x=tf.concat([inputs_state,input_action],-1)
        x = self.block_1(inputs)
        x = tf.nn.relu(x)
        x = self.block_2(x)
        x = tf.nn.relu(x)
        x = self.block_3(x)
        return x

    def train(self,x_train,y_ture,size):
        optimizer = tf.keras.optimizers.Adam(learning_rate=1.5e-3)
        self.compile(optimizer, loss=tf.keras.losses.MeanSquaredError())
        self.fit(x_train, y_ture, epochs=5,batch_size=size,verbose=1)
        self.save_weights(self.filename+"/prameter")

    def get_config(self):
        config = super(critic, self).get_config()
        return config

# KL_loss
from tensorflow.keras import backend as K
def KL_loss(y_true, y_pred):
    return tf.reduce_mean(tf.reduce_sum(tf.multiply(y_true,tf.math.log(y_pred)),axis=1))

def actor_loss(y_true, y_pred):
    return tf.reduce_mean(tf.reduce_sum(tf.multiply(y_true,tf.math.log(y_pred)),axis=1))

#policy net
class actor(tf.keras.Model):
    def __init__(self, name="actor",training=True,filename="",KL_net=[],**kwargs):
        super(actor, self).__init__(name=name,**kwargs)
        self.training=training
        self.filename=filename
        self.KL_net=KL_net
        self.e=0.000001
        self.block_1 = Linear(name="actor_linear1",units=64,training=training)
        self.block_2 = Linear(name="actor_linear2",units=64,training=training)
        self.block_3 = Linear(name="actor_linear3",units=2,training=training)
        import os.path
        if os.path.isdir(filename):
            if os.listdir(filename):
                print("init load prameter from "+filename)
                self.load_weights(filename+"/prameter")
            else:
                print("init prameter from tf.random 1")
        else:
            print("init prameter from tf.random 2")

    @tf.function
    def call(self, inputs):
        x = self.block_1(inputs)
        x = tf.nn.relu(x)
        x = self.block_2(x)
        x = tf.nn.relu(x)
        x = self.block_3(x)
        x=tf.nn.softmax(x)
        return x

    def train(self,x_train,y_true,action_KL,size,epochs=5):
        for  b in range(epochs):
            #compute KL
            # self.call(x_train)
            self.KL_net.modle_grads(KL_loss,action_KL,x_train,self)
            self.KL_net.F_matrix()
            F=self.KL_net.F

            #compute Dj
            optimizer = tf.keras.optimizers.SGD(learning_rate=-1.0)
            with tf.GradientTape() as tape:
                 logits = self(x_train, training=True)
                 loss_value = actor_loss(y_true, logits)
            Dj=tape.gradient(loss_value, self.trainable_weights)
            Dj_array=self.KL_net.grads2array(Dj)
            Dj_array_col=tf.reshape(Dj_array,shape=(-1,1))
            Dj_array_T=tf.reshape(Dj_array,shape=(1,-1))
            length=len(self.KL_net.grads_array)

            beta=np.sqrt(2*self.e/tf.matmul(tf.matmul(Dj_array_T,F),Dj_array_col).numpy()[0])
            print("a::",beta)
            # print(tf.eye(length))
            beta=0.0015
            grads=beta*tf.matmul(tf.linalg.inv(F+tf.eye(length)*0.01),Dj_array_col)
            grads=tf.reshape(grads,shape=(-1)).numpy()
            grads=self.KL_net.array2grads(self.KL_net.shapes,grads)
            optimizer.apply_gradients(zip(grads, self.trainable_weights))
        print(tf.reduce_sum(self.trainable_weights[0])+tf.reduce_sum(self.trainable_weights[1])+
              tf.reduce_sum(self.trainable_weights[2])+tf.reduce_sum(self.trainable_weights[3])+
              tf.reduce_sum(self.trainable_weights[4])+tf.reduce_sum(self.trainable_weights[5]))
        self.save_weights(self.filename+"/prameter")

    def get_config(self):
        config = super(actor, self).get_config()
        return config

class natrue_DGR:
    def __init__(self,filepath_c="./data/natural/critic",filepath_a="./data/natural/actor"):
        self.filepath_c=filepath_c
        self.filepath_a=filepath_a
        self.KL_net=KL_grad()     #KL_grd
        self.actor_net=actor(filename=filepath_a,KL_net=self.KL_net)    #计算actor=PI（at|st）,nature_DGR
        self.critic_net=critic(filename=filepath_c)  #计算V（s）
        self.car_gym=car_o=gym_CartPole_v0(action_class=self,randomaction=False)   #catch sample
        self.env=gym.make("CartPole-v1") #
        self.r=0.9
        if not os.path.exists(self.filepath_c) or not os.path.exists(self.filepath_a):
            print("no 参数")
        else:
           print("load 参数")
           self.actor_net.load_weights(filepath_a+"/prameter")
           self.critic_net.load_weights(filepath_c+"/prameter")

    def action(self,state):
        state=tf.constant([list(state)])
        probility=self.actor_net(state)[0].numpy()
        return np.random.choice([0,1],1,False,[probility[0],1.0-probility[0]])[0]

    def state_action_togethor(self,state,action): #[[a1,a2,a3,a4],zeros]
        inputs_state=state
        inputs_action=action
        action_len=2
        shapes=tf.shape(inputs_state)
        state_length=shapes[-1]
        state_action_length=state_length*action_len
        temp=[]
        for i in range(shapes[0]): #batch length
            value=inputs_state[i]
            action=inputs_action[i]
            if action==0:#left
                temp.append(tf.concat([value,tf.zeros(state_length)],-1))
            else:        #right
                temp.append(tf.concat([tf.zeros(state_length),value],-1))
        return tf.reshape(tf.concat(temp,0),shape=(-1,action_len*state_length))

    def train_critic(self,x,y,size):
        self.critic_net.train(x,y,size)

    def train_actor(self,x,y,a,size):
        self.actor_net.train(x,y,a,size,epochs=5)

    def train_all(self,bitch_size=250,N=10000):
        for g in range(N):
            self.car_gym.sample_run(sample_num=300)
            sample=self.car_gym.sample
            random.shuffle(sample)
            sample_bitch=sample[:bitch_size]  #bitch_size

            #e[0]当前s，e[1]跳转s,e[2]reward，e[3]当前action,e[4]t折扣期,e[5]done
            #train V(s)
            state_next=[e[1] for e in sample_bitch]
            state_now=[e[0] for e in sample_bitch]
            reward=[e[2] for e in sample_bitch]
            done=[e[5] for e in  sample_bitch]
            time=[e[4] for e in sample_bitch]  #r^t
            # x=self.state_action_togethor(state,action)  #[[a1,a2,a3,a4],zeros]
            Q_snext_a=self.critic_net(np.array(state_next)).numpy()
            y=[]
            for i in range(bitch_size):
                if done[i]!=-1:
                   y.append(reward[i]+self.r*Q_snext_a[i][0])
                else:
                   y.append(reward[i])
            y=np.array(y)
            x=np.array(state_now)
            self.train_critic(x,y,bitch_size)

            #train p(at|st)
            self.car_gym.sample_run(sample_num=400)
            sample=self.car_gym.sample
            random.shuffle(sample)
            sample_bitch=sample[:bitch_size]  #bitch_size

            state_next=[e[1] for e in sample_bitch]
            state_now=[e[0] for e in sample_bitch]
            reward=[e[2] for e in sample_bitch]
            done=[e[5] for e in  sample_bitch]
            time=[e[4] for e in sample_bitch]  #r^t
            action=[e[3] for e in sample_bitch]

            Q_snext_a=self.critic_net(np.array(state_next)).numpy()
            Q_s_a=self.critic_net(np.array(state_now)).numpy()
            y=[]
            KL_action=[]
            for i in range(bitch_size):
                  if done[i]!=-1:
                        if action[i]==0:
                           y.append([np.power(self.r,time[i])*(reward[i]+self.r*Q_snext_a[i][0]-Q_s_a[i][0]),0.0])
                           KL_action.append([1.0,0.0])
                        else:
                           y.append([0.0,np.power(self.r,time[i])*(reward[i]+self.r*Q_snext_a[i][0]-Q_s_a[i][0])])
                           KL_action.append([0.0,1.0])
                  else:
                        if action[i]==1:
                           y.append([0.0,np.power(self.r,time[i])*(reward[i]-Q_s_a[i][0])])
                           KL_action.append([0.0,1.0])
                        else:
                           y.append([np.power(self.r,time[i])*(reward[i]-Q_s_a[i][0]),0.0])
                           KL_action.append([1.0,0.0])
            y=np.array(y)
            KL_action=np.array(KL_action,dtype=float)
            self.train_actor(x,y,KL_action,bitch_size)

natrue=natrue_DGR()
natrue.train_all()
#print(car_o.sample)
