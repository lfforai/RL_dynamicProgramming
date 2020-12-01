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

from tensorflow.keras import backend as K
def actor_loss(y_true, y_pred):
    return -1.0*tf.reduce_mean(tf.reduce_sum(tf.multiply(y_true,tf.math.log(y_pred)),axis=1))

class actor(tf.keras.Model):# compute value
    def __init__(self,filename='./data/actor',name="actor",training=True,**kwargs):
        super(actor, self).__init__(name=name,**kwargs)
        self.training=training
        self.block_1 = Linear(name="critic_linear1",units=128,training=training)
        self.block_2 = Linear(name="critic_linear2",units=128,training=training)
        self.block_3 = Linear(name="critic_linear2",units=2,training=training)
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
    def call(self, inputs_state):
        x = self.block_1(inputs_state)
        x = tf.nn.relu(x)
        x = self.block_2(x)
        x = tf.nn.relu(x)
        x = self.block_3(x)
        x =tf.nn.softmax(x)
        return x

    def train(self,x_train,y_ture,size,filename='./data/actor'):
        optimizer = tf.keras.optimizers.Adam(learning_rate=1.5e-3)
        self.compile(optimizer, loss=actor_loss)
        self.fit(x_train, y_ture, epochs=5,batch_size=size,verbose=1)
        self.save_weights(filename+"/prameter")

    def important_action(self,state):
        state=np.reshape(state,(1,4))
        lpro,rpro=self.predict(state)[0] #left=pro,right=1-pro
        lr=numpy.random.choice([0,1],size=None,replace=False,p=[lpro,1-lpro])
        if lr==0:
           return lr,lpro
        else:
           return lr,1-lpro

    def get_config(self):
        config = super(actor, self).get_config()
        return config

c=actor()
car_gym=gym_CartPole_v0(action_class=c,randomaction=False)
def train_actor(actor_o=c,car_o=car_gym,bitch_size=350):
    import random
    for i in range(10000):
        #catch sample
        car_o.important_sample(sample_num=500,sample_total_num=10000,rdaction=False)
        temp_list=list(car_o.sample).copy()
        random.shuffle(temp_list)
        temp_list=temp_list[:bitch_size]   #bitch_size
        state_list=[list(e[0]) for e in temp_list]
        action_input_list=actor_o.predict(state_list)
        reward_list=[e[2] for e in temp_list]
        action_list=[e[3] for e in temp_list]
        length= len(action_list)
        reward_list_t=[]
        for c in range(length):
            if action_list[c]==0:
               reward_list_t.append([reward_list[c],0.0])
            else:
               reward_list_t.append([0.0,reward_list[c]])
        reward_list_temp=np.multiply(reward_list_t,np.array(action_input_list))
        #print(-tf.reduce_sum(tf.reduce_sum(tf.multiply(actor_o(state_list),reward_list_temp),axis=1)))
        actor_o.train(np.array(state_list),reward_list_temp,bitch_size,filename='./data/actor')
train_actor()