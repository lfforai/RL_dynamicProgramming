import gym
import numpy
from tensorflow.keras import layers
import tensorflow as tf
from collections import deque

#layers model
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
        #super(Linear, self).build(input_shape)

    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b

    def get_config(self):
        config = super(Linear, self).get_config()
        config.update({'units': self.units})
        return config

#gym modle
class gym_CartPole_v0:
    def __init__(self,action_class=[],randomaction=True):
        self.env=gym.make('CartPole-v0') #
        #记录样本
        self.sample=deque([])
        #初始化网络
        self.action_class=action_class
        self.random_action=randomaction
        self.r=0.9

    def action(self,state):
        if self.random_action==True:
            return  numpy.random.choice([0,1],size=None,replace=False,p=[0.5,0.5])
        else:
            return  action_class.action(state)

    def important_action(self,state):
        if self.random_action==True:
            rezult=numpy.random.choice([0,1],size=None,replace=False,p=[0.5,0.5])
            return [rezult,0.5]  #return [action,probility]
        else:
            return self.action_class.important_action(state) #return [action,probility]

    def sample(self,sample_num=1000,rdaction=True):#由于actor计算Pi（at|st）
        self.random_action=rdaction
        print("gym_CartPole_v0 samlpe start!")
        self.sample=[]#由于不可使用历史样本，每次必须清空
        d=0
        while d<sample_num:
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
                    self.sample.append((state_now,state_next,reward,action_now,-1))
                    sum_loss+=reward
                    if d%10==0:
                        print("actor sample total reward:",sum_loss)
                    break
                else:
                    if d<sample_num:
                        #e[0]当前s，e[1]跳转s,e[2]reward，e[3]当前action,e[4]t折扣期,e[5]done
                        self.sample.append((state_now,state_next,reward,action_now, 1))
                        sum_loss+=reward
                        state_now=state_next
                    else:
                        print(" total num sample:", len(self.sample))
                        exit()
                    d=d+1
        print(" total num sample:", len(self.sample))

    def imsample_change_oreward(self): #old samples reward
        #e[0]当前s，e[1]跳转s,e[2]reward，e[3]当前action,e[4]done,e[5]:pro
        temp=[]
        if len(self.sample)>0:
           temp=[(e[0],e[1],e[2]/e[5],e[3],e[4],e[5],0) for e  in self.sample if e[6]==1]
           temp.extend([e for e in self.sample if e[6]==0])
        self.sample.clear()
        self.sample=deque(temp)

    #important smaple
    def important_sample(self,sample_num=200,sample_total_num=2000,rdaction=True):#由于actor计算Pi（at|st）
        self.random_action=rdaction
        self.imsample_change_oreward()
        length=len(self.sample)
        print("gym_CartPole_v0 samlpe start! length=:",length)
        if sample_total_num-length<sample_num:
           print("sample  is not  too big ,needing pop:",sample_num-sample_total_num+length)
           for index  in range(sample_num-sample_total_num+length):
               self.sample.pop()
        d=0
        while d<sample_num:
            state_now = self.env.reset()
            sum_loss=0
            temp_list=[]
            while True:
                self.env.render()
                action_now,probility=self.important_action(state_now)
                state_next,reward,done,info = self.env.step(action_now)
                x, x_dot, theta, theta_dot = state_next
                r1 = (self.env.x_threshold - abs(x))/self.env.x_threshold - 0.8
                r2 = (self.env.theta_threshold_radians - abs(theta))/self.env.theta_threshold_radians - 0.5
                reward = r1 + r2
                if  done:
                    if d<sample_num:
                        #e[0]当前s，e[1]跳转s,e[2]reward，e[3]当前action,e[4]done,e[5]:pro,e[6]:new
                        temp_list.append((state_now,state_next,reward,action_now,-1,probility,1))
                        reward_back_total=0
                        length_o=len(temp_list)
                        temp=[]
                        for index in range(length_o):
                            index_o=length_o-1-index
                            reward_back_total=self.r*reward_back_total+temp_list[index_o][2]
                            temp.append([temp_list[index_o][0],temp_list[index_o][1],reward_back_total,temp_list[index_o][3],\
                            temp_list[index_o][4],temp_list[index_o][5],temp_list[index_o][6]])
                        temp_list.clear()
                        sum_loss+=reward
                        self.sample.extend(temp)
                    else:
                        return 0
                    d=d+1
                    if d%10==0:
                        print("actor sample total reward:",sum_loss)
                    break
                else:
                    if d<sample_num:
                        #e[0]当前s，e[1]跳转s,e[2]reward，e[3]当前action,e[4]done,e[5]:pro,e[6]:new
                        temp_list.append((state_now,state_next,reward,action_now, 1,probility,1))
                        sum_loss+=reward
                        state_now=state_next
                    else:
                        print(" total num sample:", len(self.sample))
                        return 0
                    d=d+1
        print(" total num sample:", len(self.sample))



