import tensorflow as tf
import numpy as np
import random

import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

matrix_p=[[0,0.5,0.5,0,0,0,0,0,0,0,0,0,0],
        [0,0,0.5,0.5,0,0,0,0,0,0,0,0,0],
        [0,0,0,0.5,0.5,0,0,0,0,0,0,0,0],
        [0,0,0,0,0.5,0.5,0,0,0,0,0,0,0],
        [0,0,0,0,0,0.5,0.5,0,0,0,0,0,0],
        [0,0,0,0,0,0,0.5,0.5,0,0,0,0,0],
        [0,0,0,0,0,0,0,0.5,0.5,0,0,0,0],
        [0,0,0,0,0,0,0,0,0.5,0.5,0,0,0],
        [0,0,0,0,0,0,0,0,0,0.5,0.5,0,0],
        [0,0,0,0,0,0,0,0,0,0.0,0.5,0.5,0],
        [0,0,0,0,0,0,0,0,0,0.0,0,0.5,0.5],
        [0,0,0,0,0,0,0,0,0,0,0,0,1.0],
        [0,0,0,0,0,0,0,0,0,0,0,0,1.0]
        ]

matrix_v=[[0,-3,-3,0,0,0,0,0,0,0,0,0,0],
        [0,0,-3,-3,0,0,0,0,0,0,0,0,0],
        [0,0,0,-3,-3,0,0,0,0,0,0,0,0],
        [0,0,0,0,-3,-3,0,0,0,0,0,0,0],
        [0,0,0,0,0,-3,-3,0,0,0,0,0,0],
        [0,0,0,0,0,0,-3,-3,0,0,0,0,0],
        [0,0,0,0,0,0,0,-3,-3,0,0,0,0],
        [0,0,0,0,0,0,0,0,-3,-3,0,0,0],
        [0,0,0,0,0,0,0,0,0,-3,-3,0,0],
        [0,0,0,0,0,0,0,0,0,0.0,-3,-3,0],
        [0,0,0,0,0,0,0,0,0,0.0,0,-3,-3],
        [0,0,0,0,0,0,0,0,0,0,0,0,-2],
        [0,0,0,0,0,0,0,0,0,0,0,0,0]
        ]

def random_pick(some_list, probabilities):
    x = random.uniform(0,1)
    cumulative_probability = 0.0
    for item, item_probability in zip(some_list, probabilities):
        cumulative_probability += item_probability
        if x < cumulative_probability:
            break
    return item

class Markov_matrix:
     def __init__(self,matrix_p=matrix_p,matrix_v=matrix_v):
         self.self_matrix_p=matrix_p
         self.self_matrix_v=matrix_v
         temp=np.array(matrix_p)
         self.rows= len(temp)
         self.cols= len(temp[0])
         self.sample_list=[]

     def action2col(self,row):
         result=1
         if row<self.rows-2: #not last col
             result=random.choices([row+1,row+2], self.self_matrix_p[row][row+1:row+3])[0]
         elif row==self.cols-2:
            result=row+1
         else:
            result=row
         return  result

     def action2value(self,row,col):
         result=self.self_matrix_v[row][col]
         return  result

     def sample(self,num=100):
         self.sample_list=[]
         i=0
         state_now=0
         state_next=1
         while(1):
            state_next=self.action2col(state_now)
            self.sample_list.append([state_now,state_next,self.action2value(state_now,state_next)])
            if state_next==self.rows-1: #last
               state_now=0
            else:
               state_now=state_next
            i=i+1
            if i>=num:
               break

# TD(l)
class TD_table:
      def __init__(self,state_num,r=1.0,aphl=0.95,beta=0.75,matrix=Markov_matrix()):
          self.state_num=state_num
          self.es=np.zeros(state_num)
          self.V=np.zeros(state_num)
          self.r=r
          self.beta=beta
          self.matrix=matrix
          self.aphl=aphl
          self.N=500

      def re_aphl(self,i):
          if self.aphl>0.1:
             self.aphl =self.aphl*(self.N/(self.N+i))

      def sample(self,num):
          self.matrix.sample(num)

      def es_renew(self,state_now):
          for i in range(self.state_num):
              if i==state_now:
                 self.es[i]=self.r*self.beta*self.es[i]+1.0
              else:
                  self.es[i]=self.r*self.beta*self.es[i]

      def vs_renew(self,vlaue):
          for i in range(self.state_num):
               self.V[i]=self.V[i]+self.aphl*vlaue*self.es[i]

      def train(self):
          i=0
          for e in  self.matrix.sample_list:
              value=e[2]+self.r*self.V[e[1]]-self.V[e[0]]
              self.es_renew(e[0])
              self.vs_renew(value)
              self.re_aphl(i)
              i=i+1

# o=TD_table(13)
# o.sample(50000)
# std=np.flipud(np.arange(13))*(-2.0)
# print(std)
# print(o.matrix.sample_list)
# o.train()
# print(ti,o.V-std)

#RLS_TD
class RLS_TD:
    def __init__(self,state_num,r=1.0,aphl=0.95,beta=0.75,matrix=Markov_matrix()\
                 ,basifunnum=2):
        self.state_num=state_num
        self.baisfun_num=basifunnum #2
        self.es=np.zeros(state_num)
        self.V=np.zeros(state_num)
        self.r=r
        self.beta=beta
        self.matrix=matrix
        self.aphl=aphl
        self.N=500
        self.sum=np.sum(np.arange(13))
        self.P=tf.eye(2)      #[[1. 0.],[0. 1.]]
        self.Z=tf.zeros((2,1))  #[[0.][0.]]
        self.K=tf.zeros((2,1))
        self.W=tf.random.normal((2,1), mean=0.0, stddev=1.0)

    def sample(self,num):
            self.matrix.sample(num)

    def base_function(self,state_now):
        v=(12.0-state_now)/self.sum   #one
        return tf.convert_to_tensor(np.array([[v,v*v]]),dtype=tf.float32)

    def train(self):
        for e in self.matrix.sample_list:
            v0=self.base_function(e[0])
            v1=self.base_function(e[1])
            self.K=tf.matmul(self.P,self.Z)/(1.0+tf.matmul(tf.matmul(v0-self.r*v1,self.P),self.Z))
            self.W=self.W+tf.matmul(self.K,(tf.constant([e[2]],dtype=tf.float32)-tf.matmul((v0-self.r*v1),self.W)))
            self.P=self.P-tf.matmul(tf.matmul(self.P,self.Z)\
                   ,tf.matmul(tf.linalg.inv(1.0+tf.matmul(tf.matmul((v0-self.r*v1),self.P),self.Z)) \
                   ,tf.matmul((v0-self.r*v1),self.P)))
            if e[1]==12:
               self.Z=tf.zeros((2,1))
            else:
               self.Z=self.r*self.beta*self.Z+tf.transpose(v0)

# o=RLS_TD(13)
# o.sample(10000)
# #print(o.matrix.sample_list)
# o.train()
# print(o.W)
# lf=[]
# for i in range(13):
#     lf.append(tf.matmul(o.base_function(i),o.W).numpy()[0][0])
# lf=np.array(lf)
# std=np.flipud(np.arange(13))*(-2.0)
# print(lf-std)

#KLS_TD
class KLS_TD:
    def __init__(self,state_num,r=1.0,aphl=0.95,beta=0.5,matrix=Markov_matrix() \
                 ,t=1000,var=2.0): #t is the number of sample
        self.state_num=state_num
        self.es=np.zeros(state_num)
        self.V=np.zeros(state_num)
        self.r=r
        self.beta=beta
        self.matrix=matrix
        self.aphl=aphl
        self.t=t
        self.var=var
        self.sum=np.sum(np.arange(13))
        self.sum=12
        self.R=[]     #[[1. 0.],[0. 1.]]
        self.Z=np.zeros((self.t,self.t))  #[[0.][0.]]
        self.K=np.zeros((self.t,self.t))
        self.H=np.zeros((self.t-1,self.t))
        self.W=tf.random.normal((t,1), mean=0.0, stddev=1.0)
        self.value=[]

    def sample(self):
        self.matrix.sample(self.t)
        # print(len(self.matrix.sample_list))
        for e in self.matrix.sample_list:
            self.value.append(e[0])
            self.R.append(e[2])
        self.R=np.reshape(np.matrix(self.R,dtype=float),(self.t,1))

    def K_func(self,x,y):#gaosi
        x1=(12.0-x)/self.sum   #one
        y1=(12.0-y)/self.sum
        x1=np.array([x1,np.power(x1,2)])
        y1=np.array([y1,np.power(y1,2)])
        result=np.exp(-1.0*np.dot(x1-y1,x1-y1)/(2.0*self.var))
        return result

    def H_renew(self):
        i=0
        for e in self.matrix.sample_list:
            if i<(self.t-1):
                self.H[i][i]=1.0
                if e[1]!=12:
                   self.H[i][i+1]=-self.r
            else:
                break
            i=i+1
        #print(self.H)

    def K_renew(self):
        for i in range(self.t):
            for j in range(self.t):
                self.K[i][j]=self.K_func(self.value[i],self.value[j])
        # print(self.K)

    def Z_renew(self):
        for i in range(self.t):
               if self.matrix.sample_list[i][0]!=0:
                  self.Z[:,i]=self.K[:,i]+self.r*self.beta*self.Z[:,i-1]
               else:
                  self.Z[:,i]=self.K[:,i]
        # print(self.Z)
        # for i in range(self.t):
        #     if i>0:
        #        self.Z[:,i]=self.K[:,i]+self.r*self.beta*self.K[:,i-1]
        #     else:
        #        self.Z[:,0]=self.K[:,0]

    def train(self):
        # print(tf.matmul(tf.matmul(self.Z[:,:-1],self.H),self.K))
        self.W=tf.matmul(tf.linalg.inv(tf.matmul(tf.matmul(self.Z[:,:-1],self.H),self.K)+np.eye(self.t)*0.001) \
                 ,tf.matmul(self.Z,self.R))

    def pre(self,i):
        list_o=[]
        for e in self.matrix.sample_list:
            list_o.append(self.K_func(i,e[0]))
        list_o=np.reshape(np.array(list_o,dtype=float),(1,self.t))
        return np.matmul(list_o,self.W)[0][0]

o=KLS_TD(13)
o.sample()
o.H_renew()
o.K_renew()
o.Z_renew()
o.train()
# print(o.W)
l=[]
for i in range(13):
    l.append(o.pre(i))
print(l)


