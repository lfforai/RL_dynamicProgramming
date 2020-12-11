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

class LQR:
    def __init__(self, name="LQR",Ct_vector=[],ct_vector=[],Ft=[],ft=[],**kwargs):
        self.Ct=Ct_vector
        self.ct=ct_vector
        self.Ft=np.array(Ft)
        self.Ft_T=tf.transpose(self.Ft).numpy()
        self.ft=np.array(ft)

        self.t=0
        self.T=0
        self.Qt=[]
        self.qt=[]
        self.Kt=[]
        self.kt=[]
        self.Vt=[]
        self.vt=[]
        self.Vt_1=[]
        self.vt_1=[]
        self.V_xt=[]

    def catch_sub_matrix(self,matrix,row_index,col_index,row_size,col_size):
        #Ct=np.matrix(tf.reshape(np.array(list(range(36))),shape=(6,6)))
        # print(Ct)
        # lqr=LQR()
        # print(lqr.catch_sub_matrix(Ct,4,0,2,4))
        # print(lqr.catch_sub_matrix(Ct,0,4,4,2))
        # print(lqr.catch_sub_matrix(Ct,4,4,2,2))
        return matrix[row_index:row_index+row_size,col_index:col_index+col_size]

    def Kt_kt(self,Qt_ut_ut,Qt_ut_xt):
        Qt_ut_ut_inv=tf.linalg.inv(Qt_ut_ut)
        self.Kt=-1.0*tf.matmul(Qt_ut_ut_inv,Qt_ut_xt)
        # self.kt=-1.0*tf.

    def back_Kt(self,T,Ct=[],ct=[],Ft=[],ft=[],s_dim=4,a_dim=2):
        for  t in range(T-1,-1,-1):
             if  t==T-1:
                 self.Qt=np.matrix(Ct)
                 self.qt=np.matrix(ct)
             else:
                 self.Qt=self.Ct[t]+tf.matmul(tf.matmul(self.Ft_T,self.Vt_1),self.Ft)
                 self.qt=self.Ct[t]+tf.matmul(tf.matmul(self.Ft_T,self.Vt_1),self.ft)+tf.matmul(self.Ft_T,self.Vt_1)

             Qt_ut_ut=self.catch_sub_matrix(self.Qt,s_dim,s_dim,a_dim,a_dim)
             Qt_ut_xt=self.catch_sub_matrix(self.Qt,s_dim,0,a_dim,s_dim)
lqr=LQR()
lqr.back_Kt(T=10)