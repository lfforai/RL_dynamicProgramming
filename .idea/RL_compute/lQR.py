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
    def __init__(self, name="LQR",Ct_vector=[],ct_vector=[],T=0,**kwargs):
        self.Ct=Ct_vector
        self.ct=ct_vector

        self.A=0
        self.B=0 #
        self.Ft=0
        self.Ft_T=0
        self.ft=0
        self.s_dim=0
        self.a_dim=0

        self.t=0
        self.T=T
        self.Qt=0
        self.qt=0
        self.Kt=0
        self.kt=0
        self.Kt_vector=[]
        self.kt_vector=[]

        self.Vt=0
        self.vt=0

        self.Vt_1=0
        self.vt_1=0

        self.x_vector=[]
        self.u_vector=[]

        self.ut_vector=[]

    def catch_sub_matrix(self,matrix,row_index,col_index,row_size,col_size):
        #Ct=np.matrix(tf.reshape(np.array(list(range(36))),shape=(6,6)))
        # print(Ct)
        # lqr=LQR()
        # print(lqr.catch_sub_matrix(Ct,4,0,2,4))
        # print(lqr.catch_sub_matrix(Ct,0,4,4,2))
        # print(lqr.catch_sub_matrix(Ct,4,4,2,2))
        return matrix[row_index:row_index+row_size,col_index:col_index+col_size]

    def Kt_kt(self,Qt_ut_ut,Qt_ut_xt,q_ut):
        Qt_ut_ut_inv=tf.linalg.inv(Qt_ut_ut)
        self.Kt=-1.0*tf.matmul(Qt_ut_ut_inv,Qt_ut_xt)
        self.kt=-1.0*tf.matmul(Qt_ut_ut_inv,q_ut)

    def Vt_vt(self,Kt,kt,Qt_xt_xt,Qt_xt_ut,Qt_ut_xt,Qt_ut_ut,q_xt,q_ut):
        Kt_T=tf.transpose(Kt)
        self.Vt=Qt_xt_xt+tf.matmul(Qt_xt_ut,Kt)+tf.matmul(Kt_T,Qt_ut_xt)+\
                tf.matmul(tf.matmul(Kt_T,Qt_ut_ut),Kt)
        self.vt=q_xt+tf.matmul(Qt_xt_ut,kt)+tf.matmul(Kt_T,q_ut)+tf.matmul(Kt_T,tf.matmul(Qt_ut_ut,kt))

    def back_Kt(self):
        s_dim=self.s_dim
        a_dim=self.a_dim
        for  t in range(self.T-1,-1,-1):
             if  t==T-1:
                 self.Qt=np.matrix(Ct[t])
                 self.qt=np.matrix(ct[t])
             else:
                 self.Qt=self.Ct[t]+tf.matmul(tf.matmul(self.Ft_T,self.Vt_1),self.Ft)
                 self.qt=self.ct[t]+tf.matmul(tf.matmul(self.Ft_T,self.Vt_1),self.ft)+tf.matmul(self.Ft_T,self.vt_1)

             Qt_xt_xt=self.catch_sub_matrix(self.Qt,0,0,s_dim,s_dim)
             Qt_xt_ut=self.catch_sub_matrix(self.Qt,0,s_dim,s_dim,a_dim)
             Qt_ut_ut=self.catch_sub_matrix(self.Qt,s_dim,s_dim,a_dim,a_dim)
             Qt_ut_xt=self.catch_sub_matrix(self.Qt,s_dim,0,a_dim,s_dim)

             q_ut=self.catch_sub_matrix(s_dim,0,a_dim,1)
             q_xt=self.catch_sub_matrix(0,0,s_dim,1)

             self.Kt_kt(Qt_ut_ut,Qt_ut_xt,q_ut)
             self.Vt_vt(Kt,kt,Qt_xt_xt,Qt_xt_ut,Qt_ut_xt,Qt_ut_ut,q_xt)

             self.Kt_vector.append(self.Kt)
             self.kt_vector.append(self.kt)

             self.Vt_1=self.Vt
             self.vt_1=self.vt

    def Ft_init(self,h=600.0,m=1400.0,s_dim=4,a_dim=2):
        self.s_dim=s_dim
        self.a_dim=a_dim
        # m =car   kg
        # h =time,5 minites
        drag_ratio=(1.0/16.0)*3*0.45*15    #Fw=1/16*A*Cw*v^2 ,cw=0.45,A=3 m^2 ，v=15m/s
        A_beta=1.0-h*drag_ratio/m
        B_beta=h/m
        # A=np.matrix([[1.0,0,h,0],[0,1.0,0,h],[0.0,0.0,A_beta,0],\
        #    [0.0,0.0,0.0,A_beta]])
        A=np.matrix([[1.0,0,h-drag_ratio*h*h/m/2.0,0],[0,1.0,0,h-drag_ratio*h*h/m/2.0],[0.0,0.0,A_beta,0], \
                     [0.0,0.0,0.0,A_beta]])
        self.A=A
        B=np.matrix([[h*h/m/2.0,0],[0,h*h/m/2.0],[B_beta,0],[0,B_beta]])
        self.B=B
        # print(np.hstack((A,B)))
        self.Ft=np.hstack((A,B))
        self.Ft_T=tf.transpose(self.Ft)
        self.ft=np.zeros((self.s_dim,1))
        #xt=[x,y,vx,vy] ut=[xut,yut]

    def f(self,xt,ut): #p(st+1|st,at)=1
        return tf.matmul(self.A,xt)+tf.matmul(self.B,ut)

    #nearest point path
    # def Ct_init(self,Ct_vector=[],ct_vector=[],T=100):
    #     self.Ct=Ct_vector
    #     self.ct=ct_vector
    #     self.T=T
    #     if len(self.Ct)!=self.T:
    #        print("Ct length must be same as T")

    #1、reach end of point with lessest force
    def Ct_init_end_point(self,T=1000):
        temp_Ct=np.zeros((self.s_dim+self.a_dim,self.s_dim+self.a_dim))
        temp_ct=np.zeros((self.s_dim+self.a_dim,1))
        temp_Ct[self.s_dim][self.s_dim]=1.0
        temp_Ct[self.s_dim+1][self.s_dim+1]=1.0

        self.T=T
        self.Ct=[]
        self.ct=[]
        for i in range(self.T):
            self.Ct.append(temp_Ct)
            self.ct.append(temp_ct)

        if len(self.Ct)!=self.T:
           print("Ct length must be same as T")

    def Kt_kt_end_point(self,x_des=np.array([[100.0],[200.0]])):
        #xdes=c*A*xt+c*B*ut
        #(c*B)^T*xdes=(c*B)^T(c*A)*xt+(c*B)^T*(c*B)*ut
        #ut=-inv((c*B)^T*(c*B))*(c*B)^T(c*A)*xt+inv((c*B)^T*(c*B))*(c*B)^T*xdes
        C_x_des=np.matrix([[1.0,0.0,0.0,0.0],[0.0,1.0,0.0,0.0]])
        A_o=tf.matmul(C_x_des,self.A)
        B_o=tf.matmul(C_x_des,self.B)
        B_o_T=tf.transpose(B_o)
        A_o=tf.matmul(B_o_T,A_o)
        x_des=tf.matmul(B_o_T,x_des)
        B_o_T_mul_B_O_inv=tf.linalg.inv(tf.matmul(B_o_T,B_o))
        self.Kt=-1.0*tf.matmul(B_o_T_mul_B_O_inv,A_o)
        self.kt=tf.matmul(B_o_T_mul_B_O_inv,x_des)

    def back_Kt_end_point(self,x_des=(100.0,200.0)):
        s_dim=self.s_dim
        a_dim=self.a_dim
        for t in range(self.T-1,-1,-1):
            if  t==self.T-1:
                self.Qt=np.matrix(self.Ct[t])
                self.qt=np.matrix(self.ct[t])
            else:
                self.Qt=self.Ct[t]+tf.matmul(tf.matmul(self.Ft_T,self.Vt_1),self.Ft)
                self.qt=self.ct[t]+tf.matmul(tf.matmul(self.Ft_T,self.Vt_1),self.ft)+tf.matmul(self.Ft_T,self.vt_1)

            #print(self.Qt)
            Qt_xt_xt=self.catch_sub_matrix(self.Qt,0,0,s_dim,s_dim)
            #print(Qt_xt_xt)
            Qt_xt_ut=self.catch_sub_matrix(self.Qt,0,s_dim,s_dim,a_dim)
            #print(Qt_xt_ut)
            Qt_ut_ut=self.catch_sub_matrix(self.Qt,s_dim,s_dim,a_dim,a_dim)
            #print(Qt_ut_ut)
            Qt_ut_xt=self.catch_sub_matrix(self.Qt,s_dim,0,a_dim,s_dim)
            # print(Qt_ut_xt)

            q_ut=self.catch_sub_matrix(self.qt,s_dim,0,a_dim,1)
            q_xt=self.catch_sub_matrix(self.qt,0,0,s_dim,1)
            # print(q_ut)
            # print(q_xt)

            #self.Kt_kt(Qt_ut_ut,Qt_ut_xt,q_ut)
            if t==self.T-1:
               self.Kt_kt_end_point(x_des=x_des)
            else:
               self.Kt_kt(Qt_ut_ut,Qt_ut_xt,q_ut)

            self.Vt_vt(self.Kt,self.kt,Qt_xt_xt,Qt_xt_ut,Qt_ut_xt,Qt_ut_ut,q_xt,q_ut)
            self.Kt_vector.append(self.Kt)
            self.kt_vector.append(self.kt)

            self.Vt_1=self.Vt
            self.vt_1=self.vt

    #2.reach end of point with lessest force and reach neastest points on path
    #create random path betweed start_point:=:end_point
    def random_path(self,startpoint=[],endpoint=[],T=10,scale=20):
        x_start=startpoint[0]
        y_start=startpoint[1]
        x_end=endpoint[0]
        y_end=endpoint[1]

        dx=(x_end-x_start)/(T+1)
        dy=(y_end-y_start)/(T+1)
        result=[]
        # result.append([x_start,y_start])
        x=[]
        x.append(x_start)
        y=[]
        y.append(y_start)
        for  i in range(T):
               temp_x=x_start+dx*(i+1)+np.random.normal(loc=0,scale=scale,size=1)
               temp_y=y_start+dy*(i+1)+np.random.normal(loc=0,scale=scale,size=1)
               x.append(temp_x)
               y.append(temp_y)
               result.append([temp_x,temp_y])
        x.append(x_end)
        y.append(y_end)
        # result.append([x_end,y_end])
        import matplotlib.pyplot as plt
        plt.scatter(x,y)
        plt.show()
        return result

    def Ct_init_neast_point(self,T=100,startPoint=[],endpoint=[],scale=20):
        # self.Ft_init()
        x_des=self.random_path(startpoint=startPoint,endpoint=endpoint,T=T,scale=scale)
        temp_Ct=np.zeros((self.s_dim+self.a_dim,self.s_dim+self.a_dim))
        # temp_ct=np.zeros((self.s_dim+self.a_dim,1))
        temp_Ct[self.s_dim][self.s_dim]=1.0
        temp_Ct[self.s_dim+1][self.s_dim+1]=1.0
        temp_Ct[0][0]=1.0
        temp_Ct[1][1]=1.0

        self.T=T+1
        self.Ct=[]
        temp_Ct_start=temp_Ct.copy() #
        temp_Ct_start[0][0]=0.0
        temp_Ct_start[1][1]=0.0
        self.Ct.append(temp_Ct_start)   #start_point,only need action small,no need path nearest
        self.ct=[]
        self.ct.append(np.zeros((self.s_dim+self.a_dim,1))) #start_point
        for e in  x_des:
            # e=list(e)
            self.Ct.append(temp_Ct)
            self.ct.append(tf.reshape(-1.0*np.array([e[0],e[1],0.0,0.0,0.0,0.0],dtype=float),(-1,1)))
        # print(self.ct)
        # print(self.Ct)
        # exit()
        if len(self.Ct)!=self.T:
           print("Ct length must be same as T")

    #-----------------------------
    def forward_xt_ut(self,x_start):
        self.Kt_vector.reverse()
        self.kt_vector.reverse()
        xt=x_start
        self.x_vector.append(xt)

        for i in range(self.T):
            ut=tf.matmul(self.Kt_vector[i],xt)+self.kt_vector[i]
            xt=self.f(xt,ut)
            self.x_vector.append(xt)
            self.ut_vector.append(ut)
        # print(self.ut_vector)
        # print(self.x_vector)

    def train_neast_point(self):
        self.Ft_init()
        self.Ct_init_neast_point(T=100,startPoint=[2000.0,1000.0],endpoint=[0.0,0.0],scale=30)
        self.back_Kt_end_point(np.array([[0.0],[0.0]]))


    def train_end_point(self):
        self.Ft_init()
        self.Ct_init_end_point(100)
        self.back_Kt_end_point(np.array([[200.0],[300.0]])) #endpoint

    #----------------------------
    def trace_map(self):
        import matplotlib.pyplot as plt
        fig=plt.figure()
        ut_x=[e[0] for  e  in self.ut_vector]
        ut_y=[e[1] for  e  in self.ut_vector]
        x1=range(len(self.ut_vector))
        ax1=fig.add_subplot(211) #2*2的图形 在第一个位置
        ax1.plot(x1,ut_x)
        ax1.plot(x1,ut_y)
        squares=[list(e[1]) for e  in self.x_vector]
        x=[list(e[0]) for e  in self.x_vector]
        ax2=fig.add_subplot(212)
        ax2.plot(x,squares)
        ax2.scatter(x,squares)
        plt.show()

#1、x_des point   navigate
lqr=LQR()
lqr.train_end_point()
lqr.forward_xt_ut(np.array([[2000.0],[1000.0],[10],[-5]]))
lqr.trace_map()

#2、neast path    navigate
lqr=LQR()
lqr.train_neast_point()
lqr.forward_xt_ut(np.array([[2000.0],[1000.0],[10],[-5]]))
lqr.trace_map()

