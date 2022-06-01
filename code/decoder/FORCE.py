import sys
import os
#sys.path.append("./decoder/")
import numpy as np
from numpy.linalg import multi_dot
import matplotlib.pyplot as plt
from metrics import *
from scipy.sparse import csr_matrix
#sys.path.append("../code/decoder/")
from decoder.DecoderBase import DecoderBase

class FORCEDecoder(DecoderBase):
    def __init__(self,dt=15,num_channel=96,name="FORCE"):
        #self.tau = 75 #ms
        #self.dt = self.tau/5 #ms
        self.name       = name
        self.dt         = dt#ms
        self.tau        = dt*5#ms
        self.learning_dt = 2*self.dt #ms
        self.position_normalization = 100
        self.velocity_normalization = 800
        self.N = 1200# the number of RNN neurons
        self.n = 120 # the number of connections from one neuron to another
        self.g = 0.5 # global scaling of internal connections
        self.h = 0.5 # global scahling of inputs
        self.I = num_channel
        self.i = 12  # the number of inputs to each RNN neuron
        self.M = 4
        self.m = 2   #the number of outputs fed back to each RNN neuron
        self.sigma_bias = 0.025 # the standard deviation of the bias current to each RNN neuron
        self.alpha      = 100
        self._activation = np.tanh
        self.J   = self._sparse_W_init(self.N,self.N,self.n,0,1/self.n)# dimension=(NxN), the weights of the recurrent connections of the network, n<<N
        self.W_F = self._sparse_W_init(self.N,self.M,self.m,0,1/self.m)#np.random.normal(0,1/m) # dimension = (NxM), m<<M
        self.W_I = self._sparse_W_init(self.N,self.I,self.i,0,1/self.i)#np.random.normal(0,1/i) # dimension=(NxI), i<<I
        self.W_O = np.zeros((self.N,self.M)) # dimension = (N,M)
        self.P = (1/self.alpha)* np.identity(self.N)# dimension = (NxN)
        self.x = np.random.normal(0,1/self.N,size=(self.N,1)) # how they initialize this??????
        self.bias = np.random.normal(0,self.sigma_bias,size=(self.N,1))# dimension = (N,1)

    def reset(self):
        pass

    def save(self,path='./'):
        name = os.path.join(path,self.name)
        print("save ",name)

        np.savez(name,J      = self.J
                     ,W_F    = self.W_F
                     ,W_I    = self.W_I
                     ,W_O    = self.W_O
                     ,P      = self.P
                     ,bias   = self.bias
                     ,g      = self.g
                     ,h      = self.h
                     ,dt     = self.dt
                     ,tau    = self.tau
                     ,x      = self.x)

    def load(self,file_path):
        L = np.load(file_path,allow_pickle=True)
        self.J   =csr_matrix(L['J'].item())
        self.W_F =csr_matrix(L['W_F'].item())
        self.W_I =csr_matrix(L['W_I'].item())
        self.I = self.W_I.shape[1]
        self.W_O =L['W_O']
        self.P   =L['P']
        self.bias=L['bias']
        self.g   =L['g']
        self.h   =L['h']
        self.dt  =L['dt']
        self.tau =L['tau']
        self.x   =L['x']

    def _sparse_W_init(self,N,M,m,mu=0,std=1):
        ## N rows, M colums, m randomly choosen non-zeros elements in each row
        W = np.zeros((N,M))
        for ii in range(N):
            idx = np.random.choice(M, size=(m), replace=False)
            W[ii][idx] = np.random.normal(mu,1/m,size=m)
        return csr_matrix(W)

    def learnFromData(self,data):
        assert self.dt == data.dt
        extraBins = data.extraBins
        for trialIdx in data.trainingTrials:#data.centerOutTrials:
            ttrial = 0
            fin,fout = data.df['binSpike'][trialIdx][extraBins:], np.concatenate([data.df['binCursorPos'][trialIdx][extraBins:]
                ,data.df['binCursorVel'][trialIdx][extraBins:]],axis=1)
            fout=fout.T

            Zs = self.decode(fin,fout,'train',trialIdx=trialIdx)

    def decode(self,fin,fout=None,mode='test',trialIdx=np.inf):
        fin = fin[:,:self.I]

        if fout is not None:
            fout[0:2] = fout[0:2] / self.position_normalization
            fout[2:4] = fout[2:4] / self.velocity_normalization

        DTRLS   = self.learning_dt/self.dt
        Tinit   = 10
        dt      = self.dt
        taux    = self.tau

        Jr      = self.g * self.J
        uFout   = self.W_F
        uFin    = self.h * self.W_I

        P       = self.P
        W_O     = self.W_O

        x_learn = self.x
        r       = np.tanh(x_learn)
        Z       = self.W_O.T@r

        TTrial  = len(fin)
        Fin     = self.h*self.W_I@fin.T

        Zs = np.zeros((self.M,TTrial))# to accumulate generated outputs, for plotting
        for ttrial in range(TTrial):
            x_learn = x_learn + (dt/taux) * (-x_learn + Jr@r + uFout@Z +Fin[:,ttrial:ttrial+1]+ self.bias) #
            #x_learn = x_learn + (dt/taux) * (-x_learn + np.dot(self.g*self.J , r) + np.dot(self.W_F , Z) + Fin[:,ttrial:ttrial+1]+ self.bias)
            if mode=='train': # add Gaussian noise during training
                x_learn+= np.random.normal(0,0.01,size=x_learn.shape)

            r = np.tanh(x_learn)
            r[1]=1 ## force to be 1
            Z = np.dot(W_O.T , r)

            Zs[0:2,ttrial] = np.squeeze(Z[0:2])*self.position_normalization
            Zs[2:4,ttrial] = np.squeeze(Z[2:4])*self.velocity_normalization

            # do RLS in training mode
            if mode=='train' and (np.random.rand() < 1/DTRLS) and  (trialIdx > Tinit): #
                xP  = np.dot(P , r)
                k   = np.linalg.lstsq((1+ np.dot(r.T,xP)),xP.T)[0]
                P   = P - np.dot(xP,k)
                W_O = W_O - np.dot((Z - fout[:,ttrial:ttrial+1]),k).T

        self.x  = x_learn
        self.P  = P
        self.W_O= W_O
        return Zs


    def decodeFromData(self,data,spike,trialIdx):
        fin = spike
        Zs=self.decode(fin)
        ## smooth trajectory
        decoded_P =  np.zeros((2,Zs.shape[1]))
        decoded_P[:,0] = data.df['binCursorPos'][trialIdx][data.extraBins]
        for ii in range(1,(decoded_P).shape[1]):
            decoded_P[:,ii] = 0.05*Zs[0:2,ii]+0.95*(decoded_P[:,ii-1] + Zs[2:4,ii]*self.dt/1000) #
        return decoded_P.T
