import numpy as np
import pandas as pd

class PPOLEEncoder(object):
#################
# Linear neural generator with least-mean-square
# Y = X L
# Y: neural data (Nx192)
# X: kinematics (Nx(2+1))
# L: transformation ((2+1)x192)
# L = Y dot inverse (X)
#################
    def __init__(self,name="PPOLE",shape='v'):
        self.name = name
        self.numChannels = None
        self.shape = shape

    def train(self,data):
        print("Train ",self.name)
        self.numChannels = data.numChannels
        if('v' in self.shape):
            X = np.vstack(data.df['binHandVel'][data.trainingTrials])
        if('p' in self.shape):
            X = np.hstack([X, np.vstack([data.df['binHandPos'][trial][1:] for trial in data.trainingTrials])])

        X = np.hstack((X,np.ones((X.shape[0],1))))
        Y = np.vstack(data.df['binSpike'][data.trainingTrials])

        L = np.linalg.pinv(X).dot(Y)
        self.L = L[:,:self.numChannels]
        self.dt = data.dt

    def encode(self,v,p):
        if(self.shape == 'v'):
            x = v
        elif(self.shape=='vp'):
            x = np.hstack([v,p])
        x = np.hstack((x,np.ones((x.shape[0],1))))
        neural_output = x.dot(self.L)
        neural_output[neural_output<0] = 0
        return neural_output

    def encodeFromData(self,data,trial):
        v = data.df['binHandVel'][trial]
        p = data.df['binHandPos'][trial][1:]
        return self.encode(v,p)

    def load(self,L):
        self.L = L
