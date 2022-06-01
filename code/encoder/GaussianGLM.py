import numpy as np
import pandas as pd

class GaussianGLM(object):
#################
# Linear neural generator with least-mean-square, and 4 extra timestamp
# Y = X L
# Y: neural data (Nx192)
# X: kinematics (Nx(2*k+1))
# L: transformation ((2*k+1)x192)
# L = Y dot inverse (X)
#################

    def __init__(self,name="GGLM",shape='v'):
        self.name = name
        self.shape=shape
        self.numChannels = None
    def load(self,L):
        self.L = L
    def save(self,name='PPWF'):
        np.savez(name,L=self.L)
    def train(self,data):
        numChannels = data.numChannels
        extraBins = data.extraBins
        shape = self.shape
        X = []
        for trial in data.trainingTrials:
            V = data.df['binHandVel'][trial]
            P = data.df['binHandPos'][trial][1:]
            V = np.hstack([V[ii:len(V)-extraBins+ii] for ii in range(extraBins+1)])
            P = np.hstack([P[ii:len(P)-extraBins+ii] for ii in range(extraBins+1)])
            if shape=='vp':
                x = np.hstack([V,P])
            elif shape =='v':
                x = V
            X.append(x)
        X = np.vstack(X)
        X = np.hstack( [ X, np.ones([X.shape[0],1]) ] )

        Y = data.df['binSpike']
        Y = np.vstack([Y[trial][extraBins:] for trial in data.trainingTrials])

        L = np.linalg.pinv(X).dot(Y)
        self.L = L[:numChannels]
        self.dt = data.dt
        self.extraBins = data.extraBins
        self.numChannels = numChannels
        print("Spike.shape:",Y.shape)
        print("Vel.shape:",X.shape)
        print("L.shape:",L.shape)


    def encode(self,V,P):
        L = self.L
        shape = self.shape
        extraBins = self.extraBins

        V = np.hstack([V[ii:len(V)-extraBins+ii] for ii in range(extraBins+1)])
        P = np.hstack([P[ii:len(P)-extraBins+ii] for ii in range(extraBins+1)])
        if shape=='vp':
            X = np.hstack([V,P])
        elif shape =='v':
            X = V

        X = np.hstack([X, np.ones([X.shape[0],1]) ])

        neural_output = X.dot(L)
        neural_output[neural_output<0] = 0
        return neural_output

    def encodeFromData(self,data,trial):
        v = data.df['binHandVel'][trial]
        p = data.df['binHandPos'][trial][1:]
        return self.encode(v,p)


## another way to implement this function is to use statsmodels.api

#import statsmodels.api as sm
#X_1 = sm.add_constant(X)
#GL_GLM = sm.GLM(Y[:,0],X_1,family=sm.families.Gaussian())
#GL_results = GL_GLM.fit()
#X_pred = GL_results.predict(X_1)
#print(get_rho(X_pred,PPOLE.encode(X)[:,0]))
#print(np.mean(X_pred-PPOLE.encode(X)[:,0]))
#print(GL_results.summary())
