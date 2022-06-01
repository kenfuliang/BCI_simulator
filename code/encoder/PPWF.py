import numpy as np
import pandas as pd

class PPWFEncoder(object):
#################
# Linear neural generator with least-mean-square, and 4 extra timestamp
# Y = X L
# Y: neural data (Nx192)
# X: kinematics (Nx(2*k+1))
# L: transformation ((2*k+1)x192)
# L = Y dot inverse (X)
#################

    def __init__(self,name="PPWF"):
        self.name = name
    def load(self,L):
        self.L = L
    def save(self,name='PPWF'):
        np.savez(name,L=self.L)
    def train(self,data,numChannels=192):
        X = np.vstack(data.df['binHandVel'][data.trainingTrials])
        Y = np.vstack(data.df['binSpike'][data.trainingTrials])

        Y = Y[data.extraBins:]

        tmp=[]
        for ii in range(data.extraBins):
            if(ii==0):
                tmp.append(X[data.extraBins:])
            else:
                tmp.append(X[data.extraBins-ii:-ii])
        X = np.hstack(tmp)

        #X = np.hstack([ X[4-ii:ii] for ii in range(data.extraBins)      ])  #X[4:],X[3:-1],X[2:-2],X[1:-3],X[0:-4]
        X = np.hstack([X,np.ones((X.shape[0],1))])

        L = np.linalg.pinv(X).dot(Y)
        self.L = L[:numChannels]
        self.dt = data.dt
        self.extraBins = data.extraBins
        print("Spike.shape:",Y.shape)
        print("Vel.shape:",X.shape)
        print("L.shape:",L.shape)


    def encode(self,v):
        L = self.L
        X = v

        #X = np.hstack([X[4:],X[3:-1],X[2:-2],X[1:-3],X[0:-4]])
        tmp=[]
        for ii in range(self.extraBins):
            if(ii==0):
                tmp.append(X[self.extraBins:])
            else:
                tmp.append(X[self.extraBins-ii:-ii])
        X = np.hstack(tmp)

        X = np.hstack((X,np.ones((X.shape[0],1))))
        neural_output = X.dot(L)
        neural_output[neural_output<0] = 0
        return neural_output

    def encodeFromData(self,data,trial):
        v = data.df['binHandVel'][trial]
        return self.encode(v)


## another way to implement this function is to use statsmodels.api

#import statsmodels.api as sm
#X_1 = sm.add_constant(X)
#GL_GLM = sm.GLM(Y[:,0],X_1,family=sm.families.Gaussian())
#GL_results = GL_GLM.fit()
#X_pred = GL_results.predict(X_1)
#print(get_rho(X_pred,PPOLE.encode(X)[:,0]))
#print(np.mean(X_pred-PPOLE.encode(X)[:,0]))
#print(GL_results.summary())
