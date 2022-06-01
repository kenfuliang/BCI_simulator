import numpy as np
import pandas as pd
import statsmodels.api as sm

class PoissonGLM(object):
    def __init__(self,name="PGLM",shape='v'):
        self.name=name
        self.numChannels = None
        self.shape = shape

    def train(self,data):
        print("Train",self.name)
        extraBins = data.extraBins
        numChannels = data.numChannels
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

        print("X,Y shape:",X.shape,Y.shape)
        GLM_list = []
        for neuronIdx in range(numChannels):
            Poisson_GLM = sm.GLM(Y[:,neuronIdx],X,family=sm.families.Poisson())
            Poisson_results = Poisson_GLM.fit()
            GLM_list.append(Poisson_results)
        self.GLM_list = GLM_list
        self.extraBins = extraBins
        self.numChannels = numChannels

    def encode(self,V,P):
        extraBins = self.extraBins
        shape = self.shape
        V = np.hstack([V[ii:len(V)-extraBins+ii] for ii in range(extraBins+1)])
        P = np.hstack([P[ii:len(P)-extraBins+ii] for ii in range(extraBins+1)])
        if shape=='vp':
            X = np.hstack([V,P])
        elif shape =='v':
            X = V

        X = np.hstack([X, np.ones([X.shape[0],1]) ])

        ## Encoding
        neural_output_list = []
        for neuronIdx in range(self.numChannels):
            neural_output_list.append(self.GLM_list[neuronIdx].predict(X))
        neural_output = np.vstack(neural_output_list)
        neural_output[neural_output<0] = 0
        return neural_output.T

    def encodeFromData(self,data,trial):
        v = data.df['binHandVel'][trial]
        p = data.df['binHandPos'][trial][1:]
        return self.encode(v,p)

