import numpy as np
import os

def tuning(thetas,fr):
        A  = np.column_stack((np.ones(thetas.shape[0]),np.sin(thetas), np.cos(thetas)))
        ks = np.linalg.pinv(A).dot(fr)
        pd = np.arctan2(ks[1],ks[2])
        c0 = ks[0]
        c1 = ks[1]/np.sin(pd)
        return (c0,c1,pd)

class PPVTEncoder(object):
    def __init__(self, fr_gain=1/450,name="PPVTEncoder"):
        self.name = name
        self.input_shape = (2,1)
        self.encoder_input_shape = 'v'
        self.fr_gain = fr_gain

    def save(self,path='../saved/encoder/JenkinsC/'):
        np.savez(os.path.join(path,self.name),dt=self.dt,pds=self.pds,c0s=self.c0s,c1s=self.c1s)

    def load(self,decoder_path):
        L = np.load(decoder_path)
        self.dt  = L['dt']
        self.pds = L['pds']
        self.c0s = L['c0s']
        self.c1s = L['c1s']

    def train(self,data):
        numChannels = data.numChannels
        numTargets = len(data.uniqueTarget)
        dt = data.dt

        c0s = np.zeros(numChannels)
        c1s = np.zeros(numChannels)
        pds = np.zeros(numChannels)
        window = range(249,500)
        for neuron_idx in np.arange(numChannels):

            trialFRs = np.array([data.df['binSpike'][i][int(min(window)/dt):int(max(window)/dt),neuron_idx].sum() for i in np.arange(data.numTrial)])
            #if(neuron_idx<96):
            #    trialFRs = np.array([data.df['spikeRaster'][i][window,neuron_idx].nnz for i in np.arange(data.numTrial)])
            #elif(neuron_idx<192):
            #    trialFRs = np.array([data.df['spikeRaster2'][i][window,neuron_idx-96].nnz for i in np.arange(data.numTrial)])
            #else:
            #    raise ValueError("numChannels should less than 192")
            meanFRs = np.zeros(numTargets)
            thetas  = np.zeros(numTargets)
            for target_idx in range(numTargets):
                meanFRs[target_idx] = np.mean(trialFRs[np.intersect1d(data.TrialsOfTargets[tuple(data.uniqueTarget[target_idx])],data.trainingTrials)])
                thetas[target_idx] = np.arctan2(data.uniqueTarget[target_idx][1],data.uniqueTarget[target_idx][0])
            meanFRs = meanFRs/(np.max(window)-np.min(window))*1000
            thetas = thetas + 2*np.pi*(thetas<0)
            c0, c1, pd = tuning(thetas,meanFRs)
            pds[neuron_idx]=pd
            c1s[neuron_idx]=c1
            c0s[neuron_idx]=c0
        self.pds = pds.reshape(numChannels,1)
        self.c1s = c1s.reshape(numChannels,1)
        self.c0s = c0s.reshape(numChannels,1)
        self.dt  = data.dt

    def encodeFromData(self,data,trial):
        v = data.df['binHandVel'][trial]
        encoder_input = {}
        encoder_input['v'] = v
        return self.encode(encoder_input)

#    def encode(self,v):
#        dt  = self.dt
#        pds = self.pds
#        c0s = self.c0s
#        c1s = self.c1s
#
#        kinematics = np.arctan2(v[:,1],v[:,0])
#        theta_dif = pds.reshape(-1,1)-kinematics
#        neural_output = c0s+c1s*np.cos(theta_dif)*np.sqrt(v[:,1]**2+v[:,0]**2)/500
#        neural_output=neural_output/1000*dt
#        neural_output=neural_output.T
#        neural_output[neural_output<0]=0
#        return neural_output

    def predict(self,encoder_input):
        v   = encoder_input.reshape(-1,2)
        dt  = self.dt
        pds = self.pds
        c0s = self.c0s
        c1s = self.c1s
        fr_gain = self.fr_gain

        kinematics = np.arctan2(v[:,1],v[:,0])
        theta_dif = pds.reshape(-1,1)-kinematics
        neural_output = c0s+c1s*np.cos(theta_dif)*np.sqrt(v[:,1]**2+v[:,0]**2)*fr_gain
        neural_output=neural_output/1000*dt
        neural_output=neural_output.T
        neural_output[neural_output<0]=0
        neural_output[np.isnan(neural_output)]=0
        return neural_output



    def encode(self,encoder_input):
        v   = encoder_input['v']
        dt  = self.dt
        pds = self.pds
        c0s = self.c0s
        c1s = self.c1s
        fr_gain = self.fr_gain

        kinematics = np.arctan2(v[:,1],v[:,0])
        theta_dif = pds.reshape(-1,1)-kinematics
        neural_output = c0s+c1s*np.cos(theta_dif)*np.sqrt(v[:,1]**2+v[:,0]**2)*fr_gain
        neural_output=neural_output/1000*dt
        neural_output=neural_output.T
        neural_output[neural_output<0]=0
        return neural_output




#import numpy as np
#import pandas as pd
##from utils import *
#
#class PPVTEncoder(object):
#    def train(self,data,numChannels=192):
#        X_bin = np.vstack([data.df['binVelocity'][trial][:] for trial in data.trainingTrials])
#        X_bin = np.hstack((X_bin,np.ones((X_bin.shape[0],1))))
#        Y_bin = pd.Series()
#        Y_bin = data.df['binSpike'][data.trainingTrials]
#        Y_bin = np.vstack(Y_bin)
#        L = np.linalg.pinv(X_bin).dot(Y_bin)
#        self.L = L[:,:numChannels]
#        self.dt = data.dt
#
#    def encode(self,v):
#        x_bin = v
#        x_bin = np.hstack((x_bin,np.ones((x_bin.shape[0],1))))
#        neural_output = x_bin.dot(self.L)
#        neural_output[neural_output<0] = 0
#        return neural_output
#
#    def encodeFromData(self,data,trial):
#        v = data.df['binVelocity'][trial]
#        return self.encode(v)
#
#    def load(self,L):
#        self.L = L
