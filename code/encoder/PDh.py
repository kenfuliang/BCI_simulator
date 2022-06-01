import numpy as np
import encoder 

def tuning(thetas,fr):
        A  = np.column_stack((np.ones(thetas.shape[0]),np.sin(thetas), np.cos(thetas)))
        ks = np.linalg.pinv(A).dot(fr)
        pd = np.arctan2(ks[1],ks[2])
        c0 = ks[0]
        c1 = ks[1]/np.sin(pd)
        return (c0,c1,pd)

class PDEncoder_h(object):
    def __init__(self, name="PDEncoder"):
        self.name = name

    def save(self,name="PD"):
        np.savez(name,dt=self.dt,pds=self.pds,c0s=self.c0s,c1s=self.c1s)

    def load(self,L):
        self.dt  = L['dt']
        self.pds = L['pds']
        self.c0s = L['c0s']
        self.c1s = L['c1s']

    def train(self,data):
        numChannels = data.numChannels
        dt = data.dt
        c0s = np.zeros(numChannels)
        c1s = np.zeros(numChannels)
        pds = np.zeros(numChannels)
        window = range(249,500)
        
        for neuron_idx in np.arange(numChannels):
            #frs = np.concatenate(data.df['binSpike'][data.trainingTrials])[:,0]
            frs = np.concatenate(data.df['binSpike'][data.trainingTrials])[:,neuron_idx]/data.dt*1000
            V   = np.concatenate(data.df['binHandVel'][data.trainingTrials])
            thetas = np.arctan2(V[:,1],V[:,0])
            c0,c1,pd = tuning(thetas,frs)
            #trialFRs = np.array([data.df['binSpike'][i][int(min(window)/dt):int(max(window)/dt),neuron_idx].sum() for i in np.arange(data.numTrial)])
            #meanFRs = np.zeros(data.uniqueTarget.shape[0])
            #thetas  = np.zeros(data.uniqueTarget.shape[0])
            #for target_idx in range(data.uniqueTarget.shape[0]):
            #    meanFRs[target_idx] = np.mean(trialFRs[np.intersect1d(data.TrialsOfTargets[target_idx],data.trainingTrials)])
            #    thetas[target_idx] = np.arctan2(data.uniqueTarget[target_idx][1],data.uniqueTarget[target_idx][0])
            #meanFRs = meanFRs/(np.max(window)-np.min(window))*1000
            #thetas = thetas + 2*np.pi*(thetas<0)
            #c0, c1, pd = tuning(thetas,meanFRs)
            pds[neuron_idx]=pd
            c1s[neuron_idx]=c1
            c0s[neuron_idx]=c0
        self.pds = pds.reshape(numChannels,1)
        self.c1s = c1s.reshape(numChannels,1)
        self.c0s = c0s.reshape(numChannels,1)
        self.dt  = data.dt

    def encodeFromData(self,data,trial):
        v = data.df['binHandVel'][trial]
        return self.encode(v)

    def encode(self,v):
        dt  = self.dt
        pds = self.pds
        c0s = self.c0s
        c1s = self.c1s

        kinematics = np.arctan2(v[:,1],v[:,0])
        theta_dif = pds.reshape(-1,1)-kinematics
        neural_output = c0s+c1s*np.cos(theta_dif)
        neural_output=neural_output/1000*dt           
        neural_output=neural_output.T
        neural_output[neural_output<0]=0
        return neural_output


