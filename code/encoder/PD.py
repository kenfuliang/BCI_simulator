import numpy as np
import encoder

def tuning(thetas,fr):
        A  = np.column_stack((np.ones(thetas.shape[0]),np.sin(thetas), np.cos(thetas)))
        ks = np.linalg.pinv(A).dot(fr)
        pd = np.arctan2(ks[1],ks[2])
        c0 = ks[0]
        c1 = ks[1]/np.sin(pd)
        return (c0,c1,pd)

class PDEncoder(object):
    def __init__(self, name="PDEncoder", extraBins=-1):
        self.name = name
        self.extraBins = extraBins

    def save(self,name="PD"):
        np.savez(name,dt=self.dt,pds=self.pds,c0s=self.c0s,c1s=self.c1s)

    def load(self,decoder_path):
        L = np.load(decoder_path)
        self.dt  = L['dt']
        self.pds = L['pds']
        self.c0s = L['c0s']
        self.c1s = L['c1s']

    def train(self,data,targets=None):
        numChannels = data.numChannels
        extraBins   = self.extraBins
        dt = data.dt
        uniqueTarget = targets if targets is not None else data.uniqueTarget
        c0s = np.zeros(numChannels)
        c1s = np.zeros(numChannels)
        c2s = np.zeros(numChannels)
        pds = np.zeros(numChannels)
        window = range(249,500)
        if(extraBins==-1): # take averaged firing rates, and thetas as tuning model input.
            print("extraBins==-1")
            for neuron_idx in np.arange(numChannels):
                trialFRs = np.array([data.df['binSpike'][i][int(min(window)/dt):int(max(window)/dt),neuron_idx].sum() for i in np.arange(data.numTrial)])
                meanFRs = np.zeros(uniqueTarget.shape[0])
                thetas  = np.zeros(uniqueTarget.shape[0])
                for target_idx in range(uniqueTarget.shape[0]):
                    meanFRs[target_idx] = np.mean(trialFRs[np.intersect1d(data.TrialsOfTargets[target_idx],data.trainingTrials)])
                    thetas[target_idx] = np.arctan2(uniqueTarget[target_idx][1],uniqueTarget[target_idx][0])
                meanFRs = meanFRs/(np.max(window)-np.min(window))*1000
                thetas = thetas + 2*np.pi*(thetas<0)
                c0, c1, pd = tuning(thetas,meanFRs)
                pds[neuron_idx]=pd
                c1s[neuron_idx]=c1
                c0s[neuron_idx]=c0
        #elif(extraBins==0):
        #    for neuron_idx in np.arange(numChannels):
        #        frs = np.concatenate(data.df['binSpike'][data.trainingTrials])[:,neuron_idx]/data.dt*1000   #Nx1
        #        V   = np.concatenate(data.df['binHandVel'][data.trainingTrials])    #Nx2
        #        thetas = np.arctan2(V[:,1],V[:,0])  #Nx1
        #        c0,c1,pd = tuning(thetas,frs)
        #        pds[neuron_idx]=pd
        #        c1s[neuron_idx]=c1
        #        c0s[neuron_idx]=c0

#        else:
#            for neuron_idx in range(numChannels):
#                v_in = []
#                fr_in = []
#                for trial in data.trainingTrials:
#                    v = data.df['binHandVel'][trial]
#                    n = data.df['binSpike'][trial][:,neuron_idx]
#                    v_in.append( np.hstack([v[1:],v[0:-1]]) )
#                    fr_in.append(n[1:])
#                fr_in = np.concatenate(fr_in)/data.dt*1000
#                v_in = np.concatenate(v_in)
#                thetas = np.vstack( [np.arctan2(v_in[:,1],v_in[:,0]), np.arctan2(v_in[:,3],v_in[:,2])] ).T
#                c0,c1,c2,pd = tuning(thetas,fr_in)
#
#                pds[neuron_idx]=pd
#                c2s[neuron_idx]=c2
#                c1s[neuron_idx]=c1
#                c0s[neuron_idx]=c0


        self.pds = pds.reshape(numChannels,1)
#        self.c2s = c2s.reshape(numChannels,1)
        self.c1s = c1s.reshape(numChannels,1)
        self.c0s = c0s.reshape(numChannels,1)
        self.dt  = data.dt

    def encodeFromData(self,data,trial):
        v = data.df['binHandVel'][trial]
        return self.encode(v)

    def encode(self,encoder_input):
        v   = encoder_input
        dt  = self.dt
        pds = self.pds
        c0s = self.c0s
        c1s = self.c1s
        #c2s = self.c2s

        theta1 = np.arctan2(v[:,1],v[:,0])
        #theta2 = np.arctan2(v[:,3],v[:,2])

        theta_dif1 = pds.reshape(-1,1)-theta1
        #theta_dif2 = pds.reshape(-1,1)-theta2

        #neural_output = c0s+c1s*np.cos(theta_dif1) + c2s*np.cos(theta_dif2)
        neural_output = c0s+c1s*np.cos(theta_dif1)
        neural_output=neural_output/1000*dt
        neural_output=neural_output.T
        neural_output[neural_output<0]=0
        #neural_output = np.log(1+np.exp(neural_output))
        return neural_output


