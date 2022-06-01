import sys
sys.path.append("./decoder/")
import numpy as np
import matplotlib.pyplot as plt
from metrics import mean_square_error
#sys.path.append("../code/decoder/")
from decoder.DecoderBase import DecoderBase

class WFDecoder(DecoderBase):
####################
# Wiener filter with k timestamps
# X: kinematics (Nx2)
# Y: neural data (Nx(192*k+1))
# L: transformation ((192*k+1)x2)
# X = YxL
# L = inverse(Y) dot X
####################

    def __init__(self, name="WFDecoder"):
        self.name = name
        self.history = 3
        self.numChannels = 192
        self.spike_history = np.zeros((self.history+1,self.numChannels))

    def loadFromMat(self,mat):
        L = mat[0][0]['Lw'][0][0]['signals'][0][0]['values']
        self.L = L.T

    def load(self,decoder_path):
        L = np.load(decoder_path)
        self.L = L['L']

    def save(self,path='./',name=None):
        name= name if (name is not None) else self.name
        np.savez(path+name,L=self.L)

    def trainFromData(self,data,numChannels=192):
        history = self.history

        if(data.numTrain==0):
            trainingTrials = np.arange(0,data.numTrial)
        else:
            trainingTrials = data.trainingTrials


        #X = data.next_train_batch(replace=False, randomSelect=False)

        #X = np.squeeze(np.concatenate((X['p'],X['v']),axis=2))
        X = np.concatenate( [(data.df['binCursorPos'][trial], data.df['binCursorVel'][trial]) for trial in trainingTrials], axis=1 )
        X = np.concatenate(X,axis=1)

        Y = np.concatenate([data.df['binSpike'][trial] for trial in trainingTrials])
        #Y = np.vstack(data.df['binSpike'][trainingTrials])[:,0:numChannels]

        self.train(X,Y)


    def train(self, X, Y):
        print("Train ",self.name)
        history = self.history
        X = X[history:]
        lenY = len(Y)
        Y = np.hstack([Y[ii:lenY-history+ii] for ii in reversed(range(history+1))   ])
        Y = np.hstack([Y,np.ones((Y.shape[0],1))])
        L = np.linalg.pinv(Y).dot(X)
        print("Spike.shape:",Y.shape)
        print("Vel.shape:",X.shape)
        print("L.shape:",L.shape)

        self.L = L


    def decode(self,spikes):
###########
## this function will store previous input spikes as spike history, then decode the current spikes based on the previous.
###########
        L = self.L
        self.spike_history = np.vstack((self.spike_history,spikes))[-self.history-1:]
        Y = np.hstack([self.spike_history[ii:ii+1] for ii in reversed(range(self.history+1))])
        Y = np.concatenate((Y,np.ones((Y.shape[0],1))),axis=1)
        X = Y @ L
        return X
    def decodeFromData(self,data,spike,trial):
        decodes = np.vstack([self.decode(spike_count) for spike_count in spike])
        return decodes


#    def decode(self,Y):
#        L = self.L
#        bins = int((L.shape[0]-1)/self.numChannels)
#        lenY = len(Y)
#        Y = np.hstack([Y[ii:lenY-bins+ii+1] for ii in reversed(range(bins))   ])
#        Y = np.hstack([Y,np.ones((Y.shape[0],1))])
#        X = Y.dot(L)
#        self.decodeResult = X
#        return self.decodeResult
#
#    def decodeFromData(self,data,spike,trial):
#        initS = np.concatenate((data.df['binCursorPos'][trial][data.extraBins:data.extraBins+self.history],\
#                                data.df['binCursorVel'][trial][data.extraBins:data.extraBins+self.history]),axis=1)
#        decodes = self.decode(spike)
#        return np.vstack((initS,decodes))
#
#
