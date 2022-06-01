import matplotlib.pyplot as plt
import numpy as np
from metrics import mean_square_error
import sys
#sys.path.append("../code/decoder/")
from decoder.DecoderBase import DecoderBase


class OLEDecoder(DecoderBase):
####################
# X: kinematics (Nx2)
# Y: neural data (Nx(192+1))
# L: transformation (2x(192+1))
# X = YxL
# L = inverse(Y) dot X
####################

    def __init__(self, name = "OLEDecoder"):
        self.name = name

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
        print("Train ", self.name)
        if(data.numTrain==0):
            trainingTrials = np.arange(0,data.numTrial)
        else:
            trainingTrials = data.trainingTrials

        #X = np.vstack(data.df['binHandVel'][trainingTrials])
        X = data.next_train_batch(replace=False, randomSelect=False)
        X = np.squeeze(np.concatenate((X['p'],X['v']),axis=2))

        Y = np.vstack(data.df['binSpike'][trainingTrials])[:,0:numChannels]

        self.train(X,Y)


    def train(self,X,Y):

        Y = np.hstack((Y,np.ones((Y.shape[0],1))))
        L = np.linalg.pinv(Y).dot(X)
        self.L = L
        print("Spike.shape:",Y.shape)
        print("Vel.shape:",X.shape)
        print("L.shape:",L.shape)


    def decode(self,Y):
###################
# X = YxL
###################
        L = self.L
        Y = np.hstack((Y,np.ones((Y.shape[0],1))))
        X = Y.dot(L)
        self.decodeResult = X
        return self.decodeResult

    def decodeFromData(self,data,spike,trial):
        return self.decode(spike)
