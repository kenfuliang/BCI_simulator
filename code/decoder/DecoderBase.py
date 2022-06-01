import matplotlib.pyplot as plt
import numpy as np
from metrics import NormalizedMeanDiff
from fig_gen_util import adjust_spines
from util import binning
#from NeuralEncoder import NeuralEncoder
#from NeuralDecoder import NeuralDecoder
#from decoder.kalman import KFDecoder
#from decoder.FORCE import FORCEDecoder

class DecoderBase(object):
    def __init__(self):
        raise NotImplementedError
    def save(self):
        raise NotImplementedError
    def load(self):
        raise NotImplementedError
    def decode(self):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    #def test(self,data):
    #    neuraldecoder = NeuralDecoder()
    #    if isinstance(self,KFDecoder):
    #        neuraldecoder.decoder_type = self.type
    #        if self.type=='PVKF':
    #            neuraldecoder.alpha = 0.0
    #        else:
    #            neuraldecoder.alpha = 1.0
    #    elif isinstance(self,FORCEDecoder):
    #        neuraldecoder.decoder_type = 'FORCE'
    #        neuraldecoder.alpha = 0.95
    #    else:
    #        raise ValueError("decoder should be either KF or FORCE")
    #    neuraldecoder.setDecoder(self)
    #    neuraldecoder.refresh_rate = self.dt

    #    fig, ax = plt.subplots(1, 1, figsize=(5,5))
    #    ax.set_xlim([-150,150])
    #    ax.set_ylim([-150,150])
    #    ax.set_xticks([-100,0,100])
    #    ax.set_yticks([-100,0,100])
    #    adjust_spines(ax)



    #    for trialIdx in data.df.index:
    #        real_spike_counts = data.df['binSpike'][trialIdx]
    #        cursor_p = data.df['binCursorPos'][trialIdx]
    #        cursor_v = data.df['binCursorVel'][trialIdx]
    #        
    #        trial_length = cursor_p.shape[0]
    #        ## reset decoder position
    #        self.neuraldecoder.reset(cursor_p[0],cursor_v[0]/1000)

    #        ## record decoded results
    #        real_decoded_pos = np.empty((trial_length,2))
    #        real_decoded_vel = np.empty((trial_length,2))

    #        n = self.neuraldecoder.refresh_rate//data.dt
    #        ### start with the initial x0 as in Wu and Chestek, we get the first for free
    #        real_decoded_pos[0] = cursor_pos = cursor_p[0]
    #        real_decoded_vel[0] = cursor_vel = cursor_v[0]

    #        for tt in range(1, trial_length+1):
    #            if tt % n ==0:
    #                spike_counts = np.sum(real_spike_counts[tt-n:tt],axis=0)
    #                cursor_pos,_P,cursor_vel = neuraldecoder.decode(spike_counts)
    #            real_decoded_pos[tt:tt+1] = cursor_pos 
    #            real_decoded_vel[tt:tt+1] = cursor_vel
    #            alpha=1

    #        ax.plot(real_decoded_pos[:,0],real_decoded_pos[:,1],'r',alpha=alpha)
    #        ax.plot(cursor_p[:,0],cursor_p[:,1],'b',alpha=alpha)
    #            #MSE_pos.append(NormalizedMeanDiff(Zs,fout[:,0:2],axis=1))
    #            #MSE_vel.append(NormalizedMeanDiff(Zs[1:]-Zs[:-1],fout[1:,0:2]-fout[:-1,0:2],axis=1))
    #    #ax.set_title("(NMD pos{0:2.2f} vel{1:2.2f})".format(np.mean(MSE_pos),np.mean(MSE_vel)),size=15)




    def test(self,data):
        extraBins = data.extraBins
        #dt = data.dt
        fig, ax = plt.subplots(1, 1, figsize=(5,5))
        ax.set_xlim([-150,150])
        ax.set_ylim([-150,150])
        ax.set_xticks([-100,0,100])
        ax.set_yticks([-100,0,100])
        adjust_spines(ax)


        n = self.dt//data.dt

        MSE_pos = []
        MSE_vel = []
        
        for target in data.uniqueTarget:
            trials = np.intersect1d(data.TrialsOfTargets[tuple(target)], data.testingTrials)
            for ii,trialIdx in enumerate(trials):
                X = data.df['binSpike'][trialIdx][extraBins:]
                y = binning(np.concatenate([data.df['binCursorPos'][trialIdx][extraBins:]]),n,'first')
                y_pred = self.decodeFromData(data,X,trialIdx)
                if ii==0:
                    alpha=1
                else:
                    alpha=0.1
                ax.plot(y[:,0],y[:,1],'r',alpha=alpha)
                ax.plot(y_pred[:,0],y_pred[:,1],'b',alpha=alpha)
                MSE_pos.append(NormalizedMeanDiff(y,y_pred[:,0:2],axis=1))
                MSE_vel.append(NormalizedMeanDiff(y[1:]-y[:-1],y_pred[1:,0:2]-y_pred[:-1,0:2],axis=1))
        ax.set_title("(NMD pos{0:2.2f} vel{1:2.2f})".format(np.mean(MSE_pos),np.mean(MSE_vel)),size=15)


        #MSE_pos = []
        #MSE_vel = []
        #for trialIdx in np.intersect1d(data.testingTrials,data.centerInTrials):

        #    fin,fout = data.df['binSpike'][trialIdx][extraBins:], np.concatenate([data.df['binCursorPos'][trialIdx][extraBins:]])
        #    Zs = self.decodeFromData(data,fin,trialIdx)
        #    ### smooth trajectory
        #    #decoded_P =  np.zeros((Zs.shape[0],2))
        #    #decoded_P[0] = fout[0,0:2]
        #    #for ii in range(1,(decoded_P).shape[0]):
        #    #    decoded_P[ii] = (decoded_P[ii-1] + Zs[ii,2:4]*dt) #
        #    axes[1].plot(Zs[:,0],Zs[:,1],'b')
        #    axes[1].plot(fout[:,0],fout[:,1],'r')
        #    MSE_pos.append(MeanDiffDistance(Zs,fout[:,0:2],axis=0))
        #    MSE_vel.append(MeanDiffDistance(Zs[1:]-Zs[:-1],fout[1:,0:2]-fout[:-1,0:2],axis=0))

        #axes[1].set_title("Center-in trials(MSE pos{0:2.2f} vel{1:2.2f})".format(np.mean(MSE_pos),np.mean(MSE_vel)))


    #def test(self,data):
    #    MSE = []     
    #    for trialIdx in data.testingTrials:
    #    #for trialIdx in data.centerOutTrials:
    #        fin = data.df['binSpike'][trialIdx][:,:self.input_D]
    #        fout = np.concatenate([data.df['binCursorPos'][trialIdx][1:],data.df['binCursorVel'][trialIdx]],axis=1)
    #        Zs = self.decode(fin)

    #        ## smooth trajectory
    #        decoded_P =  np.zeros((Zs.shape[0],2))
    #        decoded_P[0] = fout[0,0:2]
    #        for ii in range(1,(decoded_P).shape[0]):
    #            decoded_P[ii] = 0.025*Zs[ii,0:2]+0.975*(decoded_P[ii-1] + Zs[ii,2:4]*self.dt/1000) #
    #        plt.plot(decoded_P[:,0],decoded_P[:,1],'b')
    #        plt.plot(fout[:,0],fout[:,1],'r')
    #        MSE.append(MeanDiffDistance(decoded_P,fout[self.history:,0:2],axis=1))
    #    plt.show()    
    #    print("MSE of decoded position(smoothed):",np.mean(MSE))




