import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import copy
import math 
from fig_gen_util import adjust_spines,remove_all_spines
from metrics import mean_square_error
from NeuralEncoder import NeuralEncoder
from NeuralDecoder import NeuralDecoder
from sklearn.decomposition import PCA
from Dataset import Dataset
from matplotlib.patches import Rectangle, Circle
from util import binning,PlotPSTH,spikesToPSTH,PlotGradientColorLine
from metrics import get_rho,get_R2,get_NRMSE,root_mean_square_error, NormalizedMeanDiff
from decoder.kalman import KFDecoder
from decoder.FORCE import FORCEDecoder

class BCIOfflineSim(object):
    def __init__(self,
                 data,
                 encoder,
                 decoder,
                 encoder_input_shape = 'pv',
                 M1_delay=0,
                 PMd_delay=0,
                 verbose=False):
        self.data = copy.deepcopy(data)
        self.neuralencoder = NeuralEncoder()
        self.neuraldecoder = NeuralDecoder()
        self.verbose = verbose
        self.dt = data.dt


        self.M1_delay = M1_delay
        self.PMd_delay = PMd_delay

        if isinstance(decoder,KFDecoder):
            self.neuraldecoder.decoder_type = decoder.type
            if decoder.type=='PVKF':
                self.neuraldecoder.alpha = 0.0
            else:
                self.neuraldecoder.alpha = 1.0
        elif isinstance(decoder,FORCEDecoder):
            self.neuraldecoder.decoder_type = 'FORCE'
            self.neuraldecoder.alpha = 0.95
        else:
            raise ValueError("decoder should be either KF or FORCE")

        self.neuralencoder.refresh_rate = self.dt
        self.neuralencoder.encoder_input_shape = encoder_input_shape
        self.neuralencoder.setEncoder(encoder)
        self.neuraldecoder.setDecoder(decoder)
        self.neuraldecoder.refresh_rate = decoder.dt
        self.neuraldecoder.Hspikes = np.zeros([4,192])
        self.neuraldecoder.lastState = np.hstack(([0,0,0,0,1])).reshape(5,1)
        self.neuraldecoder.numChannels = 192
        self.neuraldecoder.screenW = 1000
        self.neuraldecoder.screenH = 1000

        self.uniqueTarget = data.uniqueTarget
        self.ColorOfTarget = data.ColorOfTarget
        self.TrialsOfTargets = data.TrialsOfTargets

        self.df = pd.DataFrame(columns=[
                                        'trialNum',
                                        'binHandPos',
                                        'binHandVel',
                                        'real_decoded_pos',
                                        'real_decoded_vel',
                                        'real_spike_counts',
                                        'pseudo_decoded_pos',
                                        'pseudo_decoded_vel',
                                        'pseudo_spike_rates',
                                        'pseudo_spike_counts'])
        

    def setEncoder(self,encoder):
        raise NotImplementedError

    def setEncoder(self,encoder):
        raise NotImplementedError

    def setTask(self,task):
        raise NotImplementedError

    def run(self):
        '''
        Use data from monkey experiment.
        1.) Check decoder performance with real neural spikes from monkey experiments.
        2.) Check encoder performance by comparing 
            * PSTH
            * Neural dimension
            * decoded trajectories given a decoder

        Input: monkey data, encoder, decoder, 
        Output: 
        '''

        df = self.df
        data = self.data
        dt = self.dt
        neuralencoder = self.neuralencoder
        neuraldecoder = self.neuraldecoder

        ## Decode from real spikes
        for trialIdx in data.df.index:
            trialNum = data.df['trialNum'][trialIdx]
            real_spike_counts = data.df['binSpike'][trialIdx]
            cursor_p = data.df['binCursorPos'][trialIdx]
            cursor_v = data.df['binCursorVel'][trialIdx]
            
            trial_length = cursor_p.shape[0]
            ## reset decoder position
            self.neuraldecoder.reset(cursor_p[0],cursor_v[0]/1000)

            ## record decoded results
            real_decoded_pos = np.empty((trial_length,2))
            real_decoded_vel = np.empty((trial_length,2))

            n = neuraldecoder.refresh_rate//neuralencoder.refresh_rate
            ### start with the initial x0 as in Wu and Chestek, we get the first for free
            real_decoded_pos[0] = cursor_pos = cursor_p[0]
            real_decoded_vel[0] = cursor_vel = cursor_v[0]

            for tt in range(1, trial_length+1):
                if tt % n ==0:
                    spike_counts = np.sum(real_spike_counts[tt-n:tt],axis=0)
                    cursor_pos,_P,cursor_vel = neuraldecoder.decode(spike_counts)
                real_decoded_pos[tt:tt+1] = cursor_pos 
                real_decoded_vel[tt:tt+1] = cursor_vel

            df = df.append({     'trialNum':trialNum,
                                 'real_decoded_pos':real_decoded_pos,
                                 'real_decoded_vel':real_decoded_vel,
                                 'real_spike_counts':real_spike_counts,
                                 }, ignore_index=True)


        ## get spike_count sequence  
        Q_spike_counts = []
        Q_spike_rates = []
        neuralencoder.reset()
        for trialIdx in data.df.index:
            trialNum = data.df['trialNum'][trialIdx]

            target_p = data.df['target'][trialIdx]
            hand_p   = data.df['binHandPos'][trialIdx]
            cursor_p = data.df['binCursorPos'][trialIdx]
            trial_length = hand_p.shape[0]

            for tt in range(0,trial_length):
                encoder_input = hand_p[tt], target_p, cursor_p[tt]
                spike_counts,spike_rates = neuralencoder.encode(*encoder_input)
                Q_spike_counts.append(spike_counts)
                Q_spike_rates.append(spike_rates)

        ## 
        Q_spike_rates = np.array(Q_spike_rates)
        M1_spike_rates = Q_spike_rates[:,-1,0:96]
        PMd_spike_rates = Q_spike_rates[:,-1,96:192]

        ## take out the first trial
        trial_length = len(data.df['binHandPos'][0])
        M1_spike_rates = M1_spike_rates[trial_length+self.M1_delay:]
        PMd_spike_rates = PMd_spike_rates[trial_length+self.PMd_delay:]

        ## take out the last trial
        trial_length = len(data.df['binHandPos'][len(data.df)-1])
        M1_spike_rates = M1_spike_rates[:-trial_length+self.M1_delay]
        PMd_spike_rates = PMd_spike_rates[:-trial_length+self.PMd_delay]

        Q_spike_rates = np.hstack([M1_spike_rates, PMd_spike_rates])


        ## 
        Q_spike_counts = np.array(Q_spike_counts)
        M1_spike_counts = Q_spike_counts[:,-1,0:96]
        PMd_spike_counts = Q_spike_counts[:,-1,96:192]

        ## take out the first trial
        trial_length = len(data.df['binHandPos'][0])
        M1_spike_counts = M1_spike_counts[trial_length+self.M1_delay:]
        PMd_spike_counts = PMd_spike_counts[trial_length+self.PMd_delay:]

        ## take out the last trial
        trial_length = len(data.df['binHandPos'][len(data.df)-1])
        M1_spike_counts = M1_spike_counts[:-trial_length+self.M1_delay]
        PMd_spike_counts = PMd_spike_counts[:-trial_length+self.PMd_delay]

        Q_spike_counts = np.hstack([M1_spike_counts, PMd_spike_counts])
        
        ## Decode from pseudo spikes
        for trialIdx in data.df.index[1:-1]:
            trialNum = data.df['trialNum'][trialIdx]

            target_p = data.df['target'][trialIdx]
            hand_p   = data.df['binHandPos'][trialIdx]
            hand_v   = data.df['binHandVel'][trialIdx]
            cursor_p = data.df['binCursorPos'][trialIdx]
            cursor_v = data.df['binCursorVel'][trialIdx]
            
            trial_length = hand_p.shape[0]
            ## reset decoder position
            self.neuraldecoder.reset(cursor_p[0],cursor_v[0]/1000)

            ## record decoded results
            pseudo_decoded_pos = np.empty((trial_length,2))
            pseudo_decoded_vel = np.empty((trial_length,2))

            n = neuraldecoder.refresh_rate//neuralencoder.refresh_rate

            ### start with the initial x0 as in Wu and Chestek, we get the first for free
            pseudo_decoded_pos[0] = cursor_pos =cursor_p[0]
            pseudo_decoded_vel[0] = cursor_vel =cursor_v[0]
            for tt in range(1, trial_length+1):
                if tt % n ==0:
                    spike_counts = np.sum(Q_spike_counts[tt-n:tt],axis=0)
                    cursor_pos,_P,cursor_vel = neuraldecoder.decode(spike_counts)
                pseudo_decoded_pos[tt:tt+1] = cursor_pos 
                pseudo_decoded_vel[tt:tt+1] = cursor_vel
   

            pseudo_spike_counts = Q_spike_counts[:trial_length]
            Q_spike_counts = Q_spike_counts[trial_length:]

            pseudo_spike_rates = Q_spike_rates[:trial_length]
            Q_spike_rates = Q_spike_rates[trial_length:]


            assert df['trialNum'][trialIdx]==trialNum
            df['binHandPos'][trialIdx]=hand_p
            df['binHandVel'][trialIdx]=hand_v
            df['pseudo_decoded_pos'][trialIdx]=pseudo_decoded_pos
            df['pseudo_decoded_vel'][trialIdx]=pseudo_decoded_vel
            df['pseudo_spike_counts'][trialIdx]=pseudo_spike_counts
            df['pseudo_spike_rates'][trialIdx]=pseudo_spike_rates

        assert len(Q_spike_counts)==0

        df['target'] = data.df['target']
        df['isSuccessful'] = data.df['isSuccessful']
        df = df.drop([0,len(df)-1]).reset_index(drop=True)

        data.df = data.df.drop([0,len(data.df)-1]).reset_index(drop=True)
        self.TrialsOfTargets = data._TrialsOfTargets()
        self.ColorOfTarget = data._ColorOfTarget()
        self.numTrial = len(df)

        self.real_PSTH = spikesToPSTH(self,df['real_spike_counts'])
        self.pseudo_PSTH = spikesToPSTH(self,df['pseudo_spike_rates']) ## pseudo_spike_rates is better to represent PSTH

        self.df = df
        self.numBins = data.numBins
        self.extraBins = data.extraBins
        self.numTrial = len(df)
        self.centerOutTrials = np.where((np.stack(df['target'])!=[0,0]).any(axis=1))[0]
        self.centerInTrials  = df[~df.index.isin(self.centerOutTrials)].index
        self.uniqueTarget = data._UniqueTarget()
        self.TrialsOfTargets = data._TrialsOfTargets()
        self.ColorOfTarget = data._ColorOfTarget()

        self.numChannels=192

        self.successfulTrials = np.where(df['isSuccessful']==1)[0]
        self.trainingTrials  = self.successfulTrials
        self.testingTrials = self.successfulTrials

        self.numTrain = numTrain = len(self.trainingTrials)
        self.numTest =  numTest = len(self.testingTrials)
        self.PCAanalysis()

    def Statistics(self): 
        df = self.df
        statistics = {}
        ## decoded traj
        #statistics['NMD(real_Pos vs pseudo_Pos)'] =  NormalizedMeanDiff(np.vstack(self.df['real_decoded_pos']),np.vstack(self.df['pseudo_decoded_pos']),axis=1)
        #statistics['NMD(hand_Pos vs pseudo_Pos)'] =  NormalizedMeanDiff(np.vstack(self.df['binHandPos']),np.vstack(self.df['pseudo_decoded_pos']),axis=1)
        #statistics['NMD(real_Vel vs pseudo_Vel)'] =  NormalizedMeanDiff(np.vstack(self.df['real_decoded_vel']),np.vstack(self.df['pseudo_decoded_vel']),axis=1)
        #statistics['NMD(hand_Vel vs pseudo_Vel)'] =  NormalizedMeanDiff(np.vstack(self.df['binHandVel']),np.vstack(self.df['pseudo_decoded_vel']),axis=1)
        statistics['NRMSE(handVel vs pseudoVel)'] = np.mean(get_NRMSE(df['binHandVel'], df['pseudo_decoded_vel']))
        statistics['NRMSE(realVel vs pseudoVel)'] = np.mean(get_NRMSE(df['real_decoded_vel'], df['pseudo_decoded_vel']))

        ## PSTH
        pseudo_PSTH = np.stack([PSTH[self.extraBins:self.extraBins+self.numBins] for PSTH in self.pseudo_PSTH])
        real_PSTH = np.stack([PSTH[self.extraBins:self.extraBins+self.numBins] for PSTH in self.real_PSTH])
        corr     = np.nanmean(get_rho(real_PSTH,pseudo_PSTH))
        R2       = np.nanmean(get_R2(real_PSTH,pseudo_PSTH))
        NRMSE = np.nanmean(get_NRMSE(real_PSTH,pseudo_PSTH))
        statistics['PSTH_PCC']=float(corr)
        #statistics['PSTH_R2']=R2
        statistics['PSTH_NRMSE']=NRMSE

        ## PCA
        statistics['Dim(real)'] = self.real_PCA_dim
        statistics['Dim(pseudo)'] = self.pseudo_PCA_dim
        #corr    = np.nanmean(get_rho(self.real_neural_dynamics, self.pseudo_neural_dynamics))
        #statistics['Dynamics_PCC'] = corr
        #NRMSE     = np.nanmean(get_NRMSE(self.real_neural_dynamics, self.pseudo_neural_dynamics))
        #statistics['Dynamics_NRMSE'] = NRMSE
        corr    = np.nanmean(get_rho(self.real_neural_dynamics[:,:,:self.real_PCA_dim], self.pseudo_neural_dynamics[:,:,:self.real_PCA_dim]))
        statistics['DynamicsTopN_PCC'] = corr
        NRMSE     = np.nanmean(get_NRMSE(self.real_neural_dynamics[:,:,:self.real_PCA_dim], self.pseudo_neural_dynamics[:,:,:self.real_PCA_dim]))
        statistics['DynamicsTopN_NRMSE'] = NRMSE

        #display(statistics)
        
        return statistics


    def PlotPSTH(self,neuronList = range(0,192,32),xlim=500,ylim=300):
        dt = self.dt
        fig,axes = plt.subplots(2,len(neuronList),figsize=(len(neuronList)*5,5),squeeze=False)
        for ii,neuronIdx in enumerate(neuronList):
            for targetIdx,target in enumerate(self.uniqueTarget):
                axes[0,ii].title.set_text('N '+str(neuronIdx))
                color = self.ColorOfTarget[tuple(target)]
                PlotPSTH(axes=axes[0,ii],PSTH=self.real_PSTH[targetIdx].T[neuronIdx]*(1000/dt),color=color,dt=self.dt)
                PlotPSTH(axes=axes[1,ii],PSTH=self.pseudo_PSTH[targetIdx].T[neuronIdx]*(1000/dt),color=color,dt=self.dt)
            axes[0,ii].set_xlim([0,xlim])
            axes[0,ii].set_ylim([0,ylim])
            axes[1,ii].set_xlim([0,xlim])
            axes[1,ii].set_ylim([0,ylim])


        for row_index in range(len(axes)):
            for column_index in range(len(axes[row_index])):
                ax = axes[row_index,column_index]
                if row_index==0 and column_index==0:
                    ax.set_yticks([0,100])
                    ax.set_xticks([0,100])
                    adjust_spines(ax)
                    ax.set_yticks([])
                    ax.set_xticks([])
                else:
                    adjust_spines(ax)
                    ax.set_yticks([])
                    ax.set_xticks([])
                    ax.spines['left'].set_visible(False)
                    ax.spines['bottom'].set_visible(False)

    def PlotCursorVelProfile(self,ylim=400,group='all',color=None,source='real_decoded_vel'):
        #self.df['binCursorVel'] = self.df['real_decoded_vel']
        #Dataset._PlotVelProfile(self,source='binCursorVel',ylim=ylim,group=group,color='b')
        #self.df['binCursorVel'] = self.df['pseudo_decoded_vel']
        #Dataset._PlotVelProfile(self,source='binCursorVel',ylim=ylim,group=group,color='r')
        df = self.df
        dt = self.neuraldecoder.refresh_rate


        allV = []
        for trialIdx in self.centerOutTrials:
            if df['isSuccessful'][trialIdx]:
                v = df[source][trialIdx][:,0:2]
                vel = np.linalg.norm(v,axis=1)
                allV.append(vel)


        lens = [len(i) for i in allV]
        arr = np.ma.empty((np.max(lens),len(allV)))
        arr.mask = True
        for idx, l in enumerate(allV):
            arr[:len(l),idx] = l
        avgV = arr.mean(axis = -1)
        avgV = avgV[::self.neuraldecoder.refresh_rate//self.neuralencoder.refresh_rate]
        avg_real_decoded_vel = avgV

        allV = []
        for trialIdx in self.centerOutTrials:
            if df['isSuccessful'][trialIdx]:
                v = df['pseudo_decoded_vel'][trialIdx][:,0:2]
                vel = np.linalg.norm(v,axis=1)
                allV.append(vel)


        lens = [len(i) for i in allV]
        arr = np.ma.empty((np.max(lens),len(allV)))
        arr.mask = True
        for idx, l in enumerate(allV):
            arr[:len(l),idx] = l
        avgV = arr.mean(axis = -1)
        avgV = avgV[::self.neuraldecoder.refresh_rate//self.neuralencoder.refresh_rate]
        avg_pseudo_decoded_vel = avgV

        
        #print("Pearson's corr:{}".format(get_rho(avg_real_decoded_vel,avg_pseudo_decoded_vel)))
        #print("RMSE:{}".format(np.sqrt(mean_square_error(avg_real_decoded_vel,avg_pseudo_decoded_vel))))

        plt.plot(range(0,len(avg_real_decoded_vel)*dt,dt),  avg_real_decoded_vel, '.-', linewidth=2,color='b')
        plt.plot(range(0,len(avg_pseudo_decoded_vel)*dt,dt),avg_pseudo_decoded_vel, '.-', linewidth=2,color='r')

        plt.xlim([0,1000])
        plt.xticks(range(0,1001,500))

        plt.ylim([0,ylim])
        plt.yticks(range(0,ylim+1,100))


    def PlotMovement(self,acceptance_window=40,lim=160,source='real_decoded_pos'):
        df = self.df
        #plt.gcf().set_size_inches(7.5, 7.5)
        ax = plt.gca()
        ax.add_patch( Rectangle((-acceptance_window/2, -acceptance_window/2),acceptance_window,acceptance_window,linewidth=2,
                        edgecolor='black',
                        facecolor='None',
                        linestyle='dotted', alpha=0.9)) # draw center target
        ax.add_patch( Circle((0, 0),5,color='green',alpha=0.5))

        for target in self.uniqueTarget:
            pos = (target[0], target[1])
            ax.add_patch( Circle(pos,5,color='green',alpha=0.5))
            pos = (target[0]-acceptance_window/2, target[1]-acceptance_window/2)
            ax.add_patch( Rectangle(pos,acceptance_window,acceptance_window,linewidth=2,
                        edgecolor='black',
                        facecolor='None',
                        linestyle='dotted', alpha=0.9)) # draw center target


        plt.axis("off")


        for target in self.uniqueTarget:
            #color = self.ColorOfTarget[tuple(target)]
            trials = self.TrialsOfTargets[tuple(target)]
            for ii,trialIdx in enumerate(trials):
                if ii==0:
                    alpha=1
                else:
                    alpha=0.1
                p = df[source][trialIdx]
                hand1, = plt.plot(p[:,0],p[:,1],'-', color='b',label='real',linewidth=1,alpha=alpha)
                p = df['pseudo_decoded_pos'][trialIdx]
                hand2, = plt.plot(p[:,0],p[:,1],'-', color='r',label='pseudo',linewidth=1,alpha=alpha)

        #handles, labels = ax.get_legend_handles_labels()
        display = (0,500)
        #ax.legend([hand1,hand2],['real','pseudo'],loc = 'best')
               

        ax.set_xlabel("(mm)")
        ax.set_ylabel("(mm)")
        ax.set_xlim([-lim,lim])
        ax.set_ylim([-lim,lim])

        #plt.show()

    def _PlotDynamics(self,source):
        if source=='real':
            neural_dynamics = self.real_neural_dynamics
        elif source=='pseudo':
            neural_dynamics = self.pseudo_neural_dynamics
        
        #plt.sca(axes[0])
        for ii,target in enumerate(self.uniqueTarget):
            xline=neural_dynamics[ii,:,0]
            yline=neural_dynamics[ii,:,1]
            zline=neural_dynamics[ii,:,2]
            color = self.ColorOfTarget[tuple(target)]
            plt.plot(xline,yline,color=color)


    def PlotDynamics(self,axes=None):
        if axes==None:
            fig, axes = plt.subplots(1, 2, figsize=(10,5), sharex=True,sharey=True)#
        
        

        #axes[0].title.set_text('Real')
        plt.sca(axes[0])
        self._PlotDynamics('real')
        

        #axes[1].title.set_text('Pseudo')
        plt.sca(axes[1])
        self._PlotDynamics('pseudo')

        remove_all_spines(axes[0])
        remove_all_spines(axes[1])
        

    def PCAanalysis(self):
        pseudo_PSTH = np.stack([PSTH[self.extraBins:self.extraBins+self.numBins] for PSTH in self.pseudo_PSTH])
        real_PSTH = np.stack([PSTH[self.extraBins:self.extraBins+self.numBins] for PSTH in self.real_PSTH])

        real_PCA = PCA(n_components=min(real_PSTH.shape[0]*real_PSTH.shape[1],real_PSTH.shape[2]))
        real_PCA.fit(real_PSTH.reshape(-1,real_PSTH.shape[2]))

        pseudo_PCA = PCA(n_components=min(pseudo_PSTH.shape[0]*pseudo_PSTH.shape[1],pseudo_PSTH.shape[2]))
        pseudo_PCA.fit(pseudo_PSTH.reshape(-1,pseudo_PSTH.shape[2]))
        
        real_neural_dynamics = real_PCA.transform(real_PSTH.reshape(-1,real_PSTH.shape[2]))
        pseudo_neural_dynamics = real_PCA.transform(pseudo_PSTH.reshape(-1,pseudo_PSTH.shape[2]))

        real_PCA_dim = np.where(real_PCA.explained_variance_ratio_.cumsum()>0.9)[0][0]
        pseudo_PCA_dim = np.where(pseudo_PCA.explained_variance_ratio_.cumsum()>0.9)[0][0]
        
        self.real_PCA_dim = real_PCA_dim
        self.pseudo_PCA_dim = pseudo_PCA_dim

        self.real_PCA = real_PCA
        self.pseudo_PCA = pseudo_PCA
        
        self.real_neural_dynamics = real_neural_dynamics.reshape(real_PSTH.shape[0], real_PSTH.shape[1],-1)
        self.pseudo_neural_dynamics = pseudo_neural_dynamics.reshape(pseudo_PSTH.shape[0], pseudo_PSTH.shape[1],-1)

    def PlotAll(self,b_save=False,path='./',postfix=''):
        fig,axes = plt.subplots(2,2,figsize=(10,10))
        plt.sca(axes[0,0])
        self.PlotMovement()
        plt.sca(axes[0,1])
        self.PlotCursorVelProfile()
        self.PlotDynamics(axes=[axes[1,0],axes[1,1]])
        #plt.tight_layout()
        if b_save:
            fig.savefig(os.path.join(path,'DecodeFromPseudo'+postfix)+'.pdf')

        self.PlotPSTH(neuronList=[9,96,145])#,axes=[axes1,axes2]
        if b_save:
            fig = plt.gcf()
            fig.savefig(os.path.join(path,'PSTH'+postfix)+'.pdf')

        
        
