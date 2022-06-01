import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import os
import json
from util import binning,PlotPSTH,spikesToPSTH,PlotGradientColorLine
from statsmodels.nonparametric.smoothers_lowess import lowess
from scipy.sparse import csr_matrix
from sklearn.decomposition import PCA
from matplotlib.patches import Rectangle, Circle
from matplotlib.ticker import PercentFormatter
from stable_baselines.common.base_class import BaseRLModel
from stable_baselines.common.cmd_util import make_vec_env
from stable_baselines.common.vec_env import VecFrameStack
from gym_centerout.envs.centerout_env import CenteroutEnv
import time
import warnings
import pickle
from fig_gen_util import get_color_def,adjust_spines

def PlotTrialTimeBar(data=[],legend=[]):
    color = get_color_def()

    n_data = len(data)

    '''
    Trial time
    '''
    fig = plt.figure(figsize=(5,5))
    ax = fig.add_axes([0,0,1,1])
    ax.set_xlabel("")
    ax.set_ylabel("Trial time (ms)",fontsize=20)
    ax.set_ylim([0,1500])
    ax.set_yticks(range(0,1500+1,250))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='y',labelsize=10)
    ax.tick_params(axis='x',labelsize=20)


    y = []
    y_sem = []
    y_std = []
    color_list = []
    for ii in range(n_data):

        df_centerOut = data[ii].df.iloc[data[ii].centerOutTrials]
        X = df_centerOut['timeLastTargetAcquire'][df_centerOut['isSuccessful']==True]
        y.append(X.mean())
        y_sem.append(X.sem())
        y_std.append(X.std())
        color_list.append(color[legend[ii]])

    ax.bar(legend,y,yerr = y_sem,color=color_list,align='center',alpha=0.5, ecolor='black', capsize=10)
    print("Trial time:")
    [print("{}: {:.2f}, sem {:.2f}, std {:.2f}".format(legend[ii],y[ii],y_sem[ii],y_std[ii])) for ii in range(n_data)]

    '''
    Dial-in time
    '''
    fig = plt.figure(figsize=(5,5))
    ax = fig.add_axes([0,0,1,1])
    ax.set_xlabel("")
    ax.set_ylabel("Dial-in time [ms]",fontsize=20)
    ax.set_ylim([0,1000])
    ax.set_yticks(range(0,1000+1,250))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='y',labelsize=10)
    ax.tick_params(axis='x',labelsize=20)

    y = []
    y_sem = []
    y_std =[]
    color_list = []


    for ii in range(n_data):

        df_centerOut = data[ii].df.iloc[data[ii].centerOutTrials]
        X_last = df_centerOut['timeLastTargetAcquire'][df_centerOut['isSuccessful']==True]
        X_first = df_centerOut['timeFirstTargetAcquire'][df_centerOut['isSuccessful']==True]

        X = X_last - X_first
        y.append(X.mean())
        y_sem.append(X.sem())
        y_std.append(X.std())
        color_list.append(color[legend[ii]])

    ax.bar(legend,y,yerr = y_sem,color=color_list,align='center',alpha=0.5, ecolor='black', capsize=10)
    print("Dail-in time:")
    [print("{}: {:.2f}, sem {:.2f}, std {:.2f}".format(legend[ii],y[ii],y_sem[ii],y_std[ii])) for ii in range(n_data)]


    '''
    First-touch time
    '''
    fig = plt.figure(figsize=(5,5))
    ax = fig.add_axes([0,0,1,1])
    ax.set_xlabel("")
    ax.set_ylabel("First-touch time (ms)",fontsize=20)
    ax.set_ylim([0,1000])
    ax.set_yticks(range(0,1000+1,250))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='y',labelsize=10)
    ax.tick_params(axis='x',labelsize=20)

    y = []
    y_sem = []
    y_std =[]
    color_list = []


    for ii in range(n_data):

        df_centerOut = data[ii].df.iloc[data[ii].centerOutTrials]
        X_first = df_centerOut['timeFirstTargetAcquire'][df_centerOut['isSuccessful']==True]
        y.append(X_first.mean())
        y_sem.append(X_first.sem())
        y_std.append(X_first.std())
        color_list.append(color[legend[ii]])

    ax.bar(legend,y,yerr = y_sem,color=color_list,align='center',alpha=0.5, ecolor='black', capsize=10)
    print("First-touch time:")
    [print("{}: {:.2f}, sem {:.2f}, std {:.2f}".format(legend[ii],y[ii],y_sem[ii],y_std[ii])) for ii in range(n_data)]

class Dataset():
    def __init__(self):
        self.verbose=False

    def _check(self):
        '''
        check the dataset structure in a very high level to avoid assert in later operation
        '''
        df = self.df
        dt = self.dt
        for trialIdx in range(len(df)):
            binHandPos=df['binHandPos'][trialIdx]
            binHandVel=df['binHandVel'][trialIdx]
            binCursorPos=df['binCursorPos'][trialIdx]
            binCursorVel=df['binCursorVel'][trialIdx]
            ## check the length of sequence are the same.
            assert df['binHandPos'][trialIdx].shape[0]==df['binSpike'][trialIdx].shape[0]
            assert df['binCursorPos'][trialIdx].shape[0]==df['binSpike'][trialIdx].shape[0]
            assert df['binHandVel'][trialIdx].shape[0]==df['binSpike'][trialIdx].shape[0]
            assert df['binCursorVel'][trialIdx].shape[0]==df['binSpike'][trialIdx].shape[0]

            ## binHandVel is not directly calculated from binHandPos in Licorice experiment. There is a sampling time difference.
            #assert ((binHandPos[1:]-binHandPos[:-1])/dt*1000 == binHandVel[1:]).all()
            assert ((binCursorPos[1:]-binCursorPos[:-1])/dt*1000 == binCursorVel[1:]).all()

        print("Dataset pass the check")

    def save(self,path=None):
        print("save:",path)
        with open(path,'wb') as f:
            pickle.dump(self.df,f)
            pickle.dump(self.dt,f)
            pickle.dump(self.extraBins,f)
            pickle.dump(self.metadata,f)


    def load(self,path=None):
        with open(path,'rb') as f:
            df = pickle.load(f)
            dt = pickle.load(f)
            extraBins = pickle.load(f)
            metadata = pickle.load(f)

        self.df = df
        self.dt = dt
        self.metadata=metadata
        self.extraBins=extraBins
        self.numTrial = len(self.df)
        self.numBins = np.min([df['binSpike'][i].shape[0] for i in range(len(df))])

        self.centerOutTrials = np.where((np.stack(df['target'])!=[0,0]).any(axis=1))[0]
        self.centerInTrials  = df[~df.index.isin(self.centerOutTrials)].index

        self.uniqueTarget = self._UniqueTarget()
        self.TrialsOfTargets = self._TrialsOfTargets()
        self.ColorOfTarget = self._ColorOfTarget()

        self._CalculateDistanceToTarget()
        self._CalculateDeviation()
        self._CalculateDistanceRatio()
        self.PSTH = spikesToPSTH(self,self.df['binSpike'])
        self.neuralDynamics = self._NeuralDynamics()
        self.meanDtT = self._getMeanDtT()

    def summary(self):
        '''
        get general information
        '''
        print('###################################')
        print('binning width:',self.dt)
        print('numBins:',self.numBins)
        print('extraBins:',self.extraBins)

        display(self.Statistics())
        print('###################################')

    def merge(self,data):
        '''
        merge two datasets
        '''
        assert self.dt==data.dt
        self.df = pd.concat([self.df, data.df], ignore_index=True)
        self.trainingTrials = np.concatenate([self.trainingTrials, self.numTrial+np.array(data.trainingTrials)])
        self.testingTrials = np.concatenate([self.testingTrials, self.numTrial+np.array(data.testingTrials)])
        self.numTrial = numTrial = self.numTrial + data.numTrial
        self.numTest = numTest = self.numTest + data.numTest
        self.numTrain = numTrain = self.numTrain + data.numTrain

        self.centerOutTrials = np.where((np.stack(self.df['target'])!=[0,0]).any(axis=1))[0]
        self.uniqueTarget = self._UniqueTarget()
        self.TrialsOfTargets = self._TrialsOfTargets()
        self.ColorOfTarget = self._ColorOfTarget()

        self._CalculateDistanceToTarget()
        self._CalculateDeviation()
        self._CalculateDistanceRatio()

        self.meanDtT = self._getMeanDtT()

    def _UniqueTarget(self):
        '''
        get a list of unique targets in this dataset
        '''
        df = self.df
        if(df.keys()=='target').any():
            uniqueTarget = [target for target in np.unique(df['target']) if target!=(0,0)]
        else:
            uniqueTarget = []
        return uniqueTarget

    def _TrialsOfTargets(self):
        '''
        get a dictionary of trials of each target
        '''
        df = self.df
        rt = {}
        for target in self.uniqueTarget:
            trials=[]
            for trial in range(len(df)):
                if(target == df['target'][trial]):
                    trials.append(trial)
            rt[tuple(target)] = trials
        return rt

    def _appendTrials(self,df,extraPeriod,dropList):
        '''
        extend trial length by cascading a trial with a previous trail
        '''
        for trialIdx in reversed(range(1,len(df))):
            trialNum = df['trialNum'][trialIdx]
            preTrialNum = df['trialNum'][trialIdx-1]

            if (trialNum-1==preTrialNum):
                df['spikeRaster'][trialIdx] = np.vstack([df['spikeRaster'][trialIdx-1][-extraPeriod:],df['spikeRaster'][trialIdx]])
                df['handPos'][trialIdx]     = np.vstack([df['handPos'][trialIdx-1][-extraPeriod:],df['handPos'][trialIdx]])
                df['cursorPos'][trialIdx]   = np.vstack([df['cursorPos'][trialIdx-1][-extraPeriod:],df['cursorPos'][trialIdx]])
            else:
                dropList.append(trialIdx)

        return df

    def _ColorOfTarget(self):
        '''
        set specific color for each target to have better visualization
        '''
        ## color setting for each target
        n = len(self.uniqueTarget)
        colors = plt.cm.jet(np.linspace(0,1,n))
        df = self.df
        rt = {}
        for ii,target in enumerate(self.uniqueTarget):
            trials=[]
            rt[tuple(target)] = colors[ii]
        return rt

    def _NeuralDynamics(self):
        '''
        calculate neural dynamics from PSTH
        '''
        if len(self.PSTH)==0:
        #if self.PSTH.shape[0]==0:
            return np.array([])
        PSTH = np.stack([PSTH[self.extraBins:self.extraBins+self.numBins] for PSTH in self.PSTH])
        pcaReal = PCA(n_components=min(PSTH.shape[0]*PSTH.shape[1],PSTH.shape[2]))
        pcaReal.fit(PSTH.reshape(-1,PSTH.shape[2]))

        neuralDynamics = pcaReal.transform(PSTH.reshape(-1,PSTH.shape[2]))
        neuralDynamics = neuralDynamics.reshape(PSTH.shape[0],PSTH.shape[1],-1)
        assert neuralDynamics.shape[1]==self.numBins,"{},{}".format(neuralDynamics.shape[1],self.numBins)
        return neuralDynamics

    def _getMeanDtT(self):
        '''
        calculate mean distance-to-target(DtT) profile
        '''
        df = self.df
        dt = self.dt
        lens = [len(i) for i in df['binCursorPosDistance'][self.centerOutTrials]]
        if len(lens)==0:
            return np.array([])
        else:
            D = np.ma.empty((np.max(lens),len(self.centerOutTrials)))
            D.mask = True
            for ii,trial in enumerate(self.centerOutTrials):
                if df['isSuccessful'][trial]:
                    D[:len(df['binCursorPosDistance'][trial]),ii] = df['binCursorPosDistance'][trial]
            meanD = D.mean(axis=1)

            return meanD

    def _CalculateDistanceToTarget(self):
        '''
        calculate binned cursor position distance and save in dataframe
        '''
        if(self.verbose>0):
            print('Start calculating DistanceToTarget')
        df          = self.df
        numTrial    = self.numTrial
        dt          = self.dt
        df['binCursorPosDistance'] = ""
        for trialIdx in range(numTrial):
            df.at[trialIdx,'binCursorPosDistance'] = np.linalg.norm( df['binCursorPos'][trialIdx]-df['target'][trialIdx], axis=1 )

    def _CalculateDeviation(self):
        '''
        calculate maximum trajectory deviation of cursor movement and save in dataframe
        trajectory deviation is defined as the deviation from the straight line of start point and end point of a trajectory.
        note: this definition may be different from in other papers.
        '''

        df = self.df
        dt = self.dt
        df['max_deviation'] = np.nan
        for trial in range(len(df)):
            if df['isSuccessful'][trial]:
                #LTT = int(df['timeLastTargetAcquire'][trial]//dt)
                #assert len(df['binCursorPos'][trial])>=LTT
                binCursorPos = df['binCursorPos'][trial]#[:LTT+1]
                point_vector = binCursorPos[-1] - binCursorPos[0]
                normalized_point_vector = point_vector / np.linalg.norm(point_vector)
                projection_length = np.dot(binCursorPos-binCursorPos[0],normalized_point_vector)

                deviation = (binCursorPos-binCursorPos[0]) - projection_length.reshape(-1,1)*normalized_point_vector
                df.at[trial,'max_deviation'] = np.max(np.linalg.norm(deviation,axis=1))

        df['max_deviation(hand)'] = np.nan
        for trial in range(len(df)):
            if df['isSuccessful'][trial]:
                #LTT = int(df['timeLastTargetAcquire'][trial]//dt)
                #assert len(df['binHandPos'][trial])>=LTT
                binHandPos = df['binHandPos'][trial]#[:LTT+1]
                point_vector = binHandPos[-1] - binHandPos[0]
                normalized_point_vector = point_vector / np.linalg.norm(point_vector)
                projection_length = np.dot(binHandPos-binHandPos[0],normalized_point_vector)

                deviation = (binHandPos-binHandPos[0]) - projection_length.reshape(-1,1)*normalized_point_vector
                df.at[trial,'max_deviation(hand)'] = np.max(np.linalg.norm(deviation,axis=1))

    def _CalculateDistanceRatio(self):
        '''
        Distance ratio is the total trajectory distance divided by the distance of a straight line. We care the trajectory from time 0 to the last-touch time, meaning exclude the dial-in trajectory. Straight line distance is the distance between cursor position at time 0 to the cursor position at last-touch time.
        Position: P0, P1, ..., P_LTT, ... PN
        Velocity: V0, V1 ... , V_LTT, ... VN
        Total distance : sum(V[1:LTT])
        Straight line distance: P_LTT - P0
        '''

        df = self.df
        numTrial = self.numTrial
        dt = self.dt
        df['distance_ratio'] = np.nan
        df['distance_ratio(hand)'] = np.nan
        for trial in range(numTrial):
            if df['isSuccessful'][trial]:
                distance = np.linalg.norm(df['binCursorPos'][trial][0]-df['binCursorPos'][trial][-1])
                if distance==0:
                    distance_ratio=1
                else:
                    distance_ratio = np.sum(np.linalg.norm(df['binCursorPos'][trial][1:]-df['binCursorPos'][trial][:-1],axis=1))/distance
                df.at[trial,'distance_ratio'] = distance_ratio

                distance = np.linalg.norm(df['binHandPos'][trial][0]-df['binHandPos'][trial][-1])
                if distance==0:
                    distance_ratio=1
                else:
                    distance_ratio = np.sum(np.linalg.norm(df['binHandPos'][trial][1:]-df['binHandPos'][trial][:-1],axis=1))/distance

                df.at[trial,'distance_ratio(hand)'] = distance_ratio


    def _smoothPos(self,df):
        '''
        smooth hand position and cursor position
        '''
        bSmoothPos = self.bSmoothPos
        dt = self.dt

        if(bSmoothPos):
            tic = time.time()
            smoothedCursorPos = []
            smoothedHandPos = []
            for ii in range(len(df['handPos'])):
                ## smooth x positions
                y = df['handPos'][ii][:,0]
                x = list(range(len(y)))
                frac = (dt+1)/len(x)#0.1#
                filtered = lowess(y,x,frac=frac,it=0,is_sorted=True)[:,1]
                assert (y.shape==filtered.shape)

                ## smooth y positions
                y = df['handPos'][ii][:,1]
                x = list(range(len(y)))
                frac = (dt+1)/len(x)#0.1#
                filtered = np.vstack([filtered,lowess(y,x,frac=frac,it=0,is_sorted=True)[:,1]]).T
                assert (df['handPos'][ii].shape == filtered.shape),"trial:{},{},{}".format(ii,df['handPos'][ii].shape,filtered.shape)

                smoothedHandPos.append(filtered)

                ## smooth x positions
                y = df['cursorPos'][ii][:,0]
                x = list(range(len(y)))
                frac = (dt+1)/len(x)#0.1#
                filtered = lowess(y,x,frac=frac,it=0,is_sorted=True)[:,1]
                assert (y.shape==filtered.shape)

                ## smooth y positions
                y = df['cursorPos'][ii][:,1]
                x = list(range(len(y)))
                frac = (dt+1)/len(x)#0.1
                filtered = np.vstack([filtered,lowess(y,x,frac=frac,it=0,is_sorted=True)[:,1]]).T
                assert (df['cursorPos'][ii].shape == filtered.shape),"trial:{},{},{}".format(ii,df['cursorPos'][ii].shape,filtered.shape)

                smoothedCursorPos.append(filtered)


            print("smooth positions ({}s)".format(time.time()-tic))

            df['cursorPos'] = smoothedCursorPos
            df['handPos'] = smoothedHandPos
        return df

    def _PlotTimeHist(self,color,ylim,source,alpha=0.5,shift_time=0):
        import seaborn as sns
        from scipy.stats import gaussian_kde
        df = self.df

        df_centerOut = df.iloc[self.centerOutTrials]

        X = list(np.array(df_centerOut[source][df_centerOut['isSuccessful']==True]-shift_time)/1000) #(s)
        #X = list(df_centerOut[source][df_centerOut['isSuccessful']==True])/1000 #(s)
        #sns.kdeplot(X,shade=True,bw=25,color=color)
        plt.hist(X,bins=np.arange(0,3,0.1), weights=np.ones(len(X)) / len(X),color=color,alpha=alpha)
        plt.gca().yaxis.set_major_formatter(PercentFormatter(1))

        plt.ylim([0,ylim])
        plt.xlim([0,3])
        plt.xticks([0,1,2,3])
        plt.ylabel("% of Trials")
        plt.xlabel("Time (s)")

    def PlotTrialTimeHist(self,color=None,ylim=None,alpha=0.5,shift_time=0):
        '''
        Plot Trial time histogram
        '''
        source = 'timeLastTargetAcquire'
        self._PlotTimeHist(color=color, ylim=ylim,source=source,alpha=alpha,shift_time=shift_time)

    def PlotTFAHist(self,color=None,ylim=None):
        '''
        Plot histogram of fist-touch time
        '''
        source = 'timeFirstTargetAcquire'
        self._PlotTimeHist(color=color, ylim=ylim,source=source)

    def PlotDITHist(self,color=None,ylim=None):
        '''
        Plot histogram of dial-in time
        '''
        #import seaborn as sns
        #from scipy.stats import gaussian_kde
        df = self.df
        df_centerOut = df.iloc[self.centerOutTrials]
        X_last = df_centerOut['timeLastTargetAcquire'][df_centerOut['isSuccessful']==True]
        X_first = df_centerOut['timeFirstTargetAcquire'][df_centerOut['isSuccessful']==True]
        X = list(X_last - X_first)
        #sns.kdeplot(X,shade=True,bw=25,color=color)
        plt.hist(X,bins=np.arange(0,3000,50), weights=np.ones(len(X)) / len(X),color=color,alpha=0.5)
        plt.gca().yaxis.set_major_formatter(PercentFormatter(1))

        plt.xlim([0,3000])
        plt.ylim([0,ylim])

    def PlotDeviationHist(self,color=None,ylim=None):
        df = self.df
        df_centerOut = df.iloc[self.centerOutTrials]
        X = list(df_centerOut['max_deviation'][df_centerOut['isSuccessful']==True])
        #sns.kdeplot(X,shade=True,bw=1,color=color[decoder])
        plt.hist(X,bins=np.arange(0,50,2), weights=np.ones(len(X)) / len(X),color=color,alpha=0.5)
        plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
        plt.xlim([0,50])
        plt.ylim([0,0.5])

    def PlotDistanceRatioHist(self,color=None,ylim=None):
        df = self.df
        df_centerOut = df.iloc[self.centerOutTrials]
        X = list(df_centerOut['distance_ratio'][df_centerOut['isSuccessful']==True])
        #sns.kdeplot(X,shade=True,bw=1,color=color[decoder])
        plt.hist(X,bins=np.arange(0,6,0.2), weights=np.ones(len(X)) / len(X),color=color,alpha=0.5)
        plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
        plt.xlim([0,6.0])
        plt.ylim([0,0.5])



    def PlotNeuralDynamics(self):
        '''
        Plot neural dynamics
        '''
        neuralDynamics = self.neuralDynamics

        plt.figure(figsize=(10,10))
        for ii in range(8):
            xline=neuralDynamics[ii,:,0]
            yline=neuralDynamics[ii,:,1]
            zline=neuralDynamics[ii,:,2]
            plt.plot(xline,yline)

        plt.axis("off")
        plt.title("Neural dynamics")

    def PlotDistanceToTarget(self,color='b',label='',ylim=8,shift_time=0):
        '''
        Plot distance-to-target profile
        '''
        dt = self.dt
        df = self.df

        meanD = self.meanDtT/10 #(cm)

        df_centerOut = df.iloc[self.centerOutTrials]
        FTT = df_centerOut['timeFirstTargetAcquire'][df_centerOut['isSuccessful']==True].mean()
        LTT = df_centerOut['timeLastTargetAcquire'][df_centerOut['isSuccessful']==True].mean()


        if ~np.isnan(FTT) and ~np.isnan(LTT):
            f=scipy.interpolate.interp1d(np.arange(0,len(meanD))*dt,meanD,kind='cubic')
            plt.plot(np.arange(FTT-shift_time,LTT-shift_time,0.1),f(np.arange(FTT,LTT,0.1)),color=color,linewidth=4,alpha=0.5)
            handle, = plt.plot(np.arange(0,FTT-shift_time,0.1),f(np.arange(0+shift_time,FTT,0.1)),color=color,linewidth=1,label=label,alpha=0.5)
            plt.legend()
        plt.xlabel("Time [ms]")
        plt.ylabel("Distance to Target [cm]")
        plt.xlim(0,1500)
        plt.xticks(range(0,1501,500))
        plt.ylim(0,ylim)
        ax = plt.gca()
        plt.yticks(ax.get_yticks())

    def PlotPSTH(self, neuronList = range(0,192,32),xlim=500,ylim=300):
        '''
        Plot PSTH
        '''
        PSTH = ([PSTH[self.extraBins:] for PSTH in self.PSTH])

        dt = self.dt
        #if len(plt.get_fignums())!=0:
        #    fig = plt.gcf()
        #    axes = []
        #    for ii in range(len(neuronList)):
        #        axes.append(fig.add_subplot(1,len(neuronList),ii+1))
        #    axes[0].get_shared_x_axes().join(*axes)
        #    axes[0].get_shared_y_axes().join(*axes)
        #else:
        fig, axes = plt.subplots(1, len(neuronList), figsize=(len(neuronList)*5,5),squeeze=False)
        axes = axes[0]
        for ii,neuronIdx in enumerate(neuronList):
            for targetIdx,target in enumerate(self.uniqueTarget):
                axes[ii].title.set_text('N '+str(neuronIdx))
                color = self.ColorOfTarget[tuple(target)]
                PlotPSTH(axes=axes[ii],PSTH=PSTH[targetIdx].T[neuronIdx]*(1000/dt),color=color,dt=self.dt)
            axes[ii].set_xlim([0,xlim])
            axes[ii].set_ylim([0,ylim])
        axes[0].set_ylabel('Firing Rate [sp/s]');


    def SynthesizeSpikes(self, neuralencoder,M1_delay=0,PMd_delay=0):
        '''
        synthesize spikes with specified neural encoder
        '''
        df = self.df
        ## get spike_count sequence
        Q_spike_counts = []
        for trialIdx in df.index:
            trialNum = df['trialNum'][trialIdx]
            target_p = df['target'][trialIdx]
            hand_p   = df['binHandPos'][trialIdx]
            cursor_p = df['binCursorPos'][trialIdx]
            trial_length = hand_p.shape[0]

            for tt in range(0,trial_length):
                encoder_input = hand_p[tt], target_p, cursor_p[tt]
                spike_counts,spike_rates = neuralencoder.encode(*encoder_input)
                Q_spike_counts.append(spike_counts)

        ##
        Q_spike_counts = np.array(Q_spike_counts)
        M1_spike_counts = Q_spike_counts[:,-1,0:96]
        PMd_spike_counts = Q_spike_counts[:,-1,96:192]

        ## take out the first trial
        trial_length = len(df['binHandPos'][0])
        M1_spike_counts = M1_spike_counts[trial_length+M1_delay:]
        PMd_spike_counts = PMd_spike_counts[trial_length+PMd_delay:]
        #Q_spike_counts = Q_spike_counts[trial_length-1+5:]

        ## take out the last trial
        trial_length = len(df['binHandPos'][len(df)-1])
        #Q_spike_counts = Q_spike_counts[:-trial_length+5]
        M1_spike_counts = M1_spike_counts[:-trial_length+M1_delay]
        PMd_spike_counts = PMd_spike_counts[:-trial_length+PMd_delay]

        Q_spike_counts = np.hstack([M1_spike_counts, PMd_spike_counts])

        ## reshape synthetic spikes
        list_pseudo_spikes = []
        for trialIdx in df.index[1:-1]:
            hand_p   = df['binHandPos'][trialIdx]
            trial_length = hand_p.shape[0]

            list_pseudo_spikes.append(Q_spike_counts[:trial_length])
            Q_spike_counts = Q_spike_counts[trial_length:]

        df = df.drop([0,len(df)-1]).reset_index(drop=True)
        df['binSpike'] = list_pseudo_spikes
        self.df = df
        self.centerOutTrials = np.where((np.stack(df['target'])!=[0,0]).any(axis=1))[0]
        self.centerInTrials  = df[~df.index.isin(self.centerOutTrials)].index
        self.uniqueTarget = self._UniqueTarget()
        self.TrialsOfTargets = self._TrialsOfTargets()
        self.ColorOfTarget = self._ColorOfTarget()

        self.numChannels=192

        self.successfulTrials = np.where(df['isSuccessful']==1)[0]
        self.trainingTrials  = self.successfulTrials
        self.testingTrials = self.successfulTrials
        self.numTrial = len(df)


        self.PSTH = spikesToPSTH(self,self.df['binSpike'])#[:,self.extraBins:]
        self.neuralDynamics = self._NeuralDynamics()


        #list_pseudo_spikes = []
        #for trialIdx in range(self.numTrial):
        #    pseudo_spikes = np.random.poisson(encoder.encodeFromData(self,trialIdx))
        #    assert pseudo_spikes.shape == self.df['binSpike'][trialIdx].shape
        #    list_pseudo_spikes.append(pseudo_spikes)
        #    ## Direct assignment is time-consuming. Therefore, I save into a list then assign later.
        #    #self.df['binSpike'][trialIdx] = pseudo_spikes #np.concatenate([pseudo_spikes,self.df['binSpike'][trialIdx][-1:]])
        #self.df['binSpike'] = list_pseudo_spikes
        #self.PSTH = spikesToPSTH(self,self.df['binSpike'])[:,self.extraBins:]
        #self.neuralDynamics = self._NeuralDynamics()

    def _PlotMovement(self,mode='centerout',source=None,acceptance_window=40,lim=125):
        assert source!=None
        assert source in ['binHandPos','binCursorPos']

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
        df=self.df
        if(mode=='all'):
            for trialIdx in range(self.numTrial):
                p = df[source][trialIdx][1:,0:2]

                plt.plot(p[:,0],p[:,1],'.',color='b')
        elif (mode=='centerout'):
            for target in self.uniqueTarget:
                color = self.ColorOfTarget[tuple(target)]
                trials = self.TrialsOfTargets[tuple(target)]
                for trialIdx in trials:
                    if df['isSuccessful'][trialIdx]:
                        p = df[source][trialIdx]
                        plt.plot(p[:,0],p[:,1],'.', color=color)

        elif (mode=='centerin'):

            trials = self.centerInTrials
            for trialIdx in trials:
                if df['isSuccessful'][trialIdx]:
                    p = df[source][trialIdx]
                    plt.plot(p[:,0],p[:,1],'.')

        elif (mode=='single'):
            for target in self.uniqueTarget:
                trials  = self.TrialsOfTargets[tuple(target)]
                trialIdx = None
                for trial in trials:
                    if df['isSuccessful'][trial] and (trial+1<self.numTrial) and df['isSuccessful'][trial+1]:
                        trialIdx = trial
                        break
                if trialIdx!=None:
                    color = self.ColorOfTarget[tuple(target)]
                    ## center-out
                    p = df[source][trialIdx][1:,0:2]
                    PlotGradientColorLine(p[:,0],p[:,1],color=color,linewidth=3)
                    ## center-back
                    trialIdx+=1
                    p = df[source][trialIdx][1:,0:2]
                    PlotGradientColorLine(p[:,0],p[:,1],color=color,linewidth=3)

        else:
            raise NotImplementedError("mode ({}) is not correct".format(mode))

        #ax.set_title(source)
        ax.set_xlabel("(mm)")
        ax.set_ylabel("(mm)")
        ax.set_xlim([-lim,lim])
        ax.set_ylim([-lim,lim])

    def PlotCursorMovement(self,mode='centerout',acceptance_window=40,lim=150):
        return self._PlotMovement(mode=mode,source='binCursorPos',acceptance_window=acceptance_window,lim=lim)

    def PlotHandMovement(self,mode='centerout',acceptance_window=40,lim=150):
        return self._PlotMovement(mode=mode,source='binHandPos',acceptance_window=acceptance_window,lim=lim)

    def _PlotVelProfile(self,source,ylim,group='target',color=None):
        '''
        group: {'target', 'all', None}
        '''
        axis = plt.gca()
        df = self.df
        dt = self.dt#self.neuraldecoder.refresh_rate

        if group=='all':
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
            avgV, stdV = arr.mean(axis = -1), arr.std(axis=-1)
            #avgV = avgV[::self.neuraldecoder.refresh_rate//self.neuralencoder.refresh_rate]
            plt.plot(range(0,len(avgV)*dt,dt),avgV, '.-', linewidth=2,color=color)
            #plt.fill_between(range(0,len(avgV)*dt,dt), avgV-stdV, avgV+stdV,alpha=0.1,color=color)
        elif group in ['target',None]:
            for target in self.uniqueTarget:
                c = self.ColorOfTarget[tuple(target)]
                allV=[]
                for trialIdx in self.TrialsOfTargets[tuple(target)]:
                    if df['isSuccessful'][trialIdx]:
                        v = df[source][trialIdx][:,0:2]
                        vel = np.linalg.norm(v,axis=1)
                        allV.append(vel)

                if len(allV)!=0:
                    lens = [len(i) for i in allV]
                    arr = np.ma.empty((np.max(lens),len(allV)))
                    arr.mask = True
                    for idx, l in enumerate(allV):
                        arr[:len(l),idx] = l
                    avgV, stdV = arr.mean(axis = -1), arr.std(axis=-1)

                    plt.plot(range(0,len(avgV)*dt,dt),avgV, '.-', linewidth=2,color=c)
                    plt.fill_between(range(0,len(avgV)*dt,dt), avgV-stdV, avgV+stdV,color=c,alpha=0.1)

                    if group==None:
                        [plt.plot(range(0,len(V)*dt,dt),np.array(V).T, '.-', linewidth=2,color=c, alpha=0.1) for V in allV]



        plt.xlim([0,1000])
        plt.xticks(range(0,1001,500))
        plt.xlabel('Time [ms]')

        plt.ylim([0,ylim])
        plt.yticks(range(0,ylim+1,100))
        plt.ylabel('Speed [mm/s]')

        plt.title(source)
        axis.spines['top'].set_visible(False)
        axis.spines['right'].set_visible(False)


    def PlotHandVelProfile(self,ylim=800,group='target',color=None):
        self._PlotVelProfile(source='binHandVel',ylim=ylim,group=group,color=color)

    def PlotCursorVelProfile(self,ylim=450,group='target',color=None):
        self._PlotVelProfile(source='binCursorVel',ylim=ylim,group=group,color=color)


    def PlotCursorVelProfileAll(self,ylim=800):
        #raise NotImplementedError
        df = self.df
        plt.gca()

        ax0 = plt.subplot2grid((3,3), (0,0))
        ax1 = plt.subplot2grid((3,3), (0,1))
        ax2 = plt.subplot2grid((3,3), (0, 2))
        ax3 = plt.subplot2grid((3,3), (1, 0))
        ax5 = plt.subplot2grid((3,3), (1,2))
        ax6 = plt.subplot2grid((3,3), (2,0))
        ax7 = plt.subplot2grid((3,3), (2, 1))
        ax8 = plt.subplot2grid((3,3), (2, 2))

        ax0.axis("off")
        ax1.axis("off")
        ax2.axis("off")
        ax3.axis("off")
        ax5.axis("off")
        ax6.axis("off")
        ax7.axis("off")
        ax8.axis("off")

        ax0.set_xlim([0,800])
        ax1.set_xlim([0,800])
        ax2.set_xlim([0,800])
        ax3.set_xlim([0,800])
        ax5.set_xlim([0,800])
        ax6.set_xlim([0,800])
        ax7.set_xlim([0,800])
        ax8.set_xlim([0,800])

        ax0.set_ylim([0,800])
        ax1.set_ylim([0,800])
        ax2.set_ylim([0,800])
        ax3.set_ylim([0,800])
        ax5.set_ylim([0,800])
        ax6.set_ylim([0,800])
        ax7.set_ylim([0,800])
        ax8.set_ylim([0,800])

        for target in self.uniqueTarget:
            c = self.ColorOfTarget[tuple(target)]
            allV=[]

            if(target[0]==0 and target[1]>0):
                ax = ax1
            if(target[0]==0 and target[1]<0):
                ax = ax7
            if(target[0]>0 and target[1]==0):
                ax = ax5
            if(target[0]<0 and target[1]==0):
                ax = ax3
            if(target[0]>0 and target[1]>0):
                ax = ax2
            if(target[0]>0 and target[1]<0):
                ax = ax8
            if(target[0]<0 and target[1]>0):
                ax = ax0
            if(target[0]<0 and target[1]<0):
                ax = ax6

            for trialIdx in self.TrialsOfTargets[tuple(target)]:
                v = df['binCursorVel'][trialIdx][:,0:2]
                vel = np.linalg.norm(v,axis=1)
                allV.append(vel)
            minBins = min([len(v) for v in allV])
            allV = [v[:minBins] for v in allV]
            avgV = np.mean(allV,axis=0)
            stdV = np.std(allV,axis=0)
            ax.plot(range(0,minBins*self.dt,self.dt),avgV, '-', linewidth=2,color=c)
            ax.fill_between(range(0,minBins*self.dt,self.dt),avgV-stdV,avgV+stdV,alpha=0.5,color=c)

    def PlotHandAccProfile(self,ylim=6000,b_plot_all=False):
        plt.gca()
        df = self.df
        plt.suptitle("Acc Profile",fontsize=30)
        ax_mean = plt.subplot2grid((3,3), (0,0),colspan=3, rowspan=3)
        ax_mean.tick_params(labelsize=18)
        ax_mean.set_xlim([0,800])
        ax_mean.set_xticks(range(0,801,200))
        ax_mean.set_ylim([-ylim,ylim])
        ax_mean.set_yticks(range(-ylim,ylim,ylim//4))
        ax_mean.set_xlabel('Time (ms)',fontsize=18)
        ax_mean.set_ylabel('Acc (mm/s^2)',fontsize=18)

        b = [1,0,-1]
        a = 2*(self.dt/1000)
        zi = scipy.signal.lfilter_zi(b,a)
        for target in self.uniqueTarget:
            c = self.ColorOfTarget[tuple(target)]
            allX=[]

            if len(self.TrialsOfTargets[tuple(target)])==0:
                continue

            for trialIdx in self.TrialsOfTargets[tuple(target)]:
                if not df['isSuccessful'][trialIdx]:
                    continue

                v = df['binHandVel'][trialIdx][:,0:2]
                x = np.linalg.norm(v,axis=1)
                allX.append(x)

            minBins = min([len(x) for x in allX])
            allX = np.array([x[:minBins] for x in allX])
            diffX = np.array([scipy.signal.lfilter(b,a,x,zi=zi*x[0])[0] for x in allX])
            avgAcc = np.mean(diffX,axis=0)
            ax_mean.plot(range(self.dt,(minBins-1)*self.dt,self.dt),avgAcc[2:], '.-', linewidth=2,color=c)

            if b_plot_all:
                ax_mean.plot(range(self.dt,(minBins-1)*self.dt,self.dt),diffX.T[2:], color=c,alpha=0.1)


    def PlotCursorAccProfile(self,ylim=6000,b_plot_all=False):

        #raise NotImplementedError
        plt.gca()
        df = self.df
        plt.suptitle("Acc Profile",fontsize=30)
        ax_mean = plt.subplot2grid((3,3), (0,0),colspan=3, rowspan=3)
        ax_mean.tick_params(labelsize=18)
        ax_mean.set_xlim([0,800])
        ax_mean.set_xticks(range(0,801,200))
        ax_mean.set_ylim([-ylim,ylim])
        ax_mean.set_yticks(range(-ylim,ylim,ylim//4))
        ax_mean.set_xlabel('Time (ms)',fontsize=18)
        ax_mean.set_ylabel('Acc (mm/s^2)',fontsize=18)

        b = [1,0,-1]
        a = 2*(self.dt/1000)
        zi = scipy.signal.lfilter_zi(b,a)
        for target in self.uniqueTarget:
            c = self.ColorOfTarget[tuple(target)]
            allX=[]

            if len(self.TrialsOfTargets[tuple(target)])==0:
                continue

            for trialIdx in self.TrialsOfTargets[tuple(target)]:
                if not df['isSuccessful'][trialIdx]:
                    continue

                v = df['binCursorVel'][trialIdx][:,0:2]
                x = np.linalg.norm(v,axis=1)
                allX.append(x)

            minBins = min([len(x) for x in allX])
            allX = np.array([x[:minBins] for x in allX])
            diffX = np.array([scipy.signal.lfilter(b,a,x,zi=zi*x[0])[0] for x in allX])
            avgAcc = np.mean(diffX,axis=0)
            ax_mean.plot(range(self.dt,(minBins-1)*self.dt,self.dt),avgAcc[2:], '.-', linewidth=2,color=c)

            if b_plot_all:
                ax_mean.plot(range(self.dt,(minBins-1)*self.dt,self.dt),diffX.T[2:], color=c,alpha=0.1)

    def Statistics(self):
        df = self.df
        statistics_df = pd.DataFrame(columns=['target','trial_time','first_touch_time','dial-in_time','max_deviation','distance_ratio','success_rate'])

        statistics = {}
        statistics['target'] = 'All'
        statistics['trial_time'] = df['timeLastTargetAcquire'][df['isSuccessful']==True].mean()
        statistics['first_touch_time'] = df['timeFirstTargetAcquire'][df['isSuccessful']==True].mean()
        statistics['dial-in_time'] = statistics['trial_time']-statistics['first_touch_time']
        statistics['max_deviation'] = df['max_deviation'][df['isSuccessful']==True].mean()
        statistics['distance_ratio'] = df['distance_ratio'][df['isSuccessful']==True].mean()
        statistics['success_rate'] = df['isSuccessful'].mean()
        statistics['# of total trials'] = len(df['isSuccessful'])

        statistics_df = statistics_df.append(pd.Series(statistics),ignore_index=True)


        statistics = {}
        statistics['target'] = 'Center-In'
        df_centerIn = df.iloc[self.centerInTrials] #df[~df.index.isin(self.centerOutTrials)]
        statistics['trial_time'] = df_centerIn['timeLastTargetAcquire'][df_centerIn['isSuccessful']==True].mean()
        statistics['first_touch_time'] = df_centerIn['timeFirstTargetAcquire'][df_centerIn['isSuccessful']==True].mean()
        statistics['dial-in_time'] = statistics['trial_time']-statistics['first_touch_time']
        statistics['max_deviation'] = df_centerIn['max_deviation'][df_centerIn['isSuccessful']==True].mean()
        statistics['distance_ratio'] = df_centerIn['distance_ratio'][df_centerIn['isSuccessful']==True].mean()
        statistics['success_rate'] = df_centerIn['isSuccessful'].mean()
        statistics['# of total trials'] = len(df_centerIn['isSuccessful'])


        statistics_df = statistics_df.append(pd.Series(statistics),ignore_index=True)


        statistics = {}
        df_centerOut = df.iloc[self.centerOutTrials]
        statistics['target'] = 'Center-Out'
        statistics['trial_time'] = df_centerOut['timeLastTargetAcquire'][df_centerOut['isSuccessful']==True].mean()
        statistics['first_touch_time'] = df_centerOut['timeFirstTargetAcquire'][df_centerOut['isSuccessful']==True].mean()
        statistics['dial-in_time'] = statistics['trial_time']-statistics['first_touch_time']
        statistics['max_deviation'] = df_centerOut['max_deviation'][df_centerOut['isSuccessful']==True].mean()
        statistics['distance_ratio'] = df_centerOut['distance_ratio'][df_centerOut['isSuccessful']==True].mean()
        statistics['success_rate'] = df_centerOut['isSuccessful'].mean()
        statistics['# of total trials'] = len(df_centerOut['isSuccessful'])

        statistics_df = statistics_df.append(pd.Series(statistics),ignore_index=True)



        for target in self.uniqueTarget:
            trials = self.TrialsOfTargets[tuple(target)]
            successful_trials = df[df['isSuccessful']==True].index & trials
            statistics = {}
            statistics['target'] = target
            statistics['trial_time'] = df['timeLastTargetAcquire'][successful_trials].mean()
            statistics['first_touch_time'] = df['timeFirstTargetAcquire'][successful_trials].mean()
            statistics['dial-in_time'] = statistics['trial_time']-statistics['first_touch_time']
            statistics['max_deviation'] = df['max_deviation'][successful_trials].mean()
            statistics['distance_ratio'] = df['distance_ratio'][successful_trials].mean()
            statistics['success_rate'] = df['isSuccessful'][trials].mean()
            statistics['# of total trials'] = len(df['isSuccessful'][trials])

            statistics_df = statistics_df.append(pd.Series(statistics),ignore_index=True)

        statistics_df = statistics_df.round(2)
        return statistics_df

    def sparseSpikeRaster(self):
        binSpikeRate = self.df['binSpike']
        COO=[]
        COO_2=[]
        for trialIdx in range(self.numTrial):
            numBins, numNeurons = binSpikeRate[trialIdx].shape
            row_ind, col_ind = [], []
            row_ind_2, col_ind_2 = [], []
            for neuronIdx in range(numNeurons):
                spikeRaster = gen.inh_poisson_generator(binSpikeRate[trialIdx][:,neuronIdx]*1000/dt,range(0,len(binSpikeRate[trialIdx][:,neuronIdx])*dt,dt),len(binSpikeRate[trialIdx][:,neuronIdx])*dt,array=True)
                spikeRaster = [int(e) for e in spikeRaster]
                if(neuronIdx<96):
                    row_ind += [neuronIdx]*len(spikeRaster)
                    col_ind += spikeRaster
                else:
                    row_ind_2 += [neuronIdx]*len(spikeRaster)
                    col_ind_2 += spikeRaster
            data = [1]*len(col_ind)
            COO.append(sparse.coo_matrix((data, (row_ind,col_ind) )))
            data_2 = [1]*len(col_ind_2)
            COO_2.append(sparse.coo_matrix((data_2, (row_ind_2,col_ind_2) )))

        self.df['spikeRaster']=COO
        self.df['spikeRaster2']=COO_2


    def next_train_batch(self,batch_size=0,M1_delay=0,PMd_delay=0
                        ,seq2seq=False,stateful=False
                        ,replace=False,randomSelect=True):
        '''

        '''
        numBins = self.numBins
        extraBins = self.extraBins
        trainingTrials = self.trainingTrials
        df = self.df
        max_delay = max(M1_delay, PMd_delay)

        if stateful==True:
            rt_hand_p = []
            rt_hand_v = []
            rt_cursor_p = []
            rt_cursor_v = []
            rt_n = []
            for trialIdx in reversed(range(1,497)):
                hand_p = df['binHandPos'][trialIdx][:numBins,0:2]
                hand_v = df['binHandVel'][trialIdx][:numBins,0:2]
                cursor_p = df['binCursorPos'][trialIdx][:numBins,0:2]
                cursor_v = df['binCursorVel'][trialIdx][:numBins,0:2]

                M1 = np.vstack([df['binSpike'][trialIdx-1][-M1_delay:,:96],df['binSpike'][trialIdx][:numBins-M1_delay,:96]])
                PMd = np.vstack([df['binSpike'][trialIdx-1][-PMd_delay:,96:],df['binSpike'][trialIdx][:numBins-PMd_delay,96:]])
                n = np.hstack([M1,PMd])
                rt_hand_p.append(hand_p)
                rt_hand_v.append(hand_v)
                rt_n.append(n)

            rt={}
            rt['n']=np.array(rt_n)
            rt['hand_p']=np.array(rt_hand_p)
            rt['hand_v']=np.array(rt_hand_v)
            rt['cursor_p']=np.array(rt_cursor_p)
            rt['cursor_v']=np.array(rt_cursor_v)

            return rt


        elif stateful==False:

            if((batch_size==0) or (batch_size>self.numTrain)):
                batch_size = self.numTrain
            print("Use {} trials as training data".format(batch_size))

            batchIdx = np.random.choice(trainingTrials,batch_size,replace=replace) if randomSelect else trainingTrials[:batch_size]
            batchIdx = np.sort(batchIdx)

            rt_n = []
            rt_hand_p = []
            rt_hand_v = []
            rt_cursor_p = []
            rt_cursor_v = []

            for trialIdx in batchIdx:

                hand_p = df['binHandPos'][trialIdx][:,0:2]
                hand_v = df['binHandVel'][trialIdx][:,0:2]
                cursor_p = df['binCursorPos'][trialIdx][:,0:2]
                cursor_v = df['binCursorVel'][trialIdx][:,0:2]
                n = df['binSpike'][trialIdx]

                for timestamp in range(max_delay, len(hand_p)-extraBins):
                    rt_hand_p.append(hand_p[timestamp:timestamp+extraBins+1])
                    rt_hand_v.append(hand_v[timestamp:timestamp+extraBins+1])
                    rt_cursor_p.append(cursor_p[timestamp:timestamp+extraBins+1])
                    rt_cursor_v.append(cursor_v[timestamp:timestamp+extraBins+1])
                    if seq2seq:
                        rt_n.append( np.concatenate([n[timestamp-M1_delay:timestamp-M1_delay+extraBins,:96], n[timestamp-PMd_delay:timestamp-PMd_delay+extraBins, 96:]],axis=1)  )
                    else:
                        rt_n.append( np.concatenate([n[timestamp-M1_delay+extraBins,:96], n[timestamp-PMd_delay+extraBins, 96:]],axis=0)  )

            rt={}
            rt['n']=np.array(rt_n)
            rt['hand_p']=np.array(rt_hand_p)
            rt['hand_v']=np.array(rt_hand_v)
            rt['cursor_p']=np.array(rt_cursor_p)
            rt['cursor_v']=np.array(rt_cursor_v)

            return rt

    def RNNTrainingData(self):
        raise NotImplementedError
        df = self.df
        trials = self.trainingTrials
        numBins = self.numBins+self.extraBins

        rt_t =  []
        rt_p =  []
        rt_v =  []
        rt_n =  []

        for trialIdx in trials:
            t = df['target'][trialIdx].reshape(1,2)
            p = df['binHandPos'][trialIdx][1:,0:2]
            v = df['binHandVel'][trialIdx][:,0:2]
            t = np.repeat(t,v.shape[0],axis=0)
            n = df['binSpike'][trialIdx]
            rt_t.append(t[0:numBins])
            rt_p.append(p[0:numBins])
            rt_v.append(v[0:numBins])
            rt_n.append(n[0:numBins])

        rt_t=np.array(rt_t)
        rt_p=np.array(rt_p)
        rt_v=np.array(rt_v)
        rt_n=np.array(rt_n)

        return (rt_t,rt_p,rt_v,rt_n)

    def PlotAll(self):
        fig,axes = plt.subplots(2,3,figsize=(15,10))
        plt.sca(axes[0,0])
        self.PlotDistanceToTarget(ylim=15)
        adjust_spines(axes[0,0])
        plt.sca(axes[0,1])
        self.PlotHandMovement('centerout',lim=160)
        plt.sca(axes[0,2])
        self.PlotHandVelProfile(ylim=800)
        plt.sca(axes[1,0])
        self.PlotTrialTimeHist(ylim=0.5)
        adjust_spines(axes[1,0])
        plt.sca(axes[1,1])
        self.PlotCursorMovement('centerout',lim=160)
        plt.sca(axes[1,2])
        self.PlotCursorVelProfile(ylim=800)
        adjust_spines(axes[1,2])
        #plt.tight_layout()
        #plt.show()

    def downSample(self,sampling_rate=1):
        df = self.df
        dt = self.dt*sampling_rate
        for trialIdx in reversed(range(0,len(df))):
            df.at[trialIdx,'binState'] = binning(df['binState'][trialIdx],sampling_rate,'first')
            df.at[trialIdx,'binSpike'] = binning(df['binSpike'][trialIdx],sampling_rate,'sum')
            df.at[trialIdx,'binHandPos'] = binning(df['binHandPos'][trialIdx],sampling_rate,'first')
            df.at[trialIdx,'binHandVel'] = binning(df['binHandVel'][trialIdx],sampling_rate,'sum')/sampling_rate
            df.at[trialIdx,'binCursorPos'] = binning(df['binCursorPos'][trialIdx],sampling_rate,'first')
            df.at[trialIdx,'binCursorVel'] = binning(df['binCursorVel'][trialIdx],sampling_rate,'sum')/sampling_rate

        #df = df.drop(0).reset_index(drop=True)

        self.dt = dt
        self.df = df

        self.successfulTrials = np.where(df['isSuccessful']==1)[0]
        self.numBins = np.min([df['binSpike'][i].shape[0] for i in range(len(df))])
        self.numSuccessfulTrial =  numSuccessfulTrial = len(self.successfulTrials)
        self.trainingTrials  = self.successfulTrials
        self.testingTrials = self.successfulTrials

        self.numTrial = len(self.df)
        self.numTest = len(self.testingTrials)
        self.numTrain = len(self.trainingTrials)

        self.centerOutTrials = np.where((np.stack(df['target'])!=[0,0]).any(axis=1))[0]
        self.centerInTrials  = df[~df.index.isin(self.centerOutTrials)].index

        self.uniqueTarget = self._UniqueTarget()
        self.TrialsOfTargets = self._TrialsOfTargets()

        self._CalculateDistanceToTarget()
        self._CalculateDeviation()
        self._CalculateDistanceRatio()

        self.meanDtT = self._getMeanDtT()

        self.PSTH = spikesToPSTH(self,self.df['binSpike'])
        self.numBins = np.min([df['binSpike'][i].shape[0] for i in range(len(df))])
        self.neuralDynamics = self._NeuralDynamics()
    
    def savePSTHForjPCA(self,folder_path):
        PSTH = self.PSTH
        numBins = self.numBins
        dt = self.dt
        time   =np.array(range(0,numBins*dt,dt)).reshape(numBins,1)
        dt     = np.dtype(dtype=[('A', 'O'), ('times', 'O')])
        x = np.array([(PSTH[0][:numBins,:],time)
                      ,(PSTH[1][:numBins,:],time)
                      ,(PSTH[2][:numBins,:],time)
                      ,(PSTH[3][:numBins,:],time)
                      ,(PSTH[4][:numBins,:],time)
                      ,(PSTH[5][:numBins,:],time)
                      ,(PSTH[6][:numBins,:],time)
                      ,(PSTH[7][:numBins,:],time)
                     ]
                     ,dtype=dt)
        #if(rt['encoder']=='real'):
        #    name='real'+"_"+rt['testName']
        #else:
        #    name = rt['encoder'].name+"_"+rt['testName']
        sio.savemat(folder_path, {'Data':x})



class RigCDataset(Dataset):
    def __init__(self
                ,filename=None
                ,dt=20
                ,extraPeriod=0
                ,bRecenter=0
                ,bSmoothPos=False
                ,bDropNonFiringNeurons=False
                ,bDropInvisible = True
                ,bIncludeFail=True
                ,verbose=False):

        assert extraPeriod%dt==0,"exptraPeriod need to be multiple of dt"
        self.filename        = filename
        self.dt              = dt
        self.bSmoothPos      = bSmoothPos
        self.bDropInvisible = bDropInvisible
        self.bIncludeFail = bIncludeFail
        self.verbose = verbose
        data= sio.loadmat(filename,squeeze_me=True)
        R   = data['R']
        df  = pd.DataFrame(R)

        dropTrialList = []
        df,_  = self.transformDataFrame(df,dropTrialList)
        if(extraPeriod):
            df = self._appendTrials(df,extraPeriod,dropTrialList)
        df = self._smoothPos(df)

        df,_ = self.binTrial(df,dropTrialList)

        dropTrialList.append(0)
        dropTrialList.append(len(df)-1)
        print("Drop trials :",dropTrialList)
        df=df.drop(dropTrialList)
        df=df.reset_index(drop=True)

        self.df = df
        self.metadata = dict({'acceptance_window':df['startTrialParams'][0]['winBox'].tolist()[0]*2})
        self._check()

        extraBins   = int(extraPeriod/dt)
        numBins     = np.min([df['binSpike'][i].shape[0] for i in np.arange(len(df))])-extraBins

        self.numBins         = numBins
        self.extraBins       = extraBins

        self.numTrial        =  numTrial   = len(df)
        self.R               = R

        self.successfulTrials = np.where(df['isSuccessful']==1)[0]
        self.numSuccessfulTrial =  numSuccessfulTrial = len(self.successfulTrials)
        self.trainingTrials  = self.successfulTrials
        self.testingTrials = self.successfulTrials
        self.numTrain = numTrain = len(self.trainingTrials)
        self.numTest =  numTest = len(self.testingTrials)


        assert len(self.trainingTrials)==self.numTrain
        assert len(self.testingTrials)==self.numTest

        self.centerOutTrials = np.where((np.stack(df['target'])!=[0,0]).any(axis=1))[0]
        self.centerInTrials  = df[~df.index.isin(self.centerOutTrials)].index

        self.numNeuron       = df['spikeRaster'][0].shape[1] if 'spikeRaster' in self.df.keys() else 0
        self.numChannels     = self.numNeuron

        self.uniqueTarget    = self._UniqueTarget()
        self.TrialsOfTargets = self._TrialsOfTargets()
        self.ColorOfTarget   = self._ColorOfTarget()

        self._CalculateDistanceToTarget()
        self._CalculateDeviation()
        self._CalculateDistanceRatio()

        self.PSTH = spikesToPSTH(self,self.df['binSpike'])#[:,self.extraBins:]
        self.neuralDynamics = self._NeuralDynamics()
        self.meanDtT = self._getMeanDtT()

        if verbose:
            self.summary()

    def transformDataFrame(self,df,dropTrialList):
        '''
        follow the similar data preprocessing in RigC matlab.
        - drop trial if invisible length is larger than 400ms
        - interpolate positions if invisible
        '''
        if self.bDropInvisible and (df.keys()=='numMarkers').any():
            for i in range(len(df)):
                validIdx = np.where(df['numMarkers'][i]==1)[0]
                if(validIdx.size<500):
                    print("ValidIdx less than 500, drop trial {}".format(i))
                    dropTrialList.append(i)
                    continue
                else:
                    maxIdx = df['timeTrialEnd'][i]

                    last_time = df['timeLastTargetAcquire'][i] if df['isSuccessful'][i] else len(df['numMarkers'][i])

                    pulse_seq = np.diff(np.concatenate([[1],df['numMarkers'][i][:last_time],[1]]))
                    invisible_length = np.where(pulse_seq==2)[0]-np.where(pulse_seq==-2)[0]

                    if( len(invisible_length)!=0 and max(invisible_length)>400): ## drop the trials with 400ms invisible hand position
                        print("Drop {} due to invisible length {}  > 400ms".format(i,invisible_length))
                        dropTrialList.append(i)
                    else:
                        ## interpolate the handposition
                        df['handPos'][i][:,(np.where(df['numMarkers'][i]==-1)[0])] = np.nan
                        df['cursorPos'][i][:,(np.where(df['numMarkers'][i]==-1)[0])] = np.nan

                        df['handPos'][i][0]   = pd.Series(df['handPos'][i][0,:]).interpolate(kind='cubic',limit_direction='both').values
                        df['handPos'][i][1]   = pd.Series(df['handPos'][i][1,:]).interpolate(kind='cubic',limit_direction='both').values
                        df['handPos'][i][2]   = pd.Series(df['handPos'][i][2,:]).interpolate(kind='cubic',limit_direction='both').values
                        df['cursorPos'][i][0]   = pd.Series(df['cursorPos'][i][0,:]).interpolate(kind='cubic',limit_direction='both').values
                        df['cursorPos'][i][1]   = pd.Series(df['cursorPos'][i][1,:]).interpolate(kind='cubic',limit_direction='both').values
                        df['cursorPos'][i][2]   = pd.Series(df['cursorPos'][i][2,:]).interpolate(kind='cubic',limit_direction='both').values
                        df['decodePos'][i][0]   = pd.Series(df['decodePos'][i][0,:]).interpolate(kind='cubic',limit_direction='both').values
                        df['decodePos'][i][1]   = pd.Series(df['decodePos'][i][1,:]).interpolate(kind='cubic',limit_direction='both').values
                        df['decodePos'][i][2]   = pd.Series(df['decodePos'][i][2,:]).interpolate(kind='cubic',limit_direction='both').values

        if(df.keys()=='target').any():
            df['target'] = [ t[0:2] for t in df['target']]
        else:
            df['target'] = [ t['posTarget'].item()[0:2] for t in df['startTrialParams']]

        if(df.keys()=='handPos').any():
            df['handPos']   = [ p[0:2].T for p in df['handPos']]
        if(df.keys()=='cursorPos').any():
            df['cursorPos'] = [ p[0:2].T for p in df['cursorPos']]

        # drop trials if start cursor position is far away from target than 100cm
        for i in range(len(df)):
            if np.linalg.norm(df['cursorPos'][i][0]-df['target'][i])>1000:
                print("Drop {} | ,Cursor started beyond 100 cm from target".format(i))
                dropTrialList.append(i)


        df['target'] = [ tuple(np.around(np.float32(t),2)) for t in df['target']]

        #drop center-back trials when the monkey failed a center-out reach, i.e. exclude successes of (0,0) when the previous target was a failure or NaN.
        for ii in range(1,len(df)):
            if (self.bIncludeFail==False):
                if (df['target'][ii]==(0,0) ) and (df['isSuccessful'][ii-1]==False):
                    dropTrialList.append(ii)

        for ii in range(len(df)):
            if (self.bIncludeFail==False):
                if df['isSuccessful'][ii]==False:
                    dropTrialList.append(ii)




        df=df.reset_index(drop=True)


        df['spikeRaster']   = [ s.toarray().T for s in df['spikeRaster']]

        if 'spikeRaster2' in df.keys():
            df['spikeRaster']  = [ np.hstack([df['spikeRaster'][ii],s.T.toarray()]) for ii,s in enumerate(df['spikeRaster2'])]
        else:
            df['spikeRaster']  = [ np.hstack([s,np.zeros(s.shape)]) for s in df['spikeRaster']]


        for ii in range(len(df)):
            if df['isSuccessful'][ii] and (df['timeLastTargetAcquire'][ii]<200) and (df['target'][ii]!=[0,0]):
                print("Trial {} has very short time ({}) to acquire target. May due to reseed in the online experiment".format(ii,df['timeLastTargetAcquire'][ii]))

        ## shift 1 due to different indexing method between matlab and python
        df['timeTargetOn'] = df['timeTargetOn']-1 #
        df['timeTargetAcquire'] = df['timeTargetAcquire']-1 #
        df['timeTrialHeld'] = df['timeTargetHeld']-1 #
        df['timeTrialEnd'] = df['timeTrialEnd']-1 #
        df['timeCerebusStart'] = df['timeCerebusStart']-1 #
        df['timeCerebusEnd'] = df['timeCerebusEnd']-1 #
        df['timeFirstTargetAcquire'] = df['timeFirstTargetAcquire']-1 #
        df['timeLastTargetAcquire'] = df['timeLastTargetAcquire']-1 #
        df['timeDialIn'] = df['timeLastTargetAcquire']-df['timeFirstTargetAcquire']#


        return df, dropTrialList

    def binTrial(self,df,dropTrialList):
        '''
        bin spikes, cursor position and hand position, and calculate binned hand velocity and cursor velocity
        '''
        if(self.verbose>0):
            print('Start binning spikes')

        dt          = self.dt

        df['binState'] = ""
        df['binSpike'] = ""

        df['binCursorPos'] = ""
        df['binHandPos']   = ""
        df['binDecodePos'] = ""

        df['binHandVel'] = ""
        df['binCursorVel'] = ""

        for trialIdx in reversed(range(1,len(df))):
            trialNum = df['trialNum'][trialIdx]
            preTrialNum = df['trialNum'][trialIdx-1]
            if (trialNum-1==preTrialNum):
                timeStart = df['timeTargetOn'][trialIdx]
                timeEnd = len(df['cursorPos'][trialIdx]) #df['timeTrialEnd'][trialIdx]+1
                df.at[trialIdx,'binState']      = binning(df['state'][trialIdx][timeStart:timeEnd+dt,None],dt,'last')
                df.at[trialIdx,'binSpike']      = binning(df.loc[trialIdx,'spikeRaster'][timeStart:timeEnd], dt, 'sum')
                binCursorPos = binning(np.vstack([df['cursorPos'][trialIdx-1][-dt:],df['cursorPos'][trialIdx]])[timeStart:timeEnd+dt], dt,'last')
                df.at[trialIdx,'binCursorPos']  = binCursorPos[1:]
                df.at[trialIdx,'binCursorVel']  = (binCursorPos[1:]-binCursorPos[:-1])/dt*1000
                binHandPos = binning(np.vstack([df['handPos'][trialIdx-1][-dt:],df['handPos'][trialIdx]])[timeStart:timeEnd+dt], dt,'last')
                df.at[trialIdx,'binHandPos']    = binHandPos[1:]
                df.at[trialIdx,'binHandVel']    = (binHandPos[1:]-binHandPos[:-1])/dt*1000
            else:
                print("Drop {} due to no previous trial for binning".format(trialIdx))
                dropTrialList.append(trialIdx)

        return df,dropTrialList

    def binningAcceleration(self,verbose=0):
        raise NotImplementedError
        if(verbose>0):
            print('Start binning Hand/Cursor acceleration')
        df          = self.df
        numTrial    = self.numTrial
        dt          = self.dt
        df['binHandAcc'] = ""
        df['binCursorAcc'] = ""
        for trialIdx in range(numTrial):
            df.at[trialIdx,'binHandAcc'] = np.vstack([(df['binHandVel'][trialIdx][1:,:] - df['binHandVel'][trialIdx][:-1,:])/dt*1000 ])
            df.at[trialIdx,'binCursorAcc'] = np.vstack([(df['binCursorVel'][trialIdx][1:,:] - df['binCursorVel'][trialIdx][:-1,:])/dt*1000 ])



    def recenter(self,df):
        raise NotImplementedError
        meanOfHandPos = np.mean(np.concatenate(df['handPos'],axis=0),axis=0)
        print('Mean of hand position:',meanOfHandPos)
        for trialIdx in range(len(df)):
            df['handPos'][trialIdx] = df['handPos'][trialIdx]-meanOfHandPos

        meanOfHandPos = np.mean(np.concatenate(df['handPos'],axis=0),axis=0)
        print('After recenter, Mean of hand position:',meanOfHandPos)
        return df



class LicoriceDataset(Dataset):
    def __init__(self
                ,log_path
                ,dt
                ,verbose=False
                ,delay_compensation = 0
                ,extraPeriod=0):

        assert dt>0
        assert log_path!=None
        assert extraPeriod>=0

        try:
            with open(os.path.join(log_path,'cfg.json'),'r') as f:
                self.metadata = json.load(f)
        except IOError:
            print("cfg.json is not accessible in "+log_path)
            self.metadata = None

        file = self._loadFiles(log_path)
        input_axis_vXY = file[:,0:2]
        input_axis_pXY = file[:,2:4]
        decoded_axis_vXY = file[:,4:6]
        decoded_axis_pXY = file[:,6:8]
        cursor_pXY = file[:,8:10]
        target_pXY = file[:,10:12]
        tick_count = file[:,12]
        spike_train= file[:,13:13+192]
        state_task = file[:,205]
        #print(state_task[0].shape)
        PV_fail    = file[:,206]
        #spike_rate = file[:,207:207+192]
        task_states = {
                 'begin'     : 1,
                 'active'    : 2,
                 'hold'      : 3,
                 'success'   : 4,
                 'fail'      : 5,
                 'end'       : 6 }
        pixelPitch = 3.1125
        #firstTouchIdx = np.nan
        trialInfo = []
        cur_state=task_states['begin']
        #holdTime=0 ## (because there is a bug)
        isSuccessful = None
        for ii in range(1,len(state_task)):
            if (state_task[ii-1]==task_states['begin'] and state_task[ii]==task_states['active']):
                timeTargetOn = ii
                timeTargetAcquire = []
                timeTargetHeld = None
                timeTrialEnd = None

                #holdTime = 0
            elif(state_task[ii-1]==task_states['active'] and state_task[ii]==task_states['hold']):
                timeTargetAcquire.append(ii)
                #if(holdTime==0):
                #    firstTouchIdx = ii
                #    holdTime+=1
            elif(state_task[ii-1]==task_states['hold'] and state_task[ii]==task_states['success']):
                isSuccessful = True
                endIdx=ii
            elif(state_task[ii-1]==task_states['active'] and state_task[ii]==task_states['fail']):
                isSuccessful = False
                endIdx=ii
            elif((state_task[ii-1]==task_states['fail'] and state_task[ii]==task_states['end']) or
                (state_task[ii-1]==task_states['success'] and state_task[ii]==task_states['end'])):
                assert isSuccessful != None
                timeTargetHeld = ii
                trialInfo.append([timeTargetOn,timeTargetAcquire,timeTargetHeld,timeTrialEnd,isSuccessful])

        df = pd.DataFrame()

        trialNum = 0
        for timeTargetOn,timeTargetAcquire,timeTargetHeld,timeTrialEnd,isSuccessful in trialInfo:

            target    = target_pXY[timeTargetOn]
            handPos   = input_axis_pXY[timeTargetOn:timeTargetHeld+1]
            handVel   = input_axis_vXY[timeTargetOn:timeTargetHeld+1]
            cursorPos = cursor_pXY[timeTargetOn:timeTargetHeld+1]
            binState  = binning(state_task[timeTargetOn:timeTargetHeld+1,None],dt,'all')

            binSpike   = binning(spike_train[timeTargetOn+delay_compensation:timeTargetHeld+1+delay_compensation],dt,'all')#[1:]

            binHandPos  = binning(input_axis_pXY[timeTargetOn-2*dt:timeTargetHeld+1],dt,'all')
            binHandVel  = (binHandPos[1:]-binHandPos[:-1])/dt*1000 # (mm/s)
            binHandAcc  = (binHandVel[1:]-binHandVel[:-1])/dt*1000 # (mm/s)
            binHandVel  = binHandVel[1:]
            binHandPos  = binHandPos[2:]

            act = binHandAcc[1:]

            binCursorPos = binning(cursor_pXY[timeTargetOn-dt:timeTargetHeld+1],dt,'all')
            binCursorVel  = (binCursorPos[1:]-binCursorPos[:-1])/dt*1000 # (mm/s)
            binCursorPos = binCursorPos[1:]

            binCursorPosDistance = np.sqrt(np.sum(np.power(binCursorPos-target,2),axis=1))

            timeTargetAcquire = np.array(timeTargetAcquire) - timeTargetOn
            timeTargetHeld = timeTargetHeld - timeTargetOn
            timeTargetOn = 0

            df=df.append({
                          'trialNum': trialNum,
                          'timeTargetOn':timeTargetOn,
                          'isSuccessful':isSuccessful,
                          'timeTargetAcquire':timeTargetAcquire,
                          'timeTargetHeld':timeTargetHeld,
                          'timeFirstTargetAcquire': timeTargetAcquire[0] if len(timeTargetAcquire) else float('nan'),
                          'timeLastTargetAcquire': timeTargetAcquire[-1] if len(timeTargetAcquire) else float('nan'),
                          'timeDialIn': timeTargetAcquire[-1]-timeTargetAcquire[0] if len(timeTargetAcquire) else float('nan'),
                          'trialLength': timeTargetHeld,

                          'target':tuple(np.around(target,2)), # use the same datatype as in rigC data
                          'binState':binState,
                          'binSpike':binSpike,
                          'handPos':handPos,
                          'handVel':handVel,
                          'binHandPos':binHandPos,
                          'binHandVel':binHandVel,
                          'binHandAcc':binHandAcc,
                          'act':act,
                          'cursorPos': cursorPos,
                          'binCursorPos':binCursorPos,
                          'binCursorVel':binCursorVel,
                          'binCursorPosDistance':binCursorPosDistance,
                         },ignore_index=True)
            trialNum +=1

        dropList = []
        #dropList = np.where([len(df['handPos'][ii])<100 for ii in range(len(df))])[0]
        #dropList = np.append(dropList,np.where([(df['binHandVel'][ii]>1000).any() for ii in range(len(df))])[0])
        dropList = np.append(dropList,np.array([0,len(df)-1])) # drop the first and last trial
        if len(dropList):
            print("Drop list:",dropList)
        df = df.drop(dropList).reset_index(drop=True)

        self.dt = dt
        self.verbose = verbose

        self.extraBins = int(extraPeriod/dt)
        self.numBins = np.min([df['binSpike'][i].shape[0] for i in np.arange(len(df))])

        if(extraPeriod):
            df = self._appendTrials(df,extraPeriod)

        self.df = df
        self.numTrial = len(df)
        self.centerOutTrials = np.where((np.stack(df['target'])!=[0,0]).any(axis=1))[0]
        self.centerInTrials  = df[~df.index.isin(self.centerOutTrials)].index
        self.uniqueTarget = self._UniqueTarget()
        self.TrialsOfTargets = self._TrialsOfTargets()
        self.ColorOfTarget = self._ColorOfTarget()

        self.numChannels=192

        self.successfulTrials = np.where(df['isSuccessful']==1)[0]
        self.trainingTrials  = self.successfulTrials
        self.testingTrials = self.successfulTrials

        self.numTrain = numTrain = len(self.trainingTrials)
        self.numTest =  numTest = len(self.testingTrials)


        self._CalculateDistanceToTarget()
        self._CalculateDeviation()
        self._CalculateDistanceRatio()

        self.PSTH = spikesToPSTH(self,self.df['binSpike'])
        self.neuralDynamics = self._NeuralDynamics()

        self.meanDtT = self._getMeanDtT()

        self._check()

    def _loadFiles(self,path):
        files = [f for f in os.listdir(path) if f[0:3]=="log"]
        files.sort()
        return np.concatenate([np.load(os.path.join(path,f)) for f in files],axis=0)


class RLDataset(Dataset):
    def __init__(self,model,env_kwargs,deterministic=False, verbose=False,M1_delay=0,PMd_delay=0):
        '''
        env_kwargs: env or env_kwargs
        '''

        self.metadata = env_kwargs

        self.verbose=False
        self.deterministic = deterministic
        self.model = model
        delay_compensation = max(M1_delay,PMd_delay)

        if isinstance(env_kwargs,dict):
            self.env_kwargs = env_kwargs
            encoder_dt = env_kwargs['exp_setting']['encoder_refresh_rate']
            decoder_dt = env_kwargs['exp_setting']['decoder_refresh_rate']
            if env_kwargs['exp_setting']['b_bypass']==True:
                assert encoder_dt==decoder_dt
            n_stack = env_kwargs['n_stack']
            env = make_vec_env(CenteroutEnv,env_kwargs=env_kwargs,n_envs=1)
            if n_stack>1:
                env = VecFrameStack(env,n_stack=n_stack)
            self.env = env
        else:
            self.env = env = env_kwargs
            encoder_dt = env.get_attr('neural_encoder')[0].refresh_rate
            decoder_dt = env.get_attr('neural_decoder')[0].refresh_rate
        if verbose:
            env.get_attr('summary')[0]()


        traj = self._generateTraj()


        task_state = traj['task_state']
        task_states = self.env.envs[0].env.task.task_states
        firstTouchIdx = np.nan
        trialInfo = []

        for ii in range(1,len(task_state)):
            #print(ii,task_state[ii-1],task_state[ii])
            if (task_state[ii-1]==task_states['begin'] and task_state[ii]==task_states['active']):
                timeTargetOn = ii
                timeTargetAcquire = []
                timeTargetHeld = None
                timeTrialEnd = None

                #holdTime = 0
                isSuccessful = None
            elif (task_state[ii-1]==task_states['active'] and task_state[ii]==task_states['hold']):
                timeTargetAcquire.append(ii)
                #if(holdTime==0):
                #    firstTouchIdx = ii
                #    holdTime+=1
            elif (task_state[ii-1]==task_states['hold'] and task_state[ii]==task_states['success']):
                isSuccessful = True
                timeTargetHeld = ii
            elif (task_state[ii-1]==task_states['hold'] and task_state[ii]==task_states['fail']):
                isSuccessful = False
                timeTargetHeld = ii
            elif (task_state[ii-1]==task_states['active'] and task_state[ii]==task_states['fail']):
                isSuccessful = False
                timeTargetHeld = ii
            elif ((task_state[ii-1]==task_states['success'] and task_state[ii]==task_states['end']) or
                 (task_state[ii-1]==task_states['fail'] and task_state[ii]==task_states['end'])):
                assert isSuccessful != None
                assert timeTargetHeld-timeTargetOn < 3750//encoder_dt
                trialInfo.append([timeTargetOn,timeTargetAcquire,timeTargetHeld,timeTrialEnd,isSuccessful])

        df = pd.DataFrame()

        trialNum = 0
        for timeTargetOn,timeTargetAcquire,timeTargetHeld,timeTrialEnd,isSuccessful in trialInfo[1:-1]: # drop the first and the last one
            act       = traj['act'][timeTargetOn:timeTargetHeld+1]
            target    = traj['target_position'][timeTargetOn]

            binState  = np.array(task_state[timeTargetOn:timeTargetHeld+1]).reshape(-1,1)
            binCursorPos = traj['cursor_position'][timeTargetOn:timeTargetHeld+1]
            binHandPos = traj['hand_position'][timeTargetOn:timeTargetHeld+1]

            binCursorVel = traj['cursor_velocity'][timeTargetOn:timeTargetHeld+1]*1000/decoder_dt
            binHandVel = traj['hand_velocity'][timeTargetOn:timeTargetHeld+1]*1000/encoder_dt

            binSpike = traj['spike_train'][timeTargetOn+delay_compensation:timeTargetHeld+1+delay_compensation]

            binCursorPosDistance = np.sqrt(np.sum(np.power(binCursorPos-target,2),axis=1))

            timeTargetAcquire = np.array(timeTargetAcquire) - timeTargetOn
            timeTargetHeld = timeTargetHeld - timeTargetOn
            timeTargetOn = 0
            #timeTrialEnd = timeTrialEnd - timeTargetOn

            df=df.append({'trialNum': trialNum,
                          'timeTargetOn':timeTargetOn*encoder_dt,
                          'isSuccessful':isSuccessful,
                          'timeTargetAcquire':timeTargetAcquire*encoder_dt,
                          'timeTargetHeld':timeTargetHeld*encoder_dt,
                          'timeFirstTargetAcquire': timeTargetAcquire[0]*encoder_dt if len(timeTargetAcquire) else None,
                          'timeLastTargetAcquire': timeTargetAcquire[-1]*encoder_dt if len(timeTargetAcquire) else None,
                          'trialLength': timeTargetHeld*encoder_dt,

                          'act':act,
                          'target': tuple(target),
                          'binState': binState,
                          'binSpike': binSpike,
                          'binCursorPos':binCursorPos,
                          'binHandPos':binHandPos,
                          'binCursorVel':binCursorVel,
                          'binHandVel':binHandVel,
                          #'binHandAcc':binHandAcc,


                          'binCursorPosDistance':binCursorPosDistance,
                         },ignore_index=True)
            trialNum +=1

        self.traj = traj

        self.extraBins=0
        self.numChannels=192
        self.successfulTrials = np.where(df['isSuccessful']==1)[0]
        self.numBins = np.min([df['binSpike'][i].shape[0] for i in range(len(df))])
        self.numSuccessfulTrial =  numSuccessfulTrial = len(self.successfulTrials)
        self.trainingTrials  = self.successfulTrials
        self.testingTrials = self.successfulTrials
        self.numTrain = numTrain = len(self.trainingTrials)
        self.numTest =  numTest = len(self.testingTrials)


        assert len(self.trainingTrials)==self.numTrain
        assert len(self.testingTrials)==self.numTest

        self.centerOutTrials = np.where((np.stack(df['target'])!=[0,0]).any(axis=1))[0]
        self.centerInTrials  = df[~df.index.isin(self.centerOutTrials)].index


        self.df = df
        self.dt = encoder_dt
        #self._binTrial()
        self._check()
        self.numTrial = len(self.df)
        self.uniqueTarget = self._UniqueTarget()
        self.TrialsOfTargets = self._TrialsOfTargets()
        self.ColorOfTarget = self._ColorOfTarget()

        self._CalculateDistanceToTarget()
        self._CalculateDeviation()
        self._CalculateDistanceRatio()
        self.PSTH = spikesToPSTH(self,self.df['binSpike'])
        self.neuralDynamics = self._NeuralDynamics()

        self.meanDtT = self._getMeanDtT()

        self.downSample(decoder_dt//encoder_dt)

    def _check(self):
        '''
        check the dataset structure in a very high level to avoid assert in later operation
        '''
        df = self.df
        dt = self.dt
        for trialIdx in range(len(df)):
            binHandPos=df['binHandPos'][trialIdx]
            binHandVel=df['binHandVel'][trialIdx]
            binCursorPos=df['binCursorPos'][trialIdx]
            binCursorVel=df['binCursorVel'][trialIdx]
            ## check the length of sequence are the same.
            assert df['binCursorPos'][trialIdx].shape[0] == df['binCursorVel'][trialIdx].shape[0]
            assert df['binHandPos'][trialIdx].shape[0] == df['binHandVel'][trialIdx].shape[0]
            ## binHandVel is not directly calculated from binHandPos in Licorice experiment. There is a sampling time difference.
            #assert ((binHandPos[1:]-binHandPos[:-1])/dt*1000 == binHandVel[1:]).all()
            #assert ((binCursorPos[1:]-binCursorPos[:-1])/dt*1000 == binCursorVel[1:]).all()

        print("Dataset pass the check")



    def _generateTraj(self):
        model = self.model
        env = self.env
        deterministic = self.deterministic
        verbose = self.verbose

        obs = env.reset()
        ## traj
        traj = {'act':[],
                #'go_cue':[],
                'ob':[],
                'task_state':[],
                'target_position':[],
                'cursor_position':[],
                'hand_position':[],
                'cursor_velocity':[],
                'hand_velocity':[],
                'spike_train':[],
                #'hand_acc':[],

}
        # state and mask for recurrent policies
        state, mask = None, None
        #traj['task_state'].append(env.get_attr('task')[0].task_state)
        #traj['target_position'].append(env.get_attr('task')[0].pos_cursor)
        #traj['cursor_position'].append(env.get_attr('task')[0].pos_target)
        #traj['hand_position'].append(env.get_attr('encoder')[0].Q_hand_inputPos[-1])

        traj['task_state'].append(env.get_attr('task')[0].task_state)
        traj['target_position'].append(env.get_attr('Q_targetPos')[0][-1])
        traj['cursor_position'].append(env.get_attr('Q_cursorPos')[0][-1])
        traj['hand_position'].append(env.get_attr('Q_handPos')[0][-1])
        traj['cursor_velocity'].append(env.get_attr('Q_cursorVel')[0][-1])
        traj['hand_velocity'].append(env.get_attr('Q_handVel')[0][-1])
        traj['spike_train'].append(env.get_attr('Q_binSpike')[0][-1])
        #traj['go_cue'].append(env.get_attr('task')[0].go_cue)
        #traj['hand_acc'].append(env.get_attr('Q_handAcc')[0][-1])

        while(1):
            #print(obs)
            if isinstance(model, BaseRLModel):
                action, state = model.predict(obs, state=state, mask=mask, deterministic=deterministic)
            else:
                action = model.predict(obs)

            traj['act'].append(action)
            traj['ob'].append(obs)

            obs, reward, done, info = env.step(action)
            #traj['task_state'].append(env.get_attr('task')[0].task_state)
            #traj['target_position'].append(env.get_attr('task')[0].pos_target)
            #traj['cursor_position'].append(env.get_attr('task')[0].pos_cursor)
            #traj['hand_position'].append(env.get_attr('encoder')[0].Q_hand_inputPos[-1])

            traj['task_state'].append(env.get_attr('task')[0].task_state)
            traj['target_position'].append(env.get_attr('Q_targetPos')[0][-1])
            traj['cursor_position'].append(env.get_attr('Q_cursorPos')[0][-1])
            traj['hand_position'].append(env.get_attr('Q_handPos')[0][-1])
            traj['cursor_velocity'].append(env.get_attr('Q_cursorVel')[0][-1])
            traj['hand_velocity'].append(env.get_attr('Q_handVel')[0][-1])
            traj['spike_train'].append(env.get_attr('Q_binSpike')[0][-1])
            #traj['go_cue'].append(env.get_attr('task')[0].go_cue)
            #traj['hand_acc'].append(env.get_attr('Q_handAcc')[0][-1])





            #print(isinstance(done,np.ndarray))
            if (isinstance(done,np.ndarray) and done[0]) or (not isinstance(done,np.ndarray) and done):
                break

        env.close()
        traj['dt']  = env.get_attr('dt')[0]
        #traj['targets'] = list(map(tuple, env.get_attr('target_list')[0]))
        traj['act'] = np.array(traj['act'])
        traj['ob']  = np.array(traj['ob'])
        traj['target_position'] = np.array(traj['target_position'])
        traj['cursor_position'] = np.array(traj['cursor_position'])
        traj['hand_position'] = np.array(traj['hand_position'])
        traj['cursor_velocity'] = np.array(traj['cursor_velocity'])
        traj['hand_velocity'] = np.array(traj['hand_velocity'])
        traj['spike_train'] = np.array(traj['spike_train'])
        #traj['hand_acc'] = np.array(traj['hand_acc'])


        return traj

    def _binTrial(self):
        '''
        bin spikes, cursor position and hand position, and calculate binned hand velocity and cursor velocity
        '''
        if(self.verbose>0):
            print('Start binning spikes')

        dt = self.dt
        df = self.df

        df['binSpike'] = ""
        df['binHandVel'] = ""
        df['binHandAcc'] = ""
        df['binCursorVel'] = ""
        df['binCursorAcc'] = ""

        for trialIdx in reversed(range(1,len(df))):
            trialNum = df['trialNum'][trialIdx]
            preTrialNum = df['trialNum'][trialIdx-1]
            if (trialNum-1==preTrialNum):
                df.at[trialIdx,'binCursorVel'] = np.concatenate([(df.loc[trialIdx,'binCursorPos'][0:1] - df.loc[trialIdx-1,'binCursorPos'][-1:]),(df.loc[trialIdx,'binCursorPos'][1:]-df.loc[trialIdx,'binCursorPos'][:-1])],axis=0)/dt*1000
                df.at[trialIdx,'binHandVel'] = np.concatenate([(df.loc[trialIdx,'binHandPos'][0:1] - df.loc[trialIdx-1,'binHandPos'][-1:]),(df.loc[trialIdx,'binHandPos'][1:]-df.loc[trialIdx,'binHandPos'][:-1])],axis=0)/dt*1000

        for trialIdx in reversed(range(2,len(df))):
            trialNum = df['trialNum'][trialIdx]
            preTrialNum = df['trialNum'][trialIdx-1]
            if (trialNum-1==preTrialNum):
                df.at[trialIdx,'binCursorAcc'] = np.concatenate([(df.loc[trialIdx,'binCursorVel'][0:1] - df.loc[trialIdx-1,'binCursorVel'][-1:]),(df.loc[trialIdx,'binCursorVel'][1:]-df.loc[trialIdx,'binCursorVel'][:-1])],axis=0)/dt*1000
                df.at[trialIdx,'binHandAcc'] = np.concatenate([(df.loc[trialIdx,'binHandVel'][0:1] - df.loc[trialIdx-1,'binHandVel'][-1:]),(df.loc[trialIdx,'binHandVel'][1:]-df.loc[trialIdx,'binHandVel'][:-1])],axis=0)/dt*1000

        self.df = df.drop(0)
        self.df = self.df.reset_index(drop=True)
