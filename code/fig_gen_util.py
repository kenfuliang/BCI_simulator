import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu,ttest_ind
import pandas as pd
import scipy
import sys
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import copy
import os
from matplotlib.ticker import PercentFormatter
#from Dataset import get_color_def
#from Dataset import PlotDistanceToTarget
#sys.path.append("../../rig_BCI/code/")
#from LicoriceDataset import LicoriceDataset, loadFiles, PlotTrialTimeBar,PlotDistanceToTarget


def get_color_def():
    color = dict({
        'hand':'green',
        'ReFIT':'goldenrod',
        'FIT':'brown',
        'VKF':'darkblue',
        'NDF':'slategrey',
        'WF':'darkviolet',
        'FORCE':'pink',
        'PVKF':'mediumpurple',
        })
    return color



def adjust_spines(ax,outward=10):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_position(('outward',outward))
    ax.spines['bottom'].set_position(('outward',outward))
    ax.spines['left'].set_bounds(ax.get_yticks()[0], ax.get_yticks()[-1])
    ax.spines['bottom'].set_bounds(ax.get_xticks()[0], ax.get_xticks()[-1])

def remove_all_spines(ax,outward=10):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])

def MergeData(df,decoders):
    rt_df = pd.DataFrame()
    for decoder in decoders:
        index_list = df[(df['decoder']==decoder)].index
        data = copy.deepcopy(df['data'][index_list[0]])
        for ii in index_list[1:]:
            data.merge(df['data'][ii])
        rt_df = rt_df.append({'decoder':decoder,'data':data},ignore_index=True)
    return rt_df

def PlotAllForPaper(df,b_save=False,path='./',postfix='',decoders=['hand','FIT','ReFIT','FORCE','VKF']):
    color = get_color_def()
    sorting_dict = {'hand':0, 'FIT':1, 'ReFIT':2,'FORCE':3, 'VKF':4, 'PVKF':5}
    df = df[df['decoder'].isin(decoders)]
    #decoders = df['decoder'].unique()
    #df = MergeData(df,decoders)
    df_stats = pd.DataFrame(columns=['decoder','trial_time','first_touch_time','dial-in_time','distance_ratio','max_deviation','success_rate'])
    for index,row in df.iterrows():
        df_stats = df_stats.append({**row['data'].Statistics().drop(columns=['target']).iloc[2],'decoder':row['decoder']},ignore_index=True)

    df = df.iloc[(df['decoder'].map(sorting_dict).argsort())].reset_index(drop=True)
    df_stats = df_stats.iloc[(df_stats['decoder'].map(sorting_dict).argsort())].reset_index(drop=True)

    display(df_stats)

    fig = plt.figure(figsize=(18,3))
    #gs = fig.add_gridspec(1, 6,wspace=1.5)
    gs = fig.add_gridspec(1, 6)
    ax0 = fig.add_subplot(gs[0:2])
    ax1 = fig.add_subplot(gs[2])
    ax2 = fig.add_subplot(gs[3])
    ax3 = fig.add_subplot(gs[4:6])
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)

    
    '''
    Distance-to-target profile
    '''
    plt.sca(ax0)
    #fig = plt.figure(figsize=(8.5,3))

    decoder_list = []
    color_list = []
    DiT_list = []
    for index, row in df.iterrows():
        data = row['data']
        decoder = row['decoder']
        decoder_list.append(row['decoder'])
        color_list.append(color[decoder])
        DiT_list.append(df_stats['dial-in_time'][index])
        data.PlotDistanceToTarget(color=color[decoder],label=decoder)
    axis = plt.gca()
    axis.get_legend().remove()
    axis.set_xlim([0,1500])
    axis.set_ylim([0,8])
    axis.set_xticks([0,500,1000,1500])
    axis.set_yticks(range(0,8+1,4))
    adjust_spines(axis,outward=5)

    #axins = inset_axes(axis, width=0.4*len(df), height=1)
    #axins.spines['top'].set_visible(False)
    #axins.spines['right'].set_visible(False)
    #axins.set_ylabel("DiT (s)")
    #plt.axes(axins)
    #plt.bar(decoder_list,np.array(DiT_list)/1000,color=color_list,align='center',alpha=0.5, ecolor='black')#yerr=df_DTT['sem(DiT)'],
    #axins.set_ylim([0,1])
    #adjust_spines(axins,outward=5)
    #axins.set_xticks([])

    #if b_save:
    #    fig.savefig(os.path.join(path,'DTT'+postfix)+'.pdf')

    '''
    Dail-in time bars (DIT)
    '''
    plt.sca(ax1)
    #fig,axis = plt.subplots(figsize=(0.7*len(df),1))
    plt.bar(decoder_list,np.array(DiT_list)/1000,color=color_list,align='center',alpha=0.5, ecolor='black')

    ax1.set_ylabel("Dail-in time [s]")
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.set_ylim([0,1])
    ax1.set_yticks([0,1])
    adjust_spines(ax1,outward=5)
    ax1.set_xticks([])


    '''
    First-touch time bars (TFA)
    '''
    plt.sca(ax2)
    #fig,axis = plt.subplots(figsize=(0.7*len(df),1))
    
    TFA_list = []
    for index, row in df.iterrows():
        data = row['data']
        decoder = row['decoder']
        TFA_list.append(df_stats['first_touch_time'][index])
    plt.bar(decoder_list,np.array(TFA_list)/1000,color=color_list,align='center',alpha=0.5, ecolor='black')

    ax2.set_ylabel("First-touch time [s]")
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.set_ylim([0,1])
    ax2.set_yticks([0,1])
    adjust_spines(ax2,outward=5)
    ax2.set_xticks([])

    '''
    TrialTimeHist
    '''
    plt.sca(ax3)
    for index,row in df.iterrows():
        decoder = row['decoder']
        data = row['data']
        data.PlotTrialTimeHist(color=color[decoder],ylim=0.5,alpha=0.3)
    ax3.set_xlabel("Trial times [s]")
    ax3.set_ylabel("% of trials")
    ax3.set_yticks([0,0.5])
    adjust_spines(ax3,outward=5)
    ax3.set_ylim([0,0.7])


    if b_save:
        fig.savefig(os.path.join(path,'DTTandBars'+postfix)+'.pdf',bbox_inches='tight',transparent=True)

    '''
    Trajectories 
    '''
    fig = plt.figure(figsize=(5*3,3))
    lim = max([max(np.linalg.norm(data.uniqueTarget,axis=-1)) for data in df['data']])*1.5
    gs = fig.add_gridspec(1,5)
    for index,row in df.iterrows():
        decoder = row['decoder']
        data = row['data']
        axes = fig.add_subplot(gs[0,index])
        plt.sca(axes)
        data.PlotCursorMovement(mode='single',acceptance_window=data.metadata['acceptance_window'],lim=lim)

    if b_save:
        fig.savefig(os.path.join(path,'Traj'+postfix)+'.pdf',bbox_inches='tight',transparent=True)


    fig = plt.figure(figsize=(2*3,3))
    gs = fig.add_gridspec(1,2)
    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1])
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)

    '''
    Max Deviation bars 
    '''
    ax=ax0
    plt.sca(ax)
    #fig,axis = plt.subplots(figsize=(0.7*len(df),1))
    MD_list = []
    for index, row in df.iterrows():
        data = row['data']
        decoder = row['decoder']
        MD_list.append(df_stats['max_deviation'][index])
    plt.bar(decoder_list,MD_list,color=color_list,align='center',alpha=0.5, ecolor='black')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylabel("Max deviation [mm]")
    ax.set_ylim([0,40])
    ax.set_yticks([0,10,20,30,40])
    adjust_spines(ax,outward=5)
    ax.set_xticks([])

    '''
    Distance ratio bars 
    '''
    ax=ax1
    plt.sca(ax)

    DR_list = []
    for index, row in df.iterrows():
        data = row['data']
        decoder = row['decoder']
        DR_list.append(df_stats['distance_ratio'][index])
    plt.bar(decoder_list,DR_list,color=color_list,align='center',alpha=0.5, ecolor='black')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylabel("Distance ratio")
    ax.set_ylim([0,3])
    ax.set_yticks([0,1,2,3])
    adjust_spines(ax,outward=5)
    ax.set_xticks([])

    if b_save:
        fig.savefig(os.path.join(path,'TrajBars'+postfix)+'.pdf',bbox_inches='tight',transparent=True)


       

        
    #'''
    #Cursor Velocity
    #'''
    #fig = plt.figure(figsize=(3,3))
    #for index,row in df.iterrows():
    #    decoder = row['decoder']
    #    data = row['data']
    #    data.PlotCursorVelProfile(group='all',color=color[decoder])
    #plt.xticks(range(0,1000,200))
    #axis = plt.gca()
    #axis.spines['top'].set_visible(False)
    #axis.spines['right'].set_visible(False)

    #if b_save:
    #    fig.savefig(os.path.join(path,'CursorVel'+postfix)+'.pdf')
    #'''
    #Hand Velocity
    #'''
    #fig = plt.figure(figsize=(3,3))
    #for index,row in df.iterrows():
    #    decoder = row['decoder']
    #    data = row['data']
    #    data.PlotHandVelProfile(group='all',color=color[decoder])
    #plt.xticks(range(0,1000,200))
    #axis = plt.gca()
    #axis.spines['top'].set_visible(False)
    #axis.spines['right'].set_visible(False)

    #if b_save:
    #    fig.savefig(os.path.join(path,'HandVel'+postfix)+'.pdf')

    
    #'''
    #FirstTouchTimeHist
    #'''
    #fig,axes = plt.subplots(1,5,figsize=(15,3))
    #fig.subplots_adjust(left=0.05,right=0.975)
    #fig.suptitle("First-touch Time")
    #for index,row in df.iterrows():
    #    decoder = row['decoder']
    #    data = row['data']
    #    plt.sca(axes[index])
    #    data.PlotTFAHist(color=color[decoder],ylim=0.5)
    #    adjust_spines(axes[index])
    #    if index!=0:
    #        axes[index].spines['left'].set_visible(False)
    #        axes[index].set_yticks([])
    #axes[0].set_ylim([0,0.6])
    #axes[0].set_yticks([0,0.5])


    #if b_save:
    #    fig.savefig(os.path.join(path,'TFAHist'+postfix)+'.pdf')

    #'''
    #Dial-in time Hist
    #'''

    #fig,axes = plt.subplots(1,5,figsize=(15,3))
    #fig.subplots_adjust(left=0.05,right=0.975)
    #fig.suptitle("Dial-in Time")
    #for index,row in df.iterrows():
    #    decoder = row['decoder']
    #    data = row['data']
    #    plt.sca(axes[index])
    #    data.PlotDITHist(color=color[decoder],ylim=1.0)
    #    adjust_spines(axes[index])
    #    if index!=0:
    #        axes[index].spines['left'].set_visible(False)
    #        axes[index].set_yticks([])

    #axes[0].set_ylim([0,1.0])
    #axes[0].set_yticks([0,1.0])

    #if b_save:
    #    fig.savefig(os.path.join(path,'DITHist'+postfix)+'.pdf')

    #'''
    #Max Deviation Hist
    #'''
    #import seaborn as sns
    #fig,axes = plt.subplots(1,5,figsize=(15,3),sharex=True)
    #fig.subplots_adjust(left=0.05,right=0.975)
    #fig.suptitle("Max Deviation")
    #for index,row in df.iterrows():
    #    decoder = row['decoder']
    #    data = row['data']
    #    plt.sca(axes[index])

    #    df_centerOut = data.df.iloc[data.centerOutTrials]
    #    X = list(df_centerOut['max_deviation'][df_centerOut['isSuccessful']==True])
    #    #sns.kdeplot(X,shade=True,bw=1,color=color[decoder])
    #    plt.hist(X,bins=np.arange(0,50,2), weights=np.ones(len(X)) / len(X),color=color[decoder],alpha=0.5)
    #    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
    #    axes[index].set_xlim([0,50])
    #    axes[index].set_ylim([0,0.5])
    #    adjust_spines(axes[index])
    #    if index!=0:
    #        axes[index].spines['left'].set_visible(False)
    #        axes[index].set_yticks([])

    #axes[0].set_ylim([0,0.5])
    #axes[0].set_yticks([0,0.5])

    #if b_save:
    #    fig.savefig(os.path.join(path,'MaxDeviationHist'+postfix)+'.pdf')

    #'''
    #Distance Ratio Hist
    #'''
    #import seaborn as sns
    #fig,axes = plt.subplots(1,5,figsize=(15,3),sharex=True)
    #fig.subplots_adjust(left=0.05,right=0.975)
    #fig.suptitle("Distance ratio")
    #for index,row in df.iterrows():
    #    decoder = row['decoder']
    #    data = row['data']
    #    plt.sca(axes[index])

    #    df_centerOut = data.df.iloc[data.centerOutTrials]
    #    X = list(df_centerOut['distance_ratio'][df_centerOut['isSuccessful']==True])
    #    #sns.kdeplot(X,shade=True,bw=0.1,color=color[decoder])
    #    #_ = plt.hist(X,20,density=True,color=color[decoder])
    #    plt.hist(X,bins=np.arange(0,6,0.2), weights=np.ones(len(X)) / len(X),color=color[decoder],alpha=0.5)
    #    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
    #    axes[index].set_xlim([0,6.0])
    #    axes[index].set_ylim([0,1.0])
    #    adjust_spines(axes[index])
    #    if index!=0:
    #        axes[index].spines['left'].set_visible(False)
    #        axes[index].set_yticks([])

    #axes[0].set_ylim([0,1.0])
    #axes[0].set_yticks([0,1.0])

    #if b_save:
    #    fig.savefig(os.path.join(path,'DistanceRatioHist'+postfix)+'.pdf')



def AveragedDTT(df_data):
    allD = []
    FTT_list = []
    LTT_list = []
    for ii in df_data.index:
        data = df_data['data'][ii]
        for trial in data.centerOutTrials:
            if data.df['isSuccessful'][trial]:
                D = data.df['binCursorPosDistance'][trial]
                if len(D)>200:
                    D = D[:200]
                else:
                    D = np.pad(D,(0,200-len(D)),mode='constant',constant_values=(np.nan,))

                allD.append(D)

        df_centerOut = data.df.iloc[data.centerOutTrials]
        FTT = df_centerOut['timeFirstTargetAcquire'][df_centerOut['isSuccessful']==True]
        LTT = df_centerOut['timeLastTargetAcquire'][df_centerOut['isSuccessful']==True]

        FTT_list += list(FTT)
        LTT_list += list(LTT)
    meanD = np.nanmean(allD,axis=0)
    meanD= meanD[~np.isnan(meanD)]
    return meanD,FTT_list,LTT_list

def CreateDFforPlot(df,decoders=['hand','FIT','ReFIT','VKF']):
    rt_df = pd.DataFrame(columns=['decoder','dt','FTT','LTT','DIT','MaxDeviation','DistanceRatio'])
    for decoder in decoders:
        rt = {}
        rt['FTT'] = []
        rt['DIT'] = []
        rt['LTT'] = []
        rt['MaxDeviation'] = []
        rt['DistanceRatio'] = []
        index_list = df[(df['decoder']==decoder)].index
        for ii in index_list:
            data = df['data'][ii]
            df_centerOut = data.df.iloc[data.centerOutTrials]
            rt['FTT']+=list(df_centerOut['timeFirstTargetAcquire'][df_centerOut['isSuccessful']==True])
            rt['LTT']+=list(df_centerOut['timeLastTargetAcquire'][df_centerOut['isSuccessful']==True])
            rt['MaxDeviation']+=list(df_centerOut['max_deviation'][df_centerOut['isSuccessful']==True])
            rt['DistanceRatio']+=list(df_centerOut['distance_ratio'][df_centerOut['isSuccessful']==True])

        rt_df = rt_df.append({
            'decoder':decoder,
            'dt':df[(df['decoder']==decoder)]['data'].iloc[0].dt,
            'FTT':rt['FTT'],
            'DIT':np.array(rt['LTT'])-np.array(rt['FTT']),
            'LTT':rt['LTT'],
            'MaxDeviation':rt['MaxDeviation'],
            'DistanceRatio':rt['DistanceRatio'],
            },ignore_index=True)

# for metric in ['LTT','FTT','DIT','MaxDeviation','DistanceRatio']:
#     df_Plot["mean({})".format(metric)] = df_Plot[metric].apply(lambda x:np.mean(x))
    return rt_df




def CreateDTTDF(df,decoders=['hand','FIT','ReFIT','VKF']):
    '''
    Form averaged Distance-to-Target dataframe
    '''
    df_DTT = pd.DataFrame(columns=['decoder','dt','meanD','FTT','LTT'])
    for decoder in decoders:
        meanD, FTT_list, LTT_list = AveragedDTT(df[(df['decoder']==decoder)])
        df_DTT = df_DTT.append({
            'decoder':decoder,
            'dt':df[(df['decoder']==decoder)]['data'].iloc[0].dt,
            'meanD':meanD,
            'FTT':FTT_list,
            'LTT':LTT_list,
            'DiT':np.array(LTT_list)-np.array(FTT_list)
        },ignore_index=True)

    df_DTT['trials'] = df_DTT['FTT'].str.len()
    df_DTT['mean(FTT)'] = df_DTT['FTT'].apply(np.mean,axis=0)
    df_DTT['mean(LTT)'] = df_DTT['LTT'].apply(np.mean,axis=0)
    df_DTT['mean(DiT)'] = df_DTT['DiT'].apply(np.mean,axis=0)
    df_DTT['std(FTT)']  = df_DTT['FTT'].apply(np.std,axis=0)
    df_DTT['std(LTT)']  = df_DTT['LTT'].apply(np.std,axis=0)
    df_DTT['std(DiT)']  = df_DTT['DiT'].apply(np.std,axis=0)
    df_DTT['sem(FTT)']  = df_DTT['FTT'].apply(scipy.stats.sem,axis=0)
    df_DTT['sem(LTT)']  = df_DTT['LTT'].apply(scipy.stats.sem,axis=0)
    df_DTT['sem(DiT)']  = df_DTT['DiT'].apply(scipy.stats.sem,axis=0)
    #display(df_DTT.drop(['meanD','FTT','LTT','DiT'], axis=1))
    return df_DTT

def PlotAllDistanceToTarget(df,decoders=['hand','FIT','ReFIT','VKF']):
    for decoder in decoders:
        index = df[df['decoder']==decoder].index
        assert len(index)==1
        FTT_list = df['FTT'][index[0]]
        LTT_list = df['LTT'][index[0]]
        meanD = df['meanD'][index[0]]
        FTT = np.mean(FTT_list)
        LTT = np.mean(LTT_list)
        dt = df['dt'][index[0]]
        #PlotDistanceToTarget(meanD, FTT, LTT,dt,color[decoder],decoder)

        if ~np.isnan(FTT) and ~np.isnan(LTT):
            f=scipy.interpolate.interp1d(np.arange(0,len(meanD))*dt,meanD,kind='cubic')
            plt.plot(np.arange(FTT,LTT,0.1),f(np.arange(FTT,LTT,0.1)),color=color[decoder],linewidth=10,alpha=0.5)
            handle, = plt.plot(np.arange(0,FTT,0.1),f(np.arange(0,FTT,0.1)),color=color[decoder],linewidth=2,label=decoder,alpha=0.5)

    axes = plt.gca()
    axes.spines['top'].set_visible(False)
    axes.spines['right'].set_visible(False)
    axes.set_xlabel("Time (ms)")
    axes.set_ylabel("Distance to target (mm)")
    axes.set_xticks(range(0,1751,250))
    axes.set_yticks(range(0,120,50))
    axes.set_ylim(0,120)
    axes.set_xlim(0,1600)



def PlotMaxDeviation(df,decoders = ['hand','FIT','VKF'],source='cursor'):
    y = []
    legend_list = []
    color_list = []
    for decoder in decoders:
        color_list.append(color[decoder])
        legend_list.append(decoder)
        X = []
        index_list = df[(df['decoder']==decoder)].index
        for ii in index_list:
            data = df['data'][ii]
            df_centerOut = data.df.iloc[data.centerOutTrials]
            if source=='cursor':
                X = X+ list(df_centerOut['max_deviation'][df_centerOut['isSuccessful']==True])
            elif source=='hand':
                X = X+ list(df_centerOut['max_deviation(hand)'][df_centerOut['isSuccessful']==True])
        y.append(X)
    trials = [len(y[ii]) for ii in range(len(y))]
    mean = [np.nanmean(y[ii]) for ii in range(len(y))]
    std = [np.nanstd(y[ii]) for ii in range(len(y))]
    sem = [scipy.stats.sem(y[ii],nan_policy='omit') for ii in range(len(y))]
    plt.bar(legend_list,mean,yerr = sem,color=color_list,align='center',alpha=0.5, ecolor='black', capsize=10)
    print("Max deviation (mm):\n number of successful center-out trials: {} \n mean:{}\n sem:{}\n std:{}\n".format(trials,np.around(mean,2),np.around(sem,2),np.around(std,2)))


    ## show mannwhiteneyu p-value
    for ii in range(1,len(decoders)):
        print("{} v.s. {}: p-value={}".format(decoders[ii-1],decoders[ii], mannwhitneyu(y[ii-1],y[ii])[1]))

    axes = plt.gca()
    axes.spines['top'].set_visible(False)
    axes.spines['right'].set_visible(False)
    axes.set_ylabel('Max Deviation (mm)')
    axes.set_yticks([0,10,20,30,40])


def PlotDistanceRatio(df,decoders=['hand','FIT','VKF'],source='cursor',bPlot=True):

    y = []
    legend_list = []
    color_list = []
    for decoder in decoders:
        color_list.append(color[decoder])
        legend_list.append(decoder)
        X = []
        index_list = df[(df['decoder']==decoder)].index
        for ii in index_list:
            data = df['data'][ii]
            dt = data.dt
            for trial in data.centerOutTrials:#range(len(df['data'][ii].df)):#
                if data.df['isSuccessful'][trial]:
                    if source=='cursor':
                        X.append(data.df['distance_ratio'][trial])
                    elif source=='hand':
                        X.append(data.df['distance_ratio(hand)'][trial])
        y.append(X)

    trials = [len(y[ii]) for ii in range(len(y))]
    mean = [np.nanmean(y[ii]) for ii in range(len(y))]
    std = [np.nanstd(y[ii]) for ii in range(len(y))]
    sem = [scipy.stats.sem(y[ii],nan_policy='omit') for ii in range(len(y))]

    print("Distance ratio (a.u.):\n number of successful center-out trials: {} \n mean:{}\n sem:{}\n std:{}\n".format(trials,np.around(mean,2),np.around(sem,2),np.around(std,2)))
    ## show mannwhiteneyu p-value
    for ii in range(1,len(decoders)):
        print("{} v.s. {}: p-value={}".format(decoders[ii-1],decoders[ii], mannwhitneyu(y[ii-1],y[ii])[1]))


    if bPlot:
        plt.bar(legend_list,mean,yerr = sem,color=color_list,align='center',alpha=0.5, ecolor='black', capsize=10)
        axes = plt.gca()
        axes.spines['top'].set_visible(False)
        axes.spines['right'].set_visible(False)
        axes.set_ylabel('Distance Ratio (a.u.)')
        axes.set_yticks([0,1,2,3,4])

def PlotSuccessRate(df,decoders=['hand','FIT','VKF']):
    y = []
    legend_list = []
    color_list = []
    for decoder in decoders:
        color_list.append(color[decoder])
        legend_list.append(decoder)
        X = []

        index_list = df[(df['decoder']==decoder)].index
        for ii in index_list:
            data = df['data'][ii]
            X+=list(data.df['isSuccessful'][data.centerOutTrials])
        y.append(X)

    trials = [len(y[ii]) for ii in range(len(y))]
    mean = [np.mean(y[ii]) for ii in range(len(y))]
    std = [np.std(y[ii]) for ii in range(len(y))]
    sem = [scipy.stats.sem(y[ii]) for ii in range(len(y))]
    plt.bar(legend_list,mean,yerr = sem,color=color_list,align='center',alpha=0.5, ecolor='black', capsize=10)

    print("Success rate:\n number of center-out trials: {} \n mean:{}\n sem:{}\n std:{}\n".format(trials,np.around(mean,3),np.around(sem,3),np.around(std,3)))
    axes = plt.gca()
    axes.spines['top'].set_visible(False)
    axes.spines['right'].set_visible(False)
    axes.set_ylabel('Success Rate (a.u.)')
    axes.set_yticks(np.arange(0.5,1.1,0.1))
    axes.set_ylim([0.5,1.0])


def PlotCursorSpeed(df,decoders=['hand','FIT','VKF']):
    for decoder in decoders:
        allV = []
        index_list = df[(df['decoder']==decoder)].index
        for ii in index_list:
            label = df['decoder'][ii]
            data = df['data'][ii]
            for trialIdx in data.centerOutTrials:
                if data.df['isSuccessful'][trialIdx]:
                    v = data.df['binCursorVel'][trialIdx][:,0:2]
                    speed = np.sqrt(np.sum(v**2,axis=1))
                    if len(speed)>200:
                        speed = speed[:200]
                    else:
                        speed = np.pad(speed,(0,200-len(speed)),mode='constant',constant_values=(np.nan,))
                    allV.append(speed)
        avgV = np.nanmean(allV,axis=0)
        minBins = 1000//data.dt
        plt.plot(range(0,minBins*data.dt,data.dt),avgV[:minBins],'.-',linewidth=2,label=label,color=color[label],alpha=0.5)
        plt.legend()
    axes = plt.gca()
    axes.set_yticks([0,100,200,300,400,500])
    axes.set_xlabel("Time (ms)")
    axes.set_ylabel('Speed (mm/s)')
    axes.spines['top'].set_visible(False)
    axes.spines['right'].set_visible(False)
    axes.set_xlim([0,1000])
