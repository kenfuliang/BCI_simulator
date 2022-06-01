import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import scipy.signal as signal
from scipy.ndimage import gaussian_filter
from sklearn.decomposition import PCA
#from encoder.PD import PDEncoder
#from encoder.PDh import PDEncoder_h
#from encoder.PPVT import PPVTEncoder
#from encoder.PPOLE import PPOLEEncoder
#from encoder.PPWF import PPWFEncoder
#from encoder.PoissonGLM import PoissonGLM
#from encoder.BernoulliGLM import BernoulliGLM
#from encoder.MLP import MLP
#from encoder.RNN import RNN
#from encoder.AttentiveRNN import AttentiveRNN
#from encoder.CNN import CNN
#from decoder.OLE import OLEDecoder
#from decoder.wiener import WFDecoder
#from decoder.kalman import KFDecoder
#from decoder.FORCE import FORCEDecoder
from keras.engine.training import Model
from metrics import *
from mpl_toolkits.mplot3d import Axes3D
from stgen import StGen
import scipy.io as sio

def PlotSpikeRaster(axes
                   ,data
                   ,targetIdx
                   ,neuronIdx):

    numTrial = data.numTrial
    df = data.df

    spike_list=[]
    target_trials = [ i for i in range(numTrial) if (targetIdx==df['target'][i]).all()]
    for trial in target_trials:
        spikeRaster = df['spikeRaster'][trial].T[neuronIdx]
        index_a,index_b=spikeRaster.nonzero() ##Q: this could be improved
        spike_list.append(index_b)

    ##plot
    for ith, trial in enumerate(spike_list):
        axes.vlines(trial, ith + .5, ith + 1.5)
        axes.set_ylim(.5, len(spike_list) + .5)
    axes.set_title('Raster plot (neuron {})'.format(neuronIdx+1))
    axes.set_xlabel('time')
    axes.set_ylabel('trial')

def PlotPSTH(axes,PSTH,dt,color='b'):
    x = np.arange(0,len(PSTH)*dt,dt)
    axes.plot(x,PSTH,color=color)


def binning(X_in,dt,binType=None):
    ###############################################################
    # Need to be optimized to have better computation speed
    ###############################################################
    numSamples, dims = X_in.shape
    numBins = int(np.floor((numSamples)/dt))
    binX = np.zeros(dims*numBins).reshape(numBins,dims)
    for i in range(int(numBins)):
        binStart = (i)*dt
        binStop = (i+1)*dt
        if binType == 'sum':
            binX[i,:] = np.sum(X_in[binStart:binStop,:],axis=0).reshape(dims)
        elif binType == 'mean':
            binX[i,:] = np.nanmean(X_in[binStart:binStop,:],2)
        elif binType == 'first' or binType =='all':
            binX[i,:] = X_in[binStart,:]
        else:
            binX[i,:] = X_in[binStop-1,:]
    return binX

def pol2cart(rho, phi):
    rt_x = rho * np.cos(phi)
    rt_y = rho * np.sin(phi)
    return(rt_x, rt_y)

def encoderPSTH(data
               ,model
               ,input_shape
               ,numBins=40
               ):
    PSTH = []
    for targetIdx in data.uniqueTarget:
        binningSpikes = []
        trials = 0
        for trialIdx in range(data.numTrial):
            if(targetIdx[0] == data.df['target'][trialIdx][0] and targetIdx[1] == data.df['target'][trialIdx][1]):
                ## prepare Input
                p = data.df['binHandPos'][trialIdx][1:,0:2]
                v = data.df['binHandVel'][trialIdx][:,0:2]
                t = data.df['target'][trialIdx][0:2].reshape(1,2)
                t = np.repeat(t,v.shape[0],axis=0)
                if(input_shape=='p,v,t'):
                    kinematics = np.hstack((p,v,t))
                elif(input_shape=='p,v'):
                    kinematics = np.hstack((p,v))
                elif(input_shape=='t,p,v'):
                    kinematics = np.hstack((t,p,v))
                elif(input_shape=='v'):
                    kinematics = v
                ## gen spikes
                neural_output = model.predict(kinematics)[:,0:96]
                if(len(binningSpikes)==0):
                    binningSpikes=neural_output[0:numBins]
                else:
                    binningSpikes+=neural_output[0:numBins]
                trials+=1
        PSTH.append(binningSpikes)
    return PSTH

#def EncodeDecodeTest(data,encoder,decoder=None,verbose=False,name=""):
#    '''
#    20210612: PSTH calculation is from "spike" instead of "spikeRate". Therefore, PCC of PSTH is lower than shown in TBME paper.
#    '''
#    rt={}
#    rt['testName']= name
#    rt['decoder'] = decoder
#    rt['encoder'] = encoder
#
#    if( (data.df['target'][0]==(0,0)) ):
#        plottingTrials = range(1,15,2)
#    else:
#        plottingTrials = range(0,14,2)
#
#    #plottingTrials = [2,4,6,8,10,12,14,16,18,20,22]
#
#    dt = data.dt
#    decodePos = []
#    decodeVel = []
#    decodePVs = None
#    pos = None
#    encodeSpikes=[]
##    encodeSpikesRate=[]
#    for trial in range(data.numTrial):
#        ## encoder
#
#        if(encoder=='real'):
#            spikeRate = data.df['binSpike'][trial][data.extraBins:]
#        elif (isinstance(encoder,PDEncoder)):
#            spikeRate = encoder.encodeFromData(data,trial)
#        elif (isinstance(encoder,PDEncoder_h)):
#            spikeRate = encoder.encodeFromData(data,trial)
#        elif (isinstance(encoder,PPVTEncoder)):
#            spikeRate = encoder.encodeFromData(data,trial)
#        elif (isinstance(encoder,PPOLEEncoder)):
#            spikeRate = encoder.encodeFromData(data,trial)
#        elif (isinstance(encoder,PPWFEncoder)):
#            spikeRate = encoder.encodeFromData(data,trial)
#        elif (isinstance(encoder,PoissonGLM)):
#            spikeRate = encoder.encodeFromData(data,trial)
#        elif (isinstance(encoder,BernoulliGLM)):
#            spikeRate = encoder.encodeFromData(data,trial)
#        elif (isinstance(encoder,MLP)):
#            #spikeRate = encoder.encode(data)#[0:data.numBins]
#            spikeRate = encoder.encodeFromData(data,trial)[data.extraBins:]#[0:data.numBins]
#        elif (isinstance(encoder,RNN)):
#            #spikeRate = encoder.encode(data,trial)#[0:data.numBins]
#            spikeRate = encoder.encodeFromData(data,trial)[data.extraBins:]#[0:data.numBins]
#        elif (isinstance(encoder,AttentiveRNN)):
#            spikeRate = encoder.encode(data,trial)#[0:data.numBins]
#        elif (isinstance(encoder,CNN)):
#            spikeRate = encoder.encode(data,trial)#[0:data.numBins]
#        else:
#            raise ValueError("Not support this encoder")
#
#        if(encoder=='real'):
#            spike = spikeRate
#        else:
#            spike = np.random.poisson(spikeRate)
#
#        ## decoder
#        if(isinstance(decoder,OLEDecoder)):
#            decodePVs = decoder.decodeFromData(data,spike,trial)
#            _P = decodePVs[:,0:2]
#            _V = decodePVs[:,2:4]
#            #bUseVelocity=1
#            alpha = 1
#        elif (isinstance(decoder,WFDecoder)):
#            decodePVs = decoder.decodeFromData(data,spike,trial)
#            _P = decodePVs[:,0:2]
#            _V = decodePVs[:,2:4]
#            #bUseVelocity=1
#            alpha = 1
#        elif (isinstance(decoder,FORCEDecoder)):
#            decodePVs = decoder.decodeFromData(data,spike[:,0:decoder.I],trial)
#            _P = decodePVs[:,0:2]
#            _V = decodePVs[:,2:4]
#            alpha = 0.95
#        elif (isinstance(decoder,KFDecoder)):
#            decodeState = decoder.decodeFromData(data,spike,trial)
#            _P = decodeState[:,0:2]
#            _V = decodeState[:,2:4]*1000
#            if (decoder.type=='PVKF' or decoder.type=='NDF'):
#                alpha = 0.975
#            else:
#                alpha = 1.0
#            ### Use velocity to recalculate the positons##
#            #if(decoder.type=='PVKF' or decoder.type=='NDF'):
#            #    pos      = decodeState[:,0:2]
#            #    bUseVelocity=0
#            #else:
#            #    decodeVs = decodeState[1:,2:4]*1000
#            #    bUseVelocity=1
#        else:
#            #decodeState = decoder.decodeFromData(data,spike,trial)
#            #_P = decodeState[:,0:2]
#            #_V = decodeState[:,2:4]*1000
#            raise ValueError("Not support this decoder")
#
#        #if(bUseVelocity==1):
#        #    ## Integrate into position
#        #    pos = data.df['binCursorPos'][trial][data.extraBins,None]
#        #    for v in decodeVs:
#        #        CurP = pos[-1]
#        #        nextPos = (CurP+v*dt/1000)
#        #        pos = np.vstack((pos,nextPos))
#
#        #decodeState = decoder.decodeFromData(data, spike, trial)
#        #_P = decodeState[:,0:2]
#        #_V = decodeState[:,2:4]*1000
#
#        ## get the final Postion by mixing the decodedP and decodedV.
#        #pos = data.df['binCursorPos'][trial][data.extraBins-1,None]
#        # decodeFromData: {P0,V0, P1,V1, P2,V2, ... Pt,Vt}, where P0 and V0 are given by data.
#        #print(_P, _V)
#        #pos = _P[0,None]
#        #for ii in range(len(_V)-1):
#        #    lastPos = pos[-1]
#        #    nextPos = (1-alpha)*_P[ii]+alpha*(lastPos + _V[ii]*dt/1000)
#        #    pos = np.vstack((pos,nextPos))
#        pos = _P
#
#
#        decodePos.append(pos)
#        #decodeVel.append(_V)
#        encodeSpikes.append(spike)
##        encodeSpikesRate.append(spikeRate)
#    decodePos = np.array(decodePos,dtype=object)
#    decodeVel = np.array(decodeVel,dtype=object)
#    encodeSpikes = np.array(encodeSpikes,dtype=object)
##    encodeSpikesRate = np.array(encodeSpikesRate,dtype=object)
#
#    ##
#    if(encoder=='real'):
#        #print('--Real data--')
#        binHandPos   = [a[data.extraBins:] for a in data.df['binHandPos'][data.testingTrials]]
#        binCursorPos = [a[data.extraBins:] for a in data.df['binCursorPos'][data.testingTrials]]
#        binHandVel   = [a[data.extraBins:] for a in data.df['binHandVel'][data.testingTrials]]
#        binCursorVel = [a[data.extraBins:] for a in data.df['binCursorVel'][data.testingTrials]]
#        PSTH     = spikesToPSTH(data,encodeSpikes)
#        assert PSTH.shape[1]==data.numBins
#
#        # neural dynamics
#        pcaReal = PCA(n_components=min(PSTH.shape[0]*PSTH.shape[1],PSTH.shape[2]))
#        pcaReal.fit(PSTH.reshape(-1,PSTH.shape[2]))
#
#        neuralDynamics = pcaReal.transform(PSTH.reshape(-1,PSTH.shape[2]))
#        neuralDynamics = neuralDynamics.reshape(PSTH.shape[0],PSTH.shape[1],-1)
#
#        rt['PSTH'] = PSTH
#        rt['neuralDynamics'] = neuralDynamics
#
#        if(len(data.testingTrials)!=0):
#            rt['MSE_DecodedPosition_Testing'] = MeanDiffDistance(np.vstack(binCursorPos),np.vstack(decodePos[data.testingTrials]),axis=1)
#            #rt['MSE_DecodedVelocity_Testing'] = MeanDiffDistance(np.vstack(binCursorVel), np.vstack(decodeVel[data.testingTrials]) , axis=1)
#            #rt['PCC_DecodedVelocity_Testing'] = np.mean(get_rho(binCursorVel,decodeVel[data.testingTrials]))
#        else:
#            rt['MSE_DecodedPosition_Testing'] = -1
#        rt['PCC_PSTH'] = -1
#
#        #print('MSE of decode position (decodePosReal,handPos): {0:.2f} '.format(MeanDiffDistance(np.vstack(decodePos),np.vstack(binHandPos),axis=1)))
#        #print('MSE of decode position (decodePosReal,decodePos): {0:.2f} '.format(MeanDiffDistance(np.vstack(decodePos),np.vstack(binCursorPos),axis=1)))
#        #if(bUseVelocity):
#        #    print('PCC of velocity (decodeVel,binHandVel):',get_rho(np.vstack(decodeVel),np.vstack(binHandVel)))
#        #    print('PCC of velocity (decodeVel,binCursorVel):',get_rho(np.vstack(decodeVel),np.vstack(binCursorVel)))
#    else:
#
#        rtReal = EncodeDecodeTest(data=data,encoder='real',decoder=decoder)
#        decodePosReal, decodeVelReal, encodeSpikesReal = [ rtReal['decodePos'], rtReal['decodeVel'], rtReal['encodeSpikes'] ]
#        rt['decodeVelReal'] = decodeVelReal
#        rt['decodePosReal'] = decodePosReal
#        rt['encodeSpikesReal'] = encodeSpikesReal
#        print(np.vstack(decodePos).shape, np.vstack(decodePosReal).shape)
#        rt['MSE_DecodedPosition'] = MeanDiffDistance(np.vstack(decodePos),np.vstack(decodePosReal),axis=1)
#        rt['MDE_DecodedPosition_centerOut'] = MeanDiffDistance(np.vstack(decodePos[data.centerOutTrials]),np.vstack(decodePosReal[data.centerOutTrials]),axis=1)
##np.intersect1d(data9_b10_ext100.centerOutTrials,data9_b10_ext100.testingTrials)
#        if(len(data.testingTrials)!=0):
#            rt['MSE_DecodedPosition_Testing'] = MeanDiffDistance(np.vstack(decodePos[data.testingTrials]),np.vstack(decodePosReal[data.testingTrials]),axis=1)
#        else:
#            rt['MSE_DecodedPosition_Testing'] = -1
#        if(len(data.trainingTrials)!=0):
#            rt['MSE_DecodedPosition_Training'] = MeanDiffDistance(np.vstack(decodePos[data.trainingTrials]),np.vstack(decodePosReal[data.trainingTrials]),axis=1)
#        else:
#            rt['MSE_DecodedPosition_Training']  = -1
#
#        #print('--{} with {}--'.format(type(encoder).__name__,type(decoder).__name__))
#
#        ##################################
#        ## Print MSE of decode Position ##
#        ##################################
#
#        #print('MSE of all decode position (decodePos,decodePosReal): {0:.2f} '.format(MeanDiffDistance(np.vstack(decodePos),np.vstack(decodePosReal),axis=1)))
#        #if(len(data.testingTrials)!=0):
#        #    print('MSE of testing trials decode position (decodePos,decodePosReal): {0:.2f} '.format(MeanDiffDistance(np.vstack(decodePos[data.testingTrials]),np.vstack(decodePosReal[data.testingTrials]),axis=1)))
#        #if(len(data.trainingTrials)!=0):
#        #    print('MSE of training trials decode position (decodePos,decodePosReal): {0:.2f} '.format(MeanDiffDistance(np.vstack(decodePos[data.trainingTrials]),np.vstack(decodePosReal[data.trainingTrials]),axis=1)))
#
#        ### Print MSE of averaged decode position --
#        #decodePosRealAvg = np.array([np.mean(decodePosReal[data.TrialsOfTargets[targetNum]],axis=0) for targetNum in range(len(data.TrialsOfTargets))])
#        #decodePosAvg     = np.array([np.mean(decodePos[data.TrialsOfTargets[targetNum]],axis=0) for targetNum in range(len(data.TrialsOfTargets))])
#        #print('MSE of avg decode position (decodePosAvg,decodePosRealAvg): {0:.2f} '.format(MeanDiffDistance(decodePosAvg,decodePosRealAvg,axis=2)))
#        #plt.figure()
#        #plt.xlim([-150,150])
#        #plt.ylim((-150,150))
#        #[plt.plot(decodePosRealAvg[targetNum,:,0],decodePosRealAvg[targetNum,:,1]) for targetNum in range(len(decodePosRealAvg))]
#
#        #plt.figure()
#        #plt.xlim([-150,150])
#        #plt.ylim((-150,150))
#        #[plt.plot(decodePosAvg[targetNum,:,0],decodePosAvg[targetNum,:,1]) for targetNum in range(len(decodePosAvg))]
#
#        ##################################
#        ## Print MSE of decode velocity ##
#        ##################################
#
#        #if(bUseVelocity):
#        #rt['MSE_DecodedVelocity'] = MeanDiffDistance( np.vstack(decodeVel) , np.vstack(decodeVelReal) , axis=1)
#        #rt['PCC_DecodedVelocity'] = np.mean(get_rho(decodeVel,decodeVelReal))
#
#        #if(len(data.testingTrials)!=0):
#        #    rt['PCC_DecodedVelocity_Testing'] = np.mean(get_rho(decodeVel[data.testingTrials],decodeVelReal[data.testingTrials]))
#        #    rt['MSE_DecodedVelocity_Testing'] = MeanDiffDistance(np.vstack(decodeVel[data.testingTrials]), np.vstack(decodeVelReal[data.testingTrials]) , axis=1)
#        #else:
#        #    rt['PCC_DecodedVelocity_Testing'] = -1
#        #    rt['MSE_DecodedVelocity_Testing'] = -1
#        #if(len(data.trainingTrials)!=0):
#        #    rt['PCC_DecodedVelocity_Training'] = np.mean(get_rho(decodeVel[data.trainingTrials],decodeVelReal[data.trainingTrials]))
#        #    rt['MSE_DecodedVelocity_Training'] = MeanDiffDistance(np.vstack(decodeVel[data.trainingTrials]), np.vstack(decodeVelReal[data.trainingTrials]) , axis=1)
#        #else:
#        #    rt['PCC_DecodedVelocity_Training'] = -1
#        #    rt['MSE_DecodedVelocity_Training'] = -1
#
#
#        ##################################
#        ## Print PSTH similarity        ##
#        ##################################
#
#        PSTH     = spikesToPSTH(data,encodeSpikes)
#        PSTHReal = spikesToPSTH(data,encodeSpikesReal)
#
#        assert PSTH.shape[1]==data.numBins
#        assert PSTHReal.shape[1]==data.numBins
#        corr     = np.nanmean(get_rho(PSTH,PSTHReal))
#        R2       = np.mean(get_R2(PSTHReal,PSTH))
#        rt['PSTH'] = PSTH
#        rt['PSTHReal'] = PSTHReal
#        rt['PCC_PSTH'] = corr
#        rt['R2_PSTH']  = R2
#        rt['MSE_PSTH'] = ((PSTH-PSTHReal)**2).mean()
#        print("MSE_PSTH:",rt['MSE_PSTH'])
#
#        highFiringRateNeurons = np.vstack(PSTHReal).sum(axis=0)/8/32>1
#        corr     = np.nanmean(get_rho(PSTH[:,:,highFiringRateNeurons], PSTHReal[:,:,highFiringRateNeurons]))
#        R2       = np.mean(get_R2(PSTHReal[:,:,highFiringRateNeurons], PSTH[:,:,highFiringRateNeurons]))
#        rt['PCC_PSTH_high'] = corr
#        rt['R2_PSTH_high']  = R2
#
#        if(len(data.trainingTrials)!=0):
#            PSTH     = spikesToPSTH(data,encodeSpikes)
#            PSTHReal = spikesToPSTH(data,encodeSpikesReal)
#            corr     = np.nanmean(get_rho(PSTH,PSTHReal))
#            R2       = np.mean(get_R2(PSTHReal,PSTH))
#            rt['PSTH_Training']     = PSTH
#            rt['PSTHReal_Training'] = PSTHReal
#            rt['PCC_PSTH_Training'] = corr
#            rt['R2_PSTH_Training']  = R2
#        else:
#            rt['PSTH_Training']     = 0
#            rt['PSTHReal_Training'] = 0
#            rt['PCC_PSTH_Training'] = 0
#            rt['R2_PSTH_Training']  = 0
#
#
#
#
#        if(len(data.testingTrials)!=0):
#            PSTH     = spikesToPSTH(data,encodeSpikes)
#            PSTHReal = spikesToPSTH(data,encodeSpikesReal)
#            corr     = np.nanmean(get_rho(PSTH,PSTHReal))
#            R2       = np.mean(get_R2(PSTHReal,PSTH))
#            rt['PSTH_Testing']      = PSTH
#            rt['PSTHReal_Testing']  = PSTHReal
#            rt['PCC_PSTH_Testing']  = corr
#            rt['R2_PSTH_Testing']   = R2
#
#        else:
#            rt['PSTH_Testing']      = 0
#            rt['PSTHReal_Testing']  = 0
#            rt['PCC_PSTH_Testing']  = 0
#            rt['R2_PSTH_Testing']   = 0
#
#        ##################################
#        ## Print binSpikes similarity  ##
#        ##################################
#        corr = get_rho(np.vstack(encodeSpikesReal),np.vstack(encodeSpikes))
#        R2 = get_R2(np.vstack(encodeSpikesReal),np.vstack(encodeSpikes))
#        rt['PCC_binSpikes'] = np.nanmean(corr)
#        rt['R2_binSpikes']  = np.mean(R2)
#
#        ##################################
#        ## Show PCA, dimensionality     ##
#        ##################################
#
#        #PSTH     = rt['PSTH']
#        #PSTHReal = rt['PSTHReal']
#
#        PSTH = spikesToPSTH(data,encodeSpikes)
#        PSTHReal = spikesToPSTH(data,encodeSpikesReal)
#
#        pca     = PCA(n_components=min(PSTH.shape[0]*PSTH.shape[1],PSTH.shape[2]))
#        pca.fit(PSTH.reshape(-1,PSTH.shape[2]))
#        pcaReal = PCA(n_components=min(PSTHReal.shape[0]*PSTHReal.shape[1],PSTHReal.shape[2]))
#        pcaReal.fit(PSTHReal.reshape(-1,PSTHReal.shape[2]))
#
#        neuralDynamicsReal = pcaReal.transform(PSTHReal.reshape(-1,PSTHReal.shape[2]))
#        neuralDynamics = pcaReal.transform(PSTH.reshape(-1,PSTH.shape[2]))
#        dimOfNeuralDynamics = np.where(pca.explained_variance_ratio_.cumsum()>0.9)[0][0]
#        dimOfNeuralDynamicsReal = np.where(pcaReal.explained_variance_ratio_.cumsum()>0.9)[0][0]
#        rt['PCA'] = pca
#        rt['PCA_Real'] = pcaReal
#        rt['MSE_NeuralDynamics'] = mean_square_error(neuralDynamics[:,0:dimOfNeuralDynamicsReal+1],neuralDynamicsReal[:,0:dimOfNeuralDynamicsReal+1],axis=1)
#        rt['dimOfNeuralDynamicsReal'] = dimOfNeuralDynamicsReal
#        rt['neuralDynamics'] = neuralDynamics
#        rt['neuralDynamicsReal'] = neuralDynamicsReal
#        #ax.plot3D(xline, yline, zline)
#        #print('For this model, how many PCs are needed to capture 80% variance: {0:d} ({1:.2%})'.format(dimOfNeuralDynamics+1, pca.explained_variance_ratio_.cumsum()[dimOfNeuralDynamics]))
#        #print('For real spikes data, how many PCs are needed to capture 80% variance: {0:d} ({1:.2%})'.format(dimOfNeuralDynamicsReal+1, pcaReal.explained_variance_ratio_.cumsum()[dimOfNeuralDynamicsReal]))
#        #print('MSE of neural dynamics:',MeanDiffDistance(neuralDynamics[:,0:dimOfNeuralDynamicsReal+1],neuralDynamicsReal[:,0:dimOfNeuralDynamicsReal+1],axis=1))
#
#        ##################################
#        ## Plot the decodeP             ##
#        ##################################
#
#        #plt.figure()
#        #plt.title('--{} with {}--'.format(type(encoder).__name__,type(decoder).__name__))
#        #plt.xlim([-150,150])
#        #plt.ylim((-150,150))
#        #for trial in plottingTrials:
#        #    #plt.figure()
#        #    #plt.title('--{} with {}--'.format(type(encoder).__name__,type(decoder).__name__))
#        #    #plt.xlim([-120,120])
#        #    #plt.ylim((-120,120))
#        #    #plt.plot(data.df['binHandPos'][trial].T[0],data.df['binHandPos'][trial].T[1])
#        #    plt.plot(decodePos[trial].T[0],decodePos[trial].T[1],'b')
#        #    plt.plot(decodePosReal[trial].T[0],decodePosReal[trial].T[1],'r',label='Real')
#        #blue_patch = mpatches.Patch(color='b', label=format(type(encoder).__name__))
#        #red_patch = mpatches.Patch(color='r', label='real')
#        #plt.legend(handles=[blue_patch,red_patch])
#
#        ##
#        ## Plot the PSTH ---
#        ##
#        #PSTH=spikesToPSTH(data,encodeSpikesRate,numBins=data.numBins)
#        #PSTHReal = spikesToPSTH(data,encodeSpikesRateReal,numBins=data.numBins)
#
#    ## return dic
#    rt['data'] = data
#    rt['dt']   = data.dt
#    rt['centerOutTrials']  = data.centerOutTrials
#    rt['TrialsOfTargets']  = data.TrialsOfTargets
#    rt['trainingTrials']   = data.trainingTrials
#    rt['testingTrials']    = data.testingTrials
#    rt['decodePos']        = decodePos
#    rt['decodeVel']        = decodeVel
#    rt['encodeSpikes']     = encodeSpikes
#    #rt['encodeSpikesRate'] = encodeSpikesRate
#    return rt



def spikesToPSTH(data, spikes):
    '''
    spikes: numTrial x numChannels
    '''
    PSTH=[]
    for targetIdx, target in enumerate(data.uniqueTarget):
        trials = data.TrialsOfTargets[tuple(target)]
        lens = [len(i) for i in spikes[trials]]
        sumSpikes = np.ma.empty((np.max(lens),len(trials),192))
        sumSpikes.mask = True
        for idx, l in enumerate(spikes[trials]):
            sumSpikes[:len(l),idx] = l
        PSTH.append(np.asarray(sumSpikes.mean(axis=1)))

    return np.array(PSTH)


def RecordToFile(rt,PlotTrials = range(401,417,2),save_folder = "../result/JNE2021/"):
    print("save files to {}".format(save_folder))
    print("MSE of decoded posiiton (Testing) : %d" % rt['MSE_DecodedPosition_Testing'])
    if 'MSE_DecodedVelocity_Testing' in rt:
        print("MSE of decoded velocity (Testing) : %.2f" % rt['MSE_DecodedVelocity_Testing'])
    if 'PCC_DecodedVelocity_Testing' in rt:
        print("PCC of decoded velocity (Testing) : %.2f" % rt['PCC_DecodedVelocity_Testing'])
    print("PCC of PSTH : {:.4f}".format(rt['PCC_PSTH']))
    ##########################################################
    #print("encoder :", rt['encoder'].name,file=f)
    #print("decoder :", rt['decoder'].name,file=f)
    #f.write("MSE of decoded movement : %d\n" % rt['MSE_DecodedPosition'])
    #f.write("MSE of decoded movement (Training) : %d\n" % rt['MSE_DecodedPosition_Training'])
    #f.write("MSE of decoded movement (Testing) : %d\n" % rt['MSE_DecodedPosition_Testing'])
    #print("PCC of PSTH : {:.4f}".format(rt['PCC_PSTH']),file=f)
    #print("PCC of PSTH (Training) : {:.4f}".format(rt['PCC_PSTH_Training']),file=f)
    #print("PCC of PSTH (Testing) : {:.4f}".format(rt['PCC_PSTH_Testing']),file=f)
    #print("R2 of PSTH : {:.2f}".format(rt['R2_PSTH']),file=f)
    #print("R2 of PSTH (Training) : {:.2f}".format(rt['R2_PSTH_Training']),file=f)
    #print("R2 of PSTH (Testing) : {:.2f}".format(rt['R2_PSTH_Testing']),file=f)
    #print("PCC of decoded vlocity :",rt['PCC_DecodedVelocity'],file=f)
    #print("PCC of decoded vlocity (Training) :",rt['PCC_DecodedVelocity_Training'],file=f)
    #print("PCC of decoded vlocity (Testing) :",rt['PCC_DecodedVelocity_Testing'],file=f)
    #print("MSE of neural dynamics : ",rt['MSENeuralDynamics'],file=f)
    #SavePSTHForjPCA(rt,save_folder)
    SavePSTHPDF(rt,save_folder)
    #SaveSpikeRasterPDF(rt,save_folder)
    #SaveNeuralDynamicsPDF(rt,save_folder)
    SaveDecodesPDF(rt,PlotTrials,save_folder)
    #SaveDecodes(rt)

def SavePSTHForjPCA(rt,save_folder):
    PSTH   =rt['PSTH']
    numBins=rt['data'].numBins
    dt     =rt['data'].dt
    time   =np.array(range(0,numBins*dt,dt)).reshape(numBins,1)
    dt     = np.dtype(dtype=[('A', 'O'), ('times', 'O')])
    x = np.array([(PSTH[0],time)
                  ,(PSTH[1],time)
                  ,(PSTH[2],time)
                  ,(PSTH[3],time)
                  ,(PSTH[4],time)
                  ,(PSTH[5],time)
                  ,(PSTH[6],time)
                  ,(PSTH[7],time)
                 ]
                 ,dtype=dt)
    if(rt['encoder']=='real'):
        name='real'+"_"+rt['testName']
    else:
        name = rt['encoder'].name+"_"+rt['testName']
    sio.savemat(save_folder+'PSTH_'+name+'.mat', {'Data':x})

def SaveDecodesPDF(rt, PlotTrials,save_folder):
    fig, ax = plt.subplots(figsize=(5,5))
    if 'decodePosReal' in rt.keys():
        decodes = rt['decodePosReal']
        for ii in PlotTrials:
            ax.plot(decodes[ii][:,0],decodes[ii][:,1],'r',label='Real')
    if 'decodePos' in rt.keys():
        decodes = rt['decodePos']
        for ii in PlotTrials:
            ax.plot(decodes[ii][:,0],decodes[ii][:,1],'b',label='Synthetic')

    handles, labels = ax.get_legend_handles_labels()
    display = (0,len(PlotTrials))
    ax.legend([handle for i,handle in enumerate(handles) if i in display],
          [label for i,label in enumerate(labels) if i in display], loc = 'best')
    plt.xlim([-200,200])
    plt.ylim([-200,200])
    plt.axis("off")
    name = "decodes_{}_{}.pdf".format(rt['encoder'].name, rt['decoder'].name)
    plt.savefig(save_folder+name, format="pdf")
    plt.show()

def SaveNeuralDynamicsPDF(rt,save_folder):
    fig, ax = plt.subplots(figsize=(5,5))
    PSTH = rt['PSTH']
    neuralDynamics = rt['neuralDynamics'].reshape(PSTH.shape[0],PSTH.shape[1],-1)
    for ii,target in enumerate(rt['data'].uniqueTarget):
        xline=neuralDynamics[ii,:,0]
        yline=neuralDynamics[ii,:,1]
        zline=neuralDynamics[ii,:,2]
        color = rt['data'].ColorOfTarget[tuple(target)]
        plt.plot(xline,yline,color=color)
    plt.title("Neural dynamics of synthetic spikes")
    name = "NeuralDynamics_{}_{}.pdf".format(rt['encoder'].name, rt['decoder'].name)
    plt.savefig(save_folder+name, format="pdf")


def SavePSTHPDF(rt,save_folder,neuron_list = range(0,30,6)):
    dt = rt['data'].dt
    PSTH = rt['PSTH']
    PSTHReal = rt['PSTHReal']
    fig, axes = plt.subplots(len(neuron_list), 2, sharex=True,sharey=True)
    if len(neuron_list)>2:
        plt.sca(axes[0,0])
        plt.title('Synthetic')
        plt.sca(axes[0,1])
        plt.title('Real')
        fig.set_figwidth(10)
        fig.set_figheight(10)
        i=0
        for neuronIdx in neuron_list:
            for targetIdx,target in enumerate(rt['data'].uniqueTarget):
                color = rt['data'].ColorOfTarget[tuple(target)]
                PlotPSTH(axes[i,0],PSTH[targetIdx].T[neuronIdx],dt,color)
                PlotPSTH(axes[i,1],PSTHReal[targetIdx].T[neuronIdx],dt,color)
            i+=1

    else:
        plt.sca(axes[0])
        plt.title('Synthetic')
        plt.sca(axes[1])
        plt.title('Real')
        neuronIdx = neuron_list[0]
        for targetIdx,target in enumerate(rt['data'].uniqueTarget):
                color = rt['data'].ColorOfTarget[tuple(target)]
                PlotPSTH(axes[0],PSTH[targetIdx].T[neuronIdx],dt,color)
                PlotPSTH(axes[1],PSTHReal[targetIdx].T[neuronIdx],dt,color)


    
    name = "PSTH_{}_{}.pdf".format(rt['encoder'].name, rt['decoder'].name)
    plt.savefig(save_folder+name, format="pdf")


def SaveSpikeRasterPDF(rt,save_folder):
    ##################################
    ## Plot Spike Raster            ##
    ##################################
    data = rt['data']
    neuronIdx=0
    gen=StGen()
    fig, axes = plt.subplots(1,8,figsize=(40, 5))
    target_idx =0
    for target in data.uniqueTarget:#range(len(data.TrialsOfTargets)):
        color = data.ColorOfTarget[tuple(target)]
        spike_list=[]
        for trial in data.TrialsOfTargets[tuple(target)]:
            spikeRate = rt['encodeSpikes'][trial][:,neuronIdx]
            spikeTrain = gen.inh_poisson_generator(spikeRate*1000/data.dt,range(0,len(spikeRate)*data.dt,data.dt),len(spikeRate)*data.dt,array=True)
            spike_list.append(spikeTrain)

        for ith, trial in enumerate(spike_list):
            axes[target_idx].vlines(trial, ith + .5, ith + 1.5,color=color)
            axes[target_idx].set_ylim(.5, len(spike_list) + .5)
            axes[target_idx].set_xlim(.5, 1200)
            axes[target_idx].get_xaxis().set_visible(False)
            axes[target_idx].get_yaxis().set_visible(False)
        target_idx+=1

    name = "SpikeRaster_{}_{}.pdf".format(rt['encoder'].name, rt['decoder'].name)
    plt.savefig(save_folder+name, format="pdf")


def PlotGradientColorLine(x,y,color,**arg):
    assert len(x)==len(y)
    npointsHiRes = len(x)
    for i in range(npointsHiRes-1):
        plt.plot(x[i:i+2],y[i:i+2],alpha=float(i)/(npointsHiRes-1),color=color,**arg)


#def SaveDecodesPDF(rt, PlotTrials):
#    plt.figure()
#    #name = rt['encoder'].name+"_"+rt['decoder'].name+"_"+rt['testName']
#
#    if('decodePos' in rt.keys()):
#        decodes = rt['decodePos']
#        for ii in PlotTrials:
#            plt.plot(decodes[ii][:,0],decodes[ii][:,1],'b')
#
#    if('decodePosReal' in rt.keys()):
#        decodesReal = rt['decodePosReal']
#        for ii in PlotTrials:
#            plt.plot(decodesReal[ii][:,0],decodesReal[ii][:,1],'r')
#    plt.xlim([-250,250])
#    plt.ylim([-250,250])
#    plt.show()
#    #plt.savefig(saveFolder+"decodes_"+name+".pdf", format="pdf")


#def SaveDecodes(rt):
#    decodes = rt['decodePos']
#    decodesReal = rt['decodePosReal']
#    decodeVel = rt['decodeVel']
#    decodeVelReal = rt['decodeVelReal']
#    centerOutTrials = rt['data'].centerOutTrials
#    name = rt['encoder'].name+"_"+rt['decoder'].name+"_"+rt['testName']
#    sio.savemat(saveFolder+"decodes_"+name+".mat", {'decodes':decodes,
#                                                    'decodesReal':decodesReal,
#                                                    'decodeVel':decodeVel,
#                                                    'decodeVelReal':decodeVelReal,
#                                                    'centerOutTrials':centerOutTrials})
