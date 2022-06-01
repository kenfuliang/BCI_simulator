import numpy as np
import pandas as pd

class AttentiveRNN(object):
    def __init__(self,model=None,inputShape='tpv', extraBins=0):
        self.model = model
        self.inputShape_ = inputShape
        self.extraBins_  = extraBins
#
#    def encode(self,data,trial,lastSpikes=None):
#        extraBins = self.extraBins_
#
#        v = data.df['binVelocity'][trial]
#        t = np.repeat(data.df['target'][trial].reshape(1,2),v.shape[0],axis=0)
#        p = data.df['binHandPos'][trial][1:]
#        n = data.df['binSpike'][trial]
#
#        t_in=[]
#        p_in=[]
#        v_in=[]
#        rt_spike = n[0,None]
#
#        for timestamp in range(len(p)-extraBins):
#            t_in.append(t[timestamp:timestamp+extraBins+1])   
#            p_in.append(p[timestamp:timestamp+extraBins+1])   
#            v_in.append(v[timestamp:timestamp+extraBins+1])   
#
#        if(self.inputShape_ == 'tpv'):
#            kinematics = np.concatenate([t_in,p_in,v_in],axis=2)
#        elif(self.inputShape_ == 'pv'):
#            kinematics = np.concatenate([p_in,v_in],axis=2)
#
#        rt_spike = self.model.predict(kinematics)[:,-1]
#        #print(rt_spike.shape)
#
#        return rt_spike






    def encode(self,data,trial,lastSpikes=None):
        extraBins = self.extraBins_

        v = data.df['binHandVel'][trial]
        t = np.repeat(data.df['target'][trial].reshape(1,2),v.shape[0],axis=0)
        p = data.df['binHandPos'][trial][1:]
        n = data.df['binSpike'][trial]

        rt_spike = n[0,None]

        if(self.inputShape_=='tpv'):
            kinematics = np.hstack((t,p,v))
        elif(self.inputShape_=='tp'):
            kinematics = np.hstack((t,p))
        elif(self.inputShape_=='tv'):
            kinematics = np.hstack((t,v))
        elif(self.inputShape_=='pv'):
            kinematics = np.hstack((p,v))
        elif(self.inputShape_=='p'):
            kinematics = p
        elif(self.inputShape_=='v'):
            kinematics = v

        #rt_spike = np.squeeze(self.model.predict(kinematics),axis=0)
        rt_spike = np.squeeze(self.model.predict(kinematics[np.newaxis,0:data.numBins+extraBins]),axis=0)[extraBins:data.numBins+extraBins]

        return rt_spike

    def summary(self):
        print('Input shape:',self.inputShape_)
        self.model.summary() 
        
