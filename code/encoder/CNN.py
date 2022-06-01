import numpy as np
import pandas as pd

class CNN(object):
    def __init__(self,model=None,inputShape='tpv',extraBins=None):
        self.model = model
        self.inputShape_ = inputShape
        self.extraBins = extraBins


    def encode(self,data,trial,lastSpikes=None):
        extraBins = self.extraBins
        v = data.df['binVelocity'][trial]
        t = np.repeat(data.df['target'][trial].reshape(1,2),v.shape[0],axis=0)
        p = data.df['binHandPos'][trial][1:]
        n = data.df['binSpike'][trial]

        rt_spike = n[0,None]
        t_in=[]
        p_in=[]
        v_in=[]
        n_in=[] 
        if(self.inputShape_ == 'tpv'):
            for timestamp in range(len(p)-extraBins):
                t_in.append(t[timestamp:timestamp+extraBins+1])   
                p_in.append(p[timestamp:timestamp+extraBins+1])   
                v_in.append(v[timestamp:timestamp+extraBins+1])   
            kinematics = np.concatenate([t_in,p_in,v_in],axis=2)
        elif(self.inputShape_ =='pv'):
            for timestamp in range(len(p)-extraBins):
                p_in.append(p[timestamp:timestamp+extraBins+1])   
                v_in.append(v[timestamp:timestamp+extraBins+1])   
            kinematics = np.concatenate([p_in,v_in],axis=2)
        else:
            raise Exception("inputShape_ is not correct")
        kinematics = np.expand_dims(kinematics, axis=3)
        rt_spike = self.model.predict(kinematics)

        return rt_spike

    def summary(self):
        print('Input shape:',self.inputShape_)
        self.model.summary() 
        
