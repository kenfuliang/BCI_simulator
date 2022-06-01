import numpy as np
import pandas as pd

class MLP(object):
    def __init__(self, model=None, inputShape='tpv', extraBins=0, name="MLP"):
        self.model       = model
        self.inputShape_ = inputShape
        self.name        = name
        self.bins        = model.input_shape[1]
        self.input_history = np.zeros(model.input_shape[1:])

    def encode(self,encoder_input):
        if self.inputShape_=='pv':
            kinematics = np.hstack((encoder_input['p'],encoder_input['v']))
        self.input_history = np.vstack((self.input_history,kinematics))[1:]
        rt_spike = self.model.predict(np.array([self.input_history]))

        return rt_spike



    def encodeFromData(self,data,trial):
        #extraBins = data.extraBins
        extraBins = self.model.input_shape[1]-1

        hand_p = data.df['binHandPos'][trial][1:]
        hand_v = data.df['binHandVel'][trial]
        cursor_p = data.df['binCursorPos'][trial][1:]
        cursor_v = data.df['binCursorVel'][trial]

        a = data.df['binHandAcc'][trial]
        t = [data.df['target'][trial]]*hand_v.shape[0] #np.repeat(t,v.shape[0],axis=0)
        n = data.df['binSpike'][trial]


        assert data.extraBins>=extraBins, "Data needs to provide more extraBins to encoding models"

        t_in=[]
        a_in=[]
        hand_p_in=[]
        hand_v_in=[]
        cursor_p_in=[]
        cursor_v_in=[]


        rt_spike = n[0,None]
        for timestamp in range(len(hand_p)-extraBins):
            t_in.append(t[timestamp:timestamp+extraBins+1])
            hand_p_in.append(hand_p[timestamp:timestamp+extraBins+1])
            hand_v_in.append(hand_v[timestamp:timestamp+extraBins+1])
            cursor_p_in.append(cursor_p[timestamp:timestamp+extraBins+1])
            cursor_v_in.append(cursor_v[timestamp:timestamp+extraBins+1])
            a_in.append(a[timestamp:timestamp+extraBins+1])

        if(self.inputShape_ == 'tpvn'):
            for time in range(len(hand_p)-extraBins):
                lastSpike = rt_spike[-1,None]
                kinematics = np.concatenate([t_in[time],hand_p_in[time],hand_v_in[time]],axis=1)
                modelInput = np.hstack([kinematics,lastSpike])
                modelInput = np.expand_dims(modelInput,axis=0)
                spike = self.model.predict(modelInput)
                rt_spike = np.vstack([rt_spike,spike])
            rt_spike = rt_spike[1:]

        elif(self.inputShape_ == 'tpn'):
            for time in range(len(hand_p)-extraBins):
                lastSpike = rt_spike[-1,None]
                kinematics = np.concatenate([t_in[time],hand_p_in[time]],axis=1)
                modelInput = np.hstack([kinematics,lastSpike])
                modelInput = np.expand_dims(modelInput,axis=0)
                spike = self.model.predict(modelInput)
                rt_spike = np.vstack([rt_spike,spike])
            rt_spike = rt_spike[1:]

        elif(self.inputShape_ == 'tvn'):
            for time in range(len(hand_p)-extraBins):
                lastSpike = rt_spike[-1,None]
                kinematics = np.concatenate([t_in[time],hand_v_in[time]],axis=1)
                modelInput = np.hstack([kinematics,lastSpike])
                modelInput = np.expand_dims(modelInput,axis=0)
                spike = self.model.predict(modelInput)
                rt_spike = np.vstack([rt_spike,spike])
            rt_spike = rt_spike[1:]

        elif(self.inputShape_ == 'pvn'):
            for time in range(len(hand_p)-extraBins):
                lastSpike = rt_spike[-1,None]
                kinematics = np.concatenate([hand_p_in[time],hand_v_in[time]],axis=1)
                modelInput = np.hstack([kinematics,lastSpike])
                modelInput = np.expand_dims(modelInput,axis=0)
                spike = self.model.predict(modelInput)
                rt_spike = np.vstack([rt_spike,spike])
            rt_spike = rt_spike[1:]

        elif(self.inputShape_ == 'pn'):
            for time in range(len(hand_p)-extraBins):
                lastSpike = rt_spike[-1,None]
                kinematics = np.concatenate([hand_p_in[time]],axis=1)
                modelInput = np.hstack([kinematics,lastSpike])
                modelInput = np.expand_dims(modelInput,axis=0)
                spike = self.model.predict(modelInput)
                rt_spike = np.vstack([rt_spike,spike])
            rt_spike = rt_spike[1:]

        elif(self.inputShape_ == 'vn'):
            for time in range(len(hand_p)-extraBins):
                lastSpike = rt_spike[-1,None]
                kinematics = np.concatenate([hand_v_in[time]],axis=1)
                modelInput = np.hstack([kinematics,lastSpike])
                modelInput = np.expand_dims(modelInput,axis=0)
                spike = self.model.predict(modelInput)
                rt_spike = np.vstack([rt_spike,spike])
            rt_spike = rt_spike[1:]

        else:

            if(self.inputShape_ == 'tpv'):
                kinematics = np.concatenate([t_in,hand_p_in,hand_v_in],axis=2)
            elif(self.inputShape_ == 'tp'):
                kinematics = np.concatenate([t_in,hand_p_in],axis=2)
            elif(self.inputShape_ == 'tv'):
                kinematics = np.concatenate([t_in,hand_v_in],axis=2)
            elif(self.inputShape_ == 'pv'):
                kinematics = np.concatenate([hand_p_in,hand_v_in],axis=2)
            elif(self.inputShape_ == 'pvpv'):
                kinematics = np.concatenate([hand_p_in,hand_v_in,cursor_p_in,cursor_v_in],axis=2)
            elif(self.inputShape_ == 'pvt'):
                kinematics = np.concatenate([hand_p_in,hand_v_in,t_in],axis=2)
            elif(self.inputShape_ == 'pvtpv'):
                kinematics = np.concatenate([hand_p_in,hand_v_in,t_in,cursor_p_in,cursor_v_in],axis=2)
            elif(self.inputShape_ == 'pva'):
                kinematics = np.concatenate([hand_p_in,hand_v_in,a_in],axis=2)
            elif(self.inputShape_ == 'p'):
                kinematics = np.concatenate([hand_p_in],axis=2)
            elif(self.inputShape_ == 'v'):
                kinematics = np.concatenate([hand_v_in],axis=2)
            rt_spike = self.model.predict(kinematics)

        rt_spike = np.concatenate((data.df['binSpike'][trial][:extraBins],rt_spike,data.df['binSpike'][trial][-1:]))
        return rt_spike

    def summary(self):
        print('Input shape:',self.inputShape_)
        self.model.summary()

