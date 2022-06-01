import numpy as np
import pandas as pd

class RNN(object):
    def __init__(self,model=None,inputShape='pv',return_sequence=False,name="RNN"):
        self.model = model
        self.inputShape_ = inputShape
        self.name = name
        #self.extraBins_  = extraBins

    def encodeFromData(self,data,trial,lastSpikes=None):
        #print("Trial Number:",trial)
        extraBins = self.model.input_shape[1]-1

        h_p = data.df['binHandPos'][trial][1:]
        h_v = data.df['binHandVel'][trial]
        c_p = data.df['binCursorPos'][trial][1:]
        c_v = data.df['binCursorVel'][trial]
        t = np.repeat(data.df['target'][trial],h_v.shape[0],axis=0)
        n = data.df['binSpike'][trial]

        assert data.extraBins>=extraBins, "Data needs to provide more extraBins to encoding models"

        t_in=[]
        h_p_in=[]
        h_v_in=[]
        c_p_in=[]
        c_v_in=[]


        if(self.model.stateful==True):
            rt_spike = np.empty(shape=[len(n),192])
            t_in = t
            h_p_in = h_p
            h_v_in = h_v
            c_p_in = c_p
            c_v_in = c_v


            if(self.inputShape_ == 'tpv'):
                kinematics = np.concatenate([t_in,h_p_in,h_v_in],axis=1)
            elif(self.inputShape_ == 'pv'):
                kinematics = np.concatenate([h_p_in,h_v_in],axis=1)
            elif(self.inputShape_ == 'v'):
                kinematics = np.concatenate([h_v_in],axis=1)
            elif(self.inputShape_ == 'p'):
                kinematics = np.concatenate([h_p_in],axis=1)
            elif(self.inputShape_ == 'pvpv'):
                kinematics = np.concatenate([h_p_in,h_v_in,c_p_in,c_v_in],axis=1)

            self.model.reset_states()
            for ii in range(len(kinematics)):
                rt_spike[ii]=self.model.predict(kinematics[ii,None,None])

            rt_spike[-1] = data.df['binSpike'][trial][-1:]


        elif(self.model.stateful==False):

            for timestamp in range(len(h_p)-extraBins):
                t_in.append(t[timestamp:timestamp+extraBins+1])
                h_p_in.append(h_p[timestamp:timestamp+extraBins+1])
                h_v_in.append(h_v[timestamp:timestamp+extraBins+1])
                c_p_in.append(c_p[timestamp:timestamp+extraBins+1])
                c_v_in.append(c_v[timestamp:timestamp+extraBins+1])


            if(self.inputShape_ == 'tpv'):
                kinematics = np.concatenate([t_in,h_p_in,h_v_in],axis=2)
            elif(self.inputShape_ == 'pv'):
                kinematics = np.concatenate([h_p_in,h_v_in],axis=2)
            elif(self.inputShape_ == 'v'):
                kinematics = np.concatenate([h_v_in],axis=1)
            elif(self.inputShape_ == 'p'):
                kinematics = np.concatenate([h_p_in],axis=1)
            elif(self.inputShape_ == 'pvpv'):
                kinematics = np.concatenate([h_p_in,h_v_in,c_p_in,c_v_in],axis=2)



            # kinematics shape : (N,extraBins+1,4)

            rt_spike = self.model.predict(kinematics)

            #rt_spike = np.concatenate((data.df['binSpike'][trial][:extraBins],rt_spike))

            rt_spike = np.concatenate((data.df['binSpike'][trial][:extraBins],rt_spike,data.df['binSpike'][trial][-1:]))

        return rt_spike

#        v = data.df['binVelocity'][trial]
#        t = np.repeat(data.df['target'][trial].reshape(1,2),v.shape[0],axis=0)
#        p = data.df['binHandPos'][trial][1:]
#        n = data.df['binSpike'][trial]
#
#        rt_spike = n[0,None]
#
#        if(self.inputShape_=='tpv'):
#            kinematics = np.hstack((t,p,v))
#        elif(self.inputShape_=='tp'):
#            kinematics = np.hstack((t,p))
#        elif(self.inputShape_=='tv'):
#            kinematics = np.hstack((t,v))
#        elif(self.inputShape_=='pv'):
#            kinematics = np.hstack((p,v))
#        elif(self.inputShape_=='p'):
#            kinematics = p
#        elif(self.inputShape_=='v'):
#            kinematics = v
#
#        rt_spike = np.squeeze(self.model.predict(kinematics[np.newaxis,0:data.numBins]),axis=0)
#        return rt_spike

    def summary(self):
        print('Input shape:',self.inputShape_)
        self.model.summary()

