import json
import sys
import time
import collections
#sys.path.append('../../brain_models/code')
#sys.path.append('../../brain_models/code/encoder')
from encoder.PPVT import PPVTEncoder
#from PD import PDEncoder
from TrainEncoder import SimpleNoiseRNNCell
import numpy as np
import keras
import tensorflow as tf
from keras.backend import exp
import keras.backend as K

class NeuralEncoder(object):
    def __init__(self):
        self.encoder = None
        self.refresh_rate = None
        self.encoder_input_shape = 'pv'
        self.b_synthesize = True
        self.b_calculate_ratio = False
        self.RNN_contribution_ratio = None
        self.reset()

    def summary(self):
        #self.encoder_type, self.refresh_rate, self.date, self.encoder_name
        print("#### Summary of NeuralEncoder ####")
        print("     dt", self.refresh_rate)
        print("     PMd_delay", self.PMd_delay)
        print("     M1_delay" , self.M1_delay)
        print("     b_synthesize", self.b_synthesize)
        print("     encoder type", self.encoder_type)
        print("     encoder_name", self.encoder_name)

    def reset(self):
        if isinstance(self.encoder,keras.models.Model):
            self.encoder.reset_states()

        self.PMd_delay = 0
        self.M1_delay = 0
        self.Q_target   = np.zeros((100,2))
        self.Q_hand_pos = np.zeros((100,2))
        self.Q_hand_vel = np.zeros((100,2))
        self.Q_cursor_pos = np.zeros((100,2))
        self.Q_cursor_vel = np.zeros((100,2))
        self.Q_spike_counts = np.zeros((100,192))
        self.Q_spike_rates = np.zeros((100,192))

        self.RNN_contribution_ratio = []


    def _checkSetting(self):
        pass


    def setEncoder(self,encoder):
        self.encoder=encoder
        if isinstance(encoder,PPVTEncoder):
            self.b_calculate_ratio=False
        else:
            self.b_calculate_ratio=True
            



    def loadEncoder(self):
        self._checkSetting()
        if self.b_synthesize:
            encoder_path = self.encoder_path
            encoder_type, refresh_rate, encoder_name= self.encoder_type, self.refresh_rate, self.encoder_name
            model_path = encoder_path+encoder_name

            print("Load encoder model from {}".format(model_path))
            if encoder_type=='RNN':
                encoder = keras.models.load_model(model_path
                        ,custom_objects={'SimpleNoiseRNNCell':SimpleNoiseRNNCell,'exp':exp,'tf':tf})
            elif encoder_type=='PPVT':
                encoder = PPVTEncoder()
                encoder.load(model_path)
            else:
                raise ValueError

            self.encoder = encoder
        else:
            print("Ignore loading encoder. Please turn on exp_setting['b_synthesize'] if needed")

    def encode(self
                ,hand_p
                ,target_p
                ,cursor_p
                ):

        M1_delay = self.M1_delay
        PMd_delay = self.PMd_delay
        refresh_rate = self.refresh_rate

        Q_target   = self.Q_target

        Q_hand_pos = self.Q_hand_pos
        Q_hand_vel = self.Q_hand_vel
        Q_cursor_pos = self.Q_cursor_pos
        Q_cursor_vel = self.Q_cursor_vel

        Q_spike_counts = self.Q_spike_counts
        Q_spike_rates = self.Q_spike_rates


        hand_p = hand_p.reshape(1,2) #(mm)
        cursor_p = np.array(cursor_p).reshape(1,2)
        target_p = np.array(target_p).reshape(1,2)

        hand_v = (hand_p-Q_hand_pos[-1])/float(refresh_rate)*1000 #(mm/s)
        cursor_v = (cursor_p-Q_cursor_pos[-1])/float(refresh_rate)*1000 #(mm/s)

        Q_hand_pos = np.vstack((Q_hand_pos,hand_p))
        Q_hand_pos = Q_hand_pos[-100:]
        Q_cursor_pos = np.vstack((Q_cursor_pos,cursor_p))
        Q_cursor_pos = Q_cursor_pos[-100:]

        Q_hand_vel = np.vstack((Q_hand_vel,hand_v))
        Q_hand_vel = Q_hand_vel[-100:]
        Q_cursor_vel = np.vstack((Q_cursor_vel,hand_v))
        Q_cursor_vel = Q_cursor_vel[-100:]

        Q_target   = np.vstack((Q_target,target_p))
        Q_target   = Q_target[-100:]

        if self.encoder!=None:
            bins = self.encoder.input_shape[1]
            if (self.encoder_input_shape == 'v'):
                encoder_in = np.hstack([Q_hand_vel[-bins:]]).reshape(1,bins,-1)
            elif (self.encoder_input_shape == 'pv'):
                encoder_in = np.hstack([Q_hand_pos[-bins:],Q_hand_vel[-bins:]]).reshape(1,bins,-1)
            elif (self.encoder_input_shape == 'pvpv'):
                encoder_in = np.hstack([Q_hand_pos[-bins:],Q_hand_vel[-bins:],Q_cursor_pos[-bins:],Q_cursor_vel[-bins:]]).reshape(1,bins,-1)
            elif (self.encoder_input_shape == 'pvt'):
                encoder_in = np.hstack([Q_hand_pos[-bins:],Q_hand_vel[-bins:],Q_target[-bins:]]).reshape(1,bins,-1)
            elif (self.encoder_input_shape == 'pvtpv'):
                encoder_in = np.hstack([Q_hand_pos[-bins:],Q_hand_vel[-bins:],Q_target[-bins:],Q_cursor_pos[-bins:],Q_cursor_vel[-bins:]]).reshape(1,bins,-1)

            if self.b_calculate_ratio:
                state = K.get_value(self.encoder.layers[1].states[0])
                W_In = K.get_value(self.encoder.layers[1].weights[0])
                W_Re = K.get_value(self.encoder.layers[1].weights[1])
                In_contri = np.dot(encoder_in,W_In).squeeze()
                Re_contri = np.dot(state,W_Re).squeeze()
                ratio_contri = In_contri/Re_contri
                self.RNN_contribution_ratio.append(ratio_contri)

            if self.b_synthesize==True:
                pseudo_spike_rates = self.encoder.predict(encoder_in)
                pseudo_spike_counts = np.random.poisson(pseudo_spike_rates)
        else:
            pseudo_spike_rates = np.zeros(192)
            pseudo_spike_counts = np.zeros(192)

        Q_spike_counts = np.vstack((Q_spike_counts[-99:],pseudo_spike_counts))
        Q_spike_rates = np.vstack((Q_spike_rates[-99:],pseudo_spike_rates))

        rt_spike_counts = np.hstack([ Q_spike_counts[None,-max(M1_delay,PMd_delay)+M1_delay-1,:96],Q_spike_counts[None,-max(M1_delay,PMd_delay)+PMd_delay-1,96:]])
        rt_spike_rates = np.hstack([ Q_spike_rates[None,-max(M1_delay,PMd_delay)+M1_delay-1,:96],Q_spike_rates[None,-max(M1_delay,PMd_delay)+PMd_delay-1,96:]])


        self.Q_hand_pos = Q_hand_pos
        self.Q_hand_vel = Q_hand_vel
        self.Q_cursor_pos = Q_cursor_pos
        self.Q_cursor_vel = Q_cursor_vel

        self.Q_target   = Q_target

        self.Q_spike_counts = Q_spike_counts
        self.Q_spike_rates = Q_spike_rates

        assert rt_spike_counts.shape==(1,192)
        assert rt_spike_rates.shape==(1,192)
        return rt_spike_counts, rt_spike_rates


