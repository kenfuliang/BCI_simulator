'''
Define NeuralDecoder
'''
import json
import sys
import os 
import time
import collections
import scipy.io as sio
import time
import numpy as np
sys.path.append('../code')
sys.path.append('../code/decoder')
#from decoder.OLE import OLEDecoder
#from decoder.wiener import WFDecoder
from decoder.kalman import KFDecoder
from decoder.FORCE import FORCEDecoder
import keras
from keras.backend import exp
class NeuralDecoder(object):
    def __init__(self):
        pixelPitch = 3.1125 #(cm)
        self.screenW,self.screenH = int(1920/pixelPitch),int(1080/pixelPitch)
        self.decoder_type = 'VKF'
        self.alpha = 1.0
        self.pos_cursor_i = None 
        self.Hspikes = None


    def summary(self):
        print("#### Summary of NeuralDecoder ####")
        print("     dt",self.refresh_rate)
        print("     alpha",self.alpha)
        print("     decoder_type",self.decoder_type)
        print("     decoder_path",self.decoder_path)
        print("     decoder_name",self.decoder_name)

    def setDecoder(self,decoder):
        self.decoder = decoder
        self.reset()

    def setCursorPos(self,pos):
        self.pos_cursor_i = pos

    def reset(self,pos=np.zeros(2), vel=np.zeros(2)):
        self.decoder.reset()

        self.pos_cursor_i = pos
        self.Hspikes = np.zeros([4,192])

        if self.decoder_type in ['FIT','ReFIT','VKF']:
            self.alpha = 1.0
        elif self.decoder_type in ['PVKF']:
            self.alpha = 0.0
        elif self.decoder_type in ['FORCE']:
            self.alpha = 0.95
        
        if self.decoder_type =='NDF':
            self.decoder.setState(np.zeros((20,1)))
        elif self.decoder_type in ['FIT','ReFIT','VKF','PVKF']:
            self.decoder.setState(np.hstack([pos,vel,1]))
        elif self.decoder_type in ['FORCE']:
            pass
 

    def loadDecoder(self):
        decoder_path = self.decoder_path
        refresh_rate = self.refresh_rate
        decoder_type = self.decoder_type
        decoder_name = self.decoder_name

        if self.b_bypass:
            print("In bypass mode, no need to load decoder")
            return None

        if decoder_type in ['FIT','ReFIT','VKF','PVKF']:
            decoder = KFDecoder(type=decoder_type)
        elif decoder_type == 'FORCE':
            decoder = FORCEDecoder(dt=refresh_rate)
        elif decoder_type == 'WF':
            raise NotImplementedError
        elif decoder_type == 'NDF':
            raise NotImplementedError
        else:
            raise ValueError("decoder_type {} is not valid".format(decoder_type))

        model_path = os.path.join(decoder_path,decoder_name)

        print("Load decoder model from {}".format(model_path))
        decoder.load(model_path)

        self.setDecoder(decoder)

    def decode(self,spikes):
        refresh_rate = self.refresh_rate
        decoder_type = self.decoder_type
        alpha = self.alpha
        Hspikes = self.Hspikes
        pos_cursor_i = self.pos_cursor_i
        #numChannels = self.numChannels
        screenW = self.screenW
        screenH = self.screenH

        spikes = spikes.reshape(1,-1)
        Hspikes = np.vstack((Hspikes,spikes))

        if decoder_type in ['FIT','ReFIT','VKF','PVKF']:
            #decodeState = self.decoder.decode(spikes.T[:numChannels])
            decodeState = self.decoder.decode(spikes.T)
            _P = decodeState[0:2]
            _V = decodeState[2:4].reshape(2)*1000
        elif decoder_type == 'FORCE':
            #decodeState = self.decoder.decode_fast(spikes)
            decodeState = self.decoder.decode(spikes)
            _P = decodeState[0:2]
            _V = decodeState[2:4]
        elif decoder_type == 'WF':
            decodePVs = self.decoder.decode(Hspikes[-4:]).T
            _P = decodePVs[0:2]
            _V = decodePVs[2:4]
        elif decoder_type == 'NDF':
            decodeState = self.decoder.decode(spikes.T)
            decodeState = np.vstack((decodeState,np.ones((1,1))))
            decodePVs = self.decoder.Lw @ decodeState
            _P = decodePVs[0:2]
            _V = decodePVs[2:4]*1000

        else:
            raise ValueError("decoder_type is not valid")

        _P = np.squeeze(_P)
        _V = np.squeeze(_V)
        pos_cursor_i=(1-alpha)*_P + (alpha)*(pos_cursor_i+_V*refresh_rate/1000)

        #bounding box
        pos_cursor_i = (np.clip(pos_cursor_i[0],-screenW>>1,screenW>>1), np.clip(pos_cursor_i[1],-screenH>>1,screenH>>1))

        decoded_axis_pXY=_P
        decoded_axis_vXY=_V

        self.Hspikes = Hspikes[-100:]
        self.pos_cursor_i = pos_cursor_i

        return pos_cursor_i, decoded_axis_pXY, decoded_axis_vXY
