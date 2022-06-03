import os, subprocess, time, signal
import io
import base64
import gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from gym import error, spaces
from gym import utils
from gym.utils import seeding
from gym import wrappers, spaces
#from stable_baselines.common.vec_env import *
#from stable_baselines.common.cmd_util import *
from stable_baselines.common.base_class import BaseRLModel
import logging
from IPython.display import HTML


#GitHubPath = '/home3/kenfuliang/GitHub/'
GitHubPath = '../../'

import sys
sys.path.append(GitHubPath+"BCI_simulator/code/")
from NeuralEncoder import NeuralEncoder
from NeuralDecoder import NeuralDecoder
from tasks import CenterOutTask, PinballTask
import scipy.io as sio
import tensorflow as tf
import keras
from keras.backend import exp

logger = logging.getLogger(__name__)

def get_exp_setting():
    exp_setting = {}
    exp_setting['b_bypass'] = True
    exp_setting['b_synthesize'] = False 
    exp_setting['encoder_refresh_rate'] = 50
    exp_setting['decoder_refresh_rate'] = 50
    ## encoder
    exp_setting['encoder_type'] = 'RNN'
    exp_setting['encoder_input_shape'] = 'pv'
    exp_setting['encoder_path'] = GitHubPath+'/BCI_simulator/pretrained/encoder/'
    exp_setting['encoder_name'] = 'Seq2Seq_stateful_b25_20220105_r192_decay0.01_ke1.0_re0_delayed88_extraBins0_0512_ep4.h5'
    exp_setting['M1_delay'] = 8 
    exp_setting['PMd_delay'] = 8

    ## decoder
    exp_setting['decoder_type'] = None
    exp_setting['decoder_name'] = None
    exp_setting['decoder_path'] = None
    return exp_setting

def get_VKF_env_kwargs():
    env_kwargs = dict(input_mode='acc'
                  ,n_stack=1
                  ,n_eval_episodes=500
                  ,target_radius=80
                  ,acceptance_window = 40
                  ,target_type='square'
                  ,task_type='centerout'
                  ,noise_alpha=2                  
                  ,target_delay=0
                  ,time_hold=500
                  ,action_high=100
                  ,action_low=-100
                  ,reward_setting='naive'
                  ,obs_setting='pvtpv'
                  ,exp_setting=None)
    return env_kwargs
def get_FIT_env_kwargs():
    return get_VKF_env_kwargs()
def get_ReFIT_env_kwargs():
    return get_VKF_env_kwargs()
def get_PVKF_env_kwargs():
    env_kwargs = get_VKF_env_kwargs()
    env_kwargs['target_radius']=120
    env_kwargs['acceptance_window']=60
    return env_kwargs
def get_FORCE_env_kwargs():
    env_kwargs = get_VKF_env_kwargs()
    return env_kwargs
def get_pinball_env_kwargs(dt=50):
    env_kwargs = get_VKF_env_kwargs()
    env_kwargs['task_type']='pinball'
    env_kwargs['target_delay']=200//dt
    return env_kwargs
def get_hand_env_kwargs(dt=50,target_radius=120):
    env_kwargs = get_VKF_env_kwargs()
    env_kwargs['target_delay']=200//dt
    env_kwargs['target_radius']=target_radius
    env_kwargs['noise_alpha']=0
#    env_kwargs['dt']=dt
    return env_kwargs 

class NeuralEncoder_RL(NeuralEncoder):
    def __init__(self,exp_setting):
        #print("####Start Encoder Construction####")
        super().__init__()
        self.encoder_type = exp_setting['encoder_type']
        self.encoder_input_shape = exp_setting['encoder_input_shape'] 
        self.b_bypass = exp_setting['b_bypass']
        self.b_synthesize = exp_setting['b_synthesize']
        self.refresh_rate = exp_setting['encoder_refresh_rate']
        self.encoder_path = exp_setting['encoder_path']
        self.encoder_name = exp_setting['encoder_name']
        self.M1_delay = exp_setting['M1_delay']
        self.PMd_delay = exp_setting['PMd_delay']
        self.loadEncoder()


class NeuralDecoder_RL(NeuralDecoder):
    def __init__(self,exp_setting):
        #print("####Start Decoder Construction####")
        super().__init__()
        self.b_bypass = exp_setting['b_bypass']
        self.decoder_type = exp_setting['decoder_type']
        self.decoder_name = exp_setting['decoder_name']
        self.decoder_path = exp_setting['decoder_path']
        self.refresh_rate = exp_setting['decoder_refresh_rate']
       
        self.loadDecoder()

class CenteroutEnv(gym.Env):
    def __init__(self
                 ,verbose=False
                # shared 
                 ,exp_setting= None
                # task setting
                 ,task_type = 'centerout'
                 ,target_type='square'
                 ,target_radius=80
                 ,acceptance_window=50
                 ,time_hold = 500
                 ,b_bounded_cursor = False
                 ,target_list=[]

                # RL setting
                 ,n_eval_episodes=10
                 ,input_mode='vel' #{pos,vel,acc}
                 #,cursor_size=20
                 ,obs_history = 1
                 ,target_history = 1
                 ,obs_delay = 0
                 ,obs_hand_delay = 0
                 ,target_delay = 0
                 ,obs_setting='pvt'  # {pvt, pvtpv}
                 ,act_setting='h' #{'h': hand, 'full': hand_pos, cursor_pos, target_pos, cursor_color, target_color}
                 ,render_setting='c' #{'c': cursor, 'hc': hand and cursor}

                 ,action_high = 100
                 ,action_low = -100
                 ,action_gain = 1


                 ,noise_alpha= 0 # constant noise
                 ,noise_beta = 0 # input-dependent noise

                # reward setting
                 ,reward_delay = 0
                 ,reward_setting='naive' # {guided,naive}
                 ,acc_cost = 0
                 ,b_energy_cost=False
                 #,neural_decoder = None
                 ,**kwargs
                    ):

        if verbose: print("Version:",str(1.3))
        self.graph = None 
        self.sess = None 
##########################################################################

        self.exp_setting = exp_setting = exp_setting if exp_setting!=None else get_exp_setting()
##########################################################################
     
        self.dt = exp_setting['encoder_refresh_rate']
        self.input_mode = input_mode
        self.metadata = {
            'render.modes': ['human', 'rgb_array'],
            'video.frames_per_second' : 1000//self.dt
            }
        self.reward_setting = reward_setting
        self.render_setting = render_setting
        self.acc_cost = acc_cost
        self.act_setting = act_setting
        self.obs_setting = obs_setting
        self.action_gain = action_gain
        self.action_high = action_high
        self.action_low = action_low
###########################################################################
        ## BCI
        self.b_bypass = exp_setting['b_bypass']
        self.b_bounded_cursor = b_bounded_cursor

        ## reward shaping
        self.b_energy_cost = b_energy_cost
        self.dist_factor = kwargs.get('dist_factor',1/80)

        ## target
        self.target_type = target_type
        self.noise_alpha = noise_alpha
        self.noise_beta = noise_beta
########################tmp################################################
        self.n_eval_episodes = n_eval_episodes
############ EXP SETTING ##################################################
        if task_type=='centerout':
            self.task = CenterOutTask(dt=self.dt,radius=target_radius,acceptance_window=acceptance_window,target_list=target_list,time_hold=time_hold)
        elif task_type=='pinball':
            self.task = PinballTask(dt=self.dt,acceptance_window=acceptance_window)
##############FIXED SETTING ##################################################
        self.cursor_size = 20
        self.acceptance_window = acceptance_window
        self.screen_width = 1920/3.1125
        self.screen_height = 1080/3.1125

############ EXP STATUS (CHANGING THROUGH TIME) ############
        self.viewer = None
############## ACTION SPACE################################
        if self.act_setting=='h':
            self.action_space = spaces.Box(low=action_low, high=action_high, shape=(2,), dtype=np.float32)
        elif self.act_setting=='hc':
            self.action_space = spaces.Box(low=action_low, high=action_high, shape=(4,), dtype=np.float32)
############## OBSERVATION SPACE################################
        if (self.obs_setting=='pvt'): self.observation_space = spaces.Box(low=-1000, high=1000, shape=(4*obs_history+2*target_history,1), dtype=np.float32) 
        elif (self.obs_setting=='pt'): self.observation_space = spaces.Box(low=-1000, high=1000, shape=(2*obs_history+2*target_history,1), dtype=np.float32) 
        elif (self.obs_setting=='pvtpv'): self.observation_space = spaces.Box(low=-1000, high=1000, shape=(8*obs_history+2*target_history,1), dtype=np.float32) 
        elif (self.obs_setting=='pvat'): self.observation_space = spaces.Box(low=-1000, high=1000, shape=(6*obs_history+2*target_history,1), dtype=np.float32) 
        else: raise NotImplementedError("wrong obs_setting:{}".format(self.obs_setting))

        self.obs_history = obs_history
        self.target_history = target_history

        self.obs_delay = obs_delay
        self.obs_hand_delay = obs_hand_delay
        self.target_delay = target_delay
        self.reward_delay = reward_delay


############## Queue of hand kinemaitcs to support different input mode ################################
        self.step_counter   = None
        self.Q_targetPos    = None 
        self.Q_handPos      = None 
        self.Q_handVel      = None 
        self.Q_handAcc      = None 
        self.Q_cursorPos    = None 
        self.Q_cursorVel    = None 
        self.Q_cursorAcc    = None 
        self.Q_reward       = None 
        self.Q_binSpike     = None 

########################################################        

        self.graph = tf.Graph()
        self.sess = tf.Session()
        with self.graph.as_default():
            self.sess = tf.Session()
            with self.sess.as_default():
                self.neural_encoder = NeuralEncoder_RL(self.exp_setting)
                self.neural_decoder = NeuralDecoder_RL(self.exp_setting)

        if verbose:
            self.summary()

    def summary(self):
        print("#### Summary of RL environment ####")
        print("     dt",self.dt)
        print("     input_mode", self.input_mode)
        print("     b_bypass", self.b_bypass)
        print("     target_type", self.target_type)
        print("     noise_beta", self.noise_beta)
        print("     noise_alpha", self.noise_alpha)
        print("     obs_delay", self.obs_delay)
        print("     obs_hand_delay", self.obs_hand_delay)
        print("     target_delay", self.target_delay)
        self.neural_encoder.summary()
        self.neural_decoder.summary()
        self.task.summary()
    

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def updateHandPosition(self,action):
        if self.input_mode=='pos':
            hand_position = action
            hand_velocity = (hand_position - self.Q_handPos[-1])
            hand_acc      = (hand_velocity - self.Q_handVel[-1])

        elif self.input_mode=='vel': # action is velocity (mm/dt)
            hand_velocity = action #(mm/dt)
            hand_position = hand_velocity + self.Q_handPos[-1] 
            hand_acc      = hand_velocity - self.Q_handVel[-1] #(mm)/dt^2 

        elif self.input_mode=='acc': # action is acceleration (mm/dt^2)
            hand_acc      = action
            hand_velocity = hand_acc        + self.Q_handVel[-1] 
            hand_position = hand_velocity   + self.Q_handPos[-1] 

        elif self.input_mode=='jerk': # action is jerk
            raise NotImplementedError("input_mode is not correct!")
            #hand_jerk     = action
            #hand_acc      = hand_jerk   + self.Q_handAcc[-1]
            #hand_velocity = hand_jerk/2 + self.Q_handAcc[-1]   + self.Q_handVel[-1] 
            #hand_position = hand_jerk/6 + self.Q_handAcc[-1]/2 + self.Q_handVel[-1] + self.Q_handPos[-1]
        else:
            raise NotImplementedError("input_mode is not correct!")

        pseudo_spike_counts, pseudo_spike_rates= self.neural_encoder.encode(hand_position,self.Q_targetPos[-1],self.Q_cursorPos[-1])
        self.Q_binSpike = np.vstack((self.Q_binSpike[1:],pseudo_spike_counts))


        self.Q_handPos = np.vstack((self.Q_handPos[1:],hand_position))
        self.Q_handVel = np.vstack((self.Q_handVel[1:],hand_velocity))
        self.Q_handAcc = np.vstack((self.Q_handAcc[1:],hand_acc))

        return hand_position


    def updateCursorPosition(self,hand_position):
        if self.b_bypass == True:
            cursor_position = hand_position
        else:
            h = int(self.neural_decoder.refresh_rate//self.neural_encoder.refresh_rate)
            pseudo_spike_counts = np.sum(self.Q_binSpike[-h:],axis=0)
            assert pseudo_spike_counts.shape==(192,)
            cursor_position,_,_ = self.neural_decoder.decode(pseudo_spike_counts)
            cursor_position = np.array(cursor_position)# turn tuple into array

        ## constraint cursor position
        if self.b_bounded_cursor:
            cursor_position[0] = min(max(-self.screen_width//2,cursor_position[0]),self.screen_width//2)
            cursor_position[1] = min(max(-self.screen_height//2,cursor_position[1]),self.screen_height//2)
                          
        cursor_velocity = cursor_position - self.Q_cursorPos[-1] 
        cursor_acc = cursor_velocity - self.Q_cursorVel[-1]

        self.Q_cursorPos = np.vstack((self.Q_cursorPos[1:],cursor_position))
        self.Q_cursorVel = np.vstack((self.Q_cursorVel[1:],cursor_velocity))
        self.Q_cursorAcc = np.vstack((self.Q_cursorAcc[1:],cursor_acc))

    def updateHandAndCursor(self,action):
        with self.graph.as_default():
            with self.sess.as_default(): 
                self.step_counter+=self.dt

                if self.step_counter%self.neural_encoder.refresh_rate==0:
                    ## Update Hand position
                    hand_position = self.updateHandPosition(action)
                
                if self.step_counter%self.neural_decoder.refresh_rate==0:
                    ## Update Cursor position by encoding then decoding
                    self.updateCursorPosition(hand_position)
                 
    
    def step(self, action):
        if self.act_setting=='h':
            ## preprocess action input
            action=np.squeeze(action)*self.action_gain
            action = action*(1+np.random.normal(0,self.noise_beta,2)) + np.random.normal(0,self.noise_alpha,2)
 
            ## update cursor and hand position
            self.updateHandAndCursor(action)
            ## update task state
            self.task.update(self.Q_cursorPos[-1],PV_fail=False)
        elif self.act_setting=='full':
            action = np.squeeze(action)
            hand_position = action[0:2]
            cursor_position = action[2:4]
            target_position = action[4:6]
            cursor_color = action[6:9]
            target_color = action[9:12]
            self.Q_handPos = np.vstack((self.Q_handPos[1:],hand_position))
            self.Q_cursorPos = np.vstack((self.Q_cursorPos[1:],cursor_position))
            self.task.force(cursor_position, target_position, cursor_color, target_color)

        self.Q_targetPos = np.vstack((self.Q_targetPos[1:],self.task.pos_target))

        ## update info
        info = {'task_state':self.task.task_state,
                'hand_position':self.Q_handPos[-1],
                'cursor_position':self.Q_cursorPos[-1],
                'target_position':self.Q_targetPos[-1],
                'task_states':self.task.task_states,
                'binSpike':self.Q_binSpike[-1],
                }

        assert (self.task.pos_cursor == self.Q_cursorPos[-1]).all(),"{} v.s. {}".format(self.task.pos_cursor, self.Q_cursorPos[-1])
        assert (self.task.pos_target == self.Q_targetPos[-1]).all()
    
        ## update observation
        observation = self.observation_function()
        ## calculate reward
        reward = self.reward_function()
        ## done or not
        done = True if (self.n_eval_episodes == self.task.trial_count) else False
                       
        return observation, reward, done, info
    
    def reset(self):        
        self.task.reset()

        self.step_counter = 0
        self.Q_targetPos = np.zeros((10,2))
        self.Q_handPos = np.zeros((20,2))
        self.Q_handVel = np.zeros((20,2))
        self.Q_handAcc = np.zeros((20,2))
        self.Q_cursorPos = np.zeros((20,2))
        self.Q_cursorVel = np.zeros((20,2))
        self.Q_cursorAcc = np.zeros((20,2))
        self.Q_reward = np.zeros(20)
        self.Q_binSpike = np.zeros((20,192))
        observation = self.observation_function()

        return observation

    def render(self, mode='human'):
        screen_width = self.screen_width*3.1125
        screen_height = self.screen_height*3.1125
        target_size = self.acceptance_window*3.1125
        cursor_size = self.cursor_size
        target_position = self.task.pos_target*3.1125
        cursor_position = self.task.pos_cursor*3.1125
        hand_position = self.Q_handPos[-1]*3.1125

        center = np.array([screen_width/2,screen_height/2],dtype=np.float32)
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width,screen_height)
            ## Background
            l,r,t,b = 0, screen_width, 0, screen_height
            self.background = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            self.background.set_color(self.task.black[0],self.task.black[1],self.task.black[2])
            self.viewer.add_geom(self.background)
            ## target
            if self.target_type == 'square':
                ## square target
                l,r,t,b = -target_size/2, target_size/2, -target_size/2, target_size/2
                self.target = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            elif self.target_type == 'circle':
                ## circular target
                self.target = rendering.make_circle(target_size/2)

            self.target.set_color(self.task.target_color[0],self.task.target_color[1],self.task.target_color[2])
            target_offset = (screen_width/2, screen_height/2)
            self.targettrans = rendering.Transform(translation=target_offset)
            self.target.add_attr(self.targettrans)
            self.viewer.add_geom(self.target)
            
            if 'h' in self.render_setting:
                ## hand
                self.hand = rendering.make_circle(cursor_size)
                self.hand.set_color(self.task.blue[0],self.task.blue[1],self.task.blue[2])
                hand_offset = (screen_width/2, screen_height/2)
                self.handtrans = rendering.Transform(translation=hand_offset)
                self.hand.add_attr(self.handtrans)
                self.viewer.add_geom(self.hand)
                
            if 'c' in self.render_setting:
                ## cursor
                self.cursor = rendering.make_circle(cursor_size)
                self.cursor.set_color(self.task.cursor_color[0],self.task.cursor_color[1],self.task.cursor_color[2])
                cursor_offset = (screen_width/2, screen_height/2)
                self.cursortrans = rendering.Transform(translation=cursor_offset)
                self.cursor.add_attr(self.cursortrans)
                self.viewer.add_geom(self.cursor)
                
        new_target_position = (target_position+center)
        self.targettrans.set_translation(newx= new_target_position[0], newy=new_target_position[1])
        self.target.set_color(self.task.target_color[0],self.task.target_color[1],self.task.target_color[2])

        if 'h' in self.render_setting:
            new_hand_position   = (hand_position+center)
            self.handtrans.set_translation(newx= new_hand_position[0], newy=new_hand_position[1])

        if 'c' in self.render_setting:
            self.cursor.set_color(self.task.cursor_color[0],self.task.cursor_color[1],self.task.cursor_color[2])
            new_cursor_position = (cursor_position+center)
            self.cursortrans.set_translation(newx= new_cursor_position[0], newy=new_cursor_position[1])

        return self.viewer.render(return_rgb_array = mode=='rgb_array')     

    def close (self):
        if self.viewer:
            self.viewer.close()
            self.viewer=None
##################### ##################### ##################### ##################### ##################### 
    def observation_function(self):
        obs_setting = self.obs_setting
    
        if obs_setting =='pvtpv':
            pvt = np.concatenate([self.Q_cursorPos[-self.obs_delay-self.obs_history:len(self.Q_cursorPos)-self.obs_delay],self.Q_cursorVel[-self.obs_delay-self.obs_history:len(self.Q_cursorVel)-self.obs_delay],self.Q_targetPos[-self.target_delay-self.target_history:len(self.Q_targetPos)-self.target_delay]]).reshape((-1,1))
            pv  = np.concatenate([self.Q_handPos[-self.obs_hand_delay-self.obs_history:len(self.Q_handPos)-self.obs_hand_delay],self.Q_handVel[-self.obs_hand_delay-self.obs_history:len(self.Q_handPos)-self.obs_hand_delay]]).reshape((-1,1))
            pvtpv = np.concatenate([pvt,pv])
            assert pvtpv.shape[0]==(8*self.obs_history+2*self.target_history)
            return pvtpv

        elif obs_setting=='pvt':
            pvt = np.concatenate([self.Q_cursorPos[-self.obs_delay-self.obs_history:],self.Q_cursorVel[-self.obs_delay-self.obs_history:],self.Q_targetPos[-self.target_delay-self.target_history:len(self.Q_targetPos)-self.target_delay]]).reshape((-1,1))
            assert pvt.shape[0]==(4*self.obs_history+2*self.target_history)
            return pvt

        elif obs_setting=='pvat':
            pva = np.concatenate([self.Q_cursorPos[-self.obs_delay-self.obs_history:],self.Q_cursorVel[-self.obs_delay-self.obs_history:],self.Q_cursorAcc[-self.obs_delay-self.obs_history:]]).reshape((-1,1))
            t = self.Q_targetPos[-self.target_delay-self.target_history:len(self.Q_targetPos)-self.target_delay].reshape(-1,1)
            pvat = np.concatenate([pva,t],axis=0)
            return pvat

       
    def reward_function(self):#,vel,acc,jerk
        task_state = self.task.task_state
        task_states = self.task.task_states
        target_position = self.task.pos_target
        cursor_position = self.task.pos_cursor
        ## NAIVE REWARD(all positive) ##
        if self.reward_setting =='naive':
            if task_state==task_states['fail']:
                reward = 0
            elif task_state==task_states['success']:
                reward = 1
            else:
                reward = 0
        ## NAIVE REWARD(negative if fail) ##
        elif self.reward_setting =='naive_neg': 
            if task_state==task_states['fail']:
                reward = -2
            elif task_state==task_states['success']:
                reward = 1
            else:
                reward = 0

        ## NAIVE REWARD(negative if fail) ##
        elif self.reward_setting =='naive_stay': 
            if task_state==task_states['fail']:
                reward = 0
            elif task_state==task_states['success']:
                reward = 50
            elif task_state==task_states['hold']:
                reward = 1
            else:
                reward = 0

        elif self.reward_setting =='neg_stay': 
            if task_state==task_states['hold']:
                reward = 0
            else:
                reward = -1
             
        ## Guided Reward (negative if far from target) ##
        elif self.reward_setting=='guided':
            if task_state==task_states['success']:
                reward = 100
            elif task_state==task_states['fail']:
                reward = -200
            elif task_state==task_states['hold']:
                reward = 0
            elif task_state==task_states['end']:
                reward = 0
            else:
                reward = -np.linalg.norm(target_position-cursor_position)*self.dist_factor
                #reward -= np.linalg.norm(jerk)*self.jerk_factor
                #reward -= bool(self.counter_duration <= 250)*np.linalg.norm(vel)*self.vel_factor

        ## Guided Reward (negative if far from target) ##
        elif self.reward_setting=='negguided':
            if task_state==task_states['success']:
                reward = 0
            elif task_state==task_states['fail']:
                reward = 0
            #elif task_state==task_states['hold']:
            #    reward = 0
            elif task_state==task_states['end']:
                reward = 0
            else:
                reward = -np.linalg.norm(target_position-cursor_position)*self.dist_factor

        ## Guided Reward (negative if far from target) ##
        elif self.reward_setting=='test':
            if task_state==task_states['success']:
                reward = 1
            elif task_state==task_states['fail']:
                reward = 0
            elif task_state==task_states['hold']:
                reward = 0
            elif task_state==task_states['end']:
                reward = 0
            else:
                reward = -np.linalg.norm(target_position-cursor_position)*self.dist_factor
            acc = self.Q_handAcc[-1]
            reward = reward - np.linalg.norm(acc)*self.acc_cost
        else:
            assert False,"No this settig ({}) for reward_setting".format(self.reward_setting)

        self.Q_reward = np.hstack((self.Q_reward[1:], reward))

        return self.Q_reward[-1-self.reward_delay]
