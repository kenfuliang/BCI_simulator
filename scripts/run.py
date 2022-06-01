import sys
import wandb
import matplotlib.pyplot as plt
import os
from argparse import ArgumentParser
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.algos import PPOCAPSZ
from stable_baselines.common.cmd_util import make_vec_env
from gym_centerout.envs.centerout_env import CenteroutEnv, get_exp_setting,get_hand_env_kwargs,get_FIT_env_kwargs,get_ReFIT_env_kwargs,get_FORCE_env_kwargs
from Dataset import RLDataset
from decoder.kalman import KFDecoder
from decoder.FORCE import FORCEDecoder

class AgentEncoderDecoder():
    def __init__(self,wandb,**kwargs):
        self.wandb = wandb
        print(kwargs)
        for key,value in kwargs.items():
            setattr(self, key, value)

    def sim(self):
        if args['decoder']=='hand':
            self.RunHand(task='centerout',b_retrain=(self.pretrained_mode==0))
        elif args['decoder']=='FIT':
            self.RunFIT()
        elif args['decoder']=='ReFIT':
            self.RunReFIT()
        elif args['decoder']=='VKF':
            self.RunVKF()
        elif args['decoder']=='PVKF':
            self.RunPVKF()
        elif args['decoder']=='FORCE':
            self.RunFORCE()
        else:
            raise ValueError("``{}'' decoder is not supported".format(args['decoder']))

    def Curriculum_learning(self, env_kwargs):
        agent = self.agent
        wandb = self.wandb
        learning_rate = self.learning_rate
        learning_epochs = self.learning_epochs
        target_radius = self.target_radius
        acceptance_window = self.acceptance_window
        min_acceptance_window = self.min_acceptance_window
        max_acceptance_window = self.max_acceptance_window
        smooth_coef = self.smooth_coef
        zero_coef = self.zero_coef


        ## Create env for evaluation 
        env_kwargs['target_radius']=target_radius
        env_kwargs['acceptance_window']=acceptance_window
        env_kwargs['n_eval_episodes']=100
        env_eval = make_vec_env(CenteroutEnv,env_kwargs=env_kwargs,n_envs=1)
        print("Evaluate env:",env_kwargs)
    
        ## Create env for training
        env_kwargs['target_radius']=target_radius
        env_kwargs['acceptance_window']=max_acceptance_window
        env_kwargs['n_eval_episodes']=100
        env_train = make_vec_env(CenteroutEnv,env_kwargs=env_kwargs,n_envs=16,seed=0)
        print("Curriculum learning starting parameters:",env_kwargs)
        agent.set_env(env_train)

        print(env_kwargs['exp_setting'])


        b_constrained_PPO = False
    
        for epoch in range(0,learning_epochs):
            ## learning
            agent.learn(100000,reset_num_timesteps=False,log_interval=100)
    
            if (epoch+1)%1==0:
                ## check performance and adjust difficulty 
                dataset = RLDataset(agent,env_train,deterministic=False,M1_delay=0,PMd_delay=0)
                if dataset.Statistics().iloc[0]['success_rate']>0.9:
                    acceptance_window = env_kwargs['acceptance_window']*0.9

                else:
                    acceptance_window = env_kwargs['acceptance_window']*1.1
                acceptance_window = min(max(min_acceptance_window,acceptance_window),max_acceptance_window)
                if env_kwargs['acceptance_window']!=acceptance_window:
                    print("New acceptance window : {}".format(acceptance_window))
                    env_kwargs['acceptance_window'] = acceptance_window
                    if 'env_train' in locals():
                        env_train.close()
                    env_train = make_vec_env(CenteroutEnv,env_kwargs=env_kwargs,n_envs=16,seed=0)
                    agent.set_env(env_train)
    
                ## eval performance and save stats
                dataset = RLDataset(agent,env_eval,deterministic=False,M1_delay=0,PMd_delay=0)
                stats = dataset.Statistics().iloc[2][1:]
                if wandb!=None:
                    dataset.PlotAll()
                    fig_all = plt.gcf()
                    dataset.PlotPSTH()
                    fig_PSTH = plt.gcf()
                    decoder_type = env_kwargs['exp_setting']['decoder_type']
                    wandb.log({**dict(stats),
                               "{}_all".format(decoder_type):fig_all,
                               "{}_PSTH".format(decoder_type):fig_PSTH,
                               "epoch":epoch+1})

    
                agent.save(os.path.join(wandb.run.dir,"{}_agent_ep{}.zip".format(decoder_type,epoch+1)))

                ## Once the agent achive good performance, start applying KL constraints. 
                #if (b_constrained_PPO==False) and (dataset.Statistics().iloc[0]['success_rate']>0.9):
                #    agent = PPOCAPSZ(MlpPolicy, env_train
                #                    ,learning_rate= learning_rate
                #                    ,smooth_coef = smooth_coef
                #                    ,zero_coef = zero_coef
                #                    )
                #    agent.load_parameters(os.path.join(wandb.run.dir,"{}_agent_ep{}.zip".format(decoder_type,epoch+1)))
                #    print("Set agent's constraints: smooth_coef:{}, zero_coef:{}".format(agent.smooth_coef, agent.zero_coef))
                #    b_constrained_PPO=True

        self.agent=agent

    def RunHand(self,task=None,b_retrain=False):
        encoder = self.encoder
        encoder_dt = self.encoder_dt
        M1_delay=self.M1_delay
        PMd_delay=self.PMd_delay
        noise_alpha = self.noise_alpha
        learning_epochs=self.learning_epochs
        smooth_coef = self.smooth_coef
        zero_coef = self.zero_coef
        min_acceptance_window = self.min_acceptance_window
        wandb = self.wandb

        print("RunHand")

        ## Train agent
        if b_retrain:
            exp_setting = get_exp_setting()
            exp_setting['b_bypass']=True
            exp_setting['b_synthesize']=False
            exp_setting['encoder_name']=encoder
            exp_setting['M1_delay']=M1_delay
            exp_setting['PMd_delay']=PMd_delay
            exp_setting['decoder_type']='hand'
            exp_setting['encoder_refresh_rate']=encoder_dt
            exp_setting['decoder_refresh_rate']=encoder_dt
            env_kwargs = get_hand_env_kwargs(dt=encoder_dt)
            env_kwargs['noise_alpha'] = noise_alpha
            env_kwargs['exp_setting'] = exp_setting

            env_train = make_vec_env(CenteroutEnv,env_kwargs=env_kwargs,n_envs=16,seed=0)

            hand_agent = PPOCAPSZ(MlpPolicy, env_train
                                    ,learning_rate=self.learning_rate
                                    ,smooth_coef = smooth_coef
                                    ,zero_coef = zero_coef
                                    )

            pretrained_agent = "../pretrained/agents/naive_PPO/hand_agent.zip"
            hand_agent.load_parameters(pretrained_agent)
            print("Agent is initialized with {}".format(pretrained_agent))
            self.agent = hand_agent
            self.Curriculum_learning(env_kwargs)

        ## Run agent
        else:
            pretrained_agent = "../pretrained/agents/constrained_PPO/hand_agent.zip"
            print("Hand agent: {}".format(pretrained_agent))
            hand_agent = PPOCAPSZ.load(pretrained_agent)
            self.agent = hand_agent

            ## Evaluate agent
            exp_setting = get_exp_setting()
            exp_setting['b_bypass']=True
            exp_setting['b_synthesize']=True
            exp_setting['encoder_name']=encoder
            exp_setting['M1_delay']=M1_delay
            exp_setting['PMd_delay']=PMd_delay
            exp_setting['encoder_refresh_rate']=encoder_dt
            exp_setting['decoder_refresh_rate']=encoder_dt
            env_kwargs = get_hand_env_kwargs(dt=encoder_dt)
            env_kwargs['noise_alpha'] = noise_alpha
            env_kwargs['exp_setting'] = exp_setting
            env_kwargs['n_eval_episodes']=100
            if task=='centerout':
                env_kwargs['task_type']='centerout'
                env_kwargs['target_radius']=120
            elif task=='pinball':
                env_kwargs['task_type']='pinball'
 
            data_hand = RLDataset(self.agent,{**env_kwargs,'verbose':True},deterministic=False,M1_delay=M1_delay,PMd_delay=PMd_delay)

            if (wandb!=None):
                data_hand.PlotAll()
                fig_all = plt.gcf()
                data_hand.PlotPSTH()
                fig_PSTH = plt.gcf()
                stats = data_hand.Statistics().iloc[2][1:]
                wandb.log({**dict(stats),
                           "Hand_all":fig_all,
                           "Hand_PSTH":fig_PSTH,
                           })
            return data_hand
    
    def RunVKF(self):
        encoder = self.encoder
        encoder_dt = self.encoder_dt
        decoder_dt = self.decoder_dt
        M1_delay=self.M1_delay
        PMd_delay=self.PMd_delay
        obs_delay=self.obs_delay//encoder_dt
        obs_hand_delay=self.obs_hand_delay//encoder_dt
        noise_alpha = self.noise_alpha
        num_channel=self.num_channel
        learning_epochs=self.learning_epochs
        wandb = self.wandb
        smooth_coef = self.smooth_coef
        zero_coef = self.zero_coef
        min_acceptance_window = self.min_acceptance_window
        max_acceptance_window = self.max_acceptance_window
        acceptance_window = self.acceptance_window

        VKF = KFDecoder(type='VKF',num_channel=num_channel,name='VKF.npz')
        if self.retrain_decoder:
            ## Run hand
            data_hand = self.RunHand(task='centerout',b_retrain=False)
            data_hand.downSample(int(self.decoder_dt//self.encoder_dt))
            
            ## Train VKF
            VKF.run(data_hand)
        else:
            decoder_path = "../pretrained/decoders/VKF.npz"
            VKF.load(decoder_path)


        VKF.save(wandb.run.dir)

        ## evaluate VKF
        exp_setting = get_exp_setting()
        exp_setting['b_bypass']=False
        exp_setting['b_synthesize']=True
        exp_setting['encoder_name']=encoder
        exp_setting['M1_delay']=M1_delay
        exp_setting['PMd_delay']=PMd_delay
        exp_setting['decoder_type']='VKF'
        exp_setting['decoder_name']='VKF.npz'
        exp_setting['decoder_path']=wandb.run.dir
        exp_setting['encoder_refresh_rate']=encoder_dt
        exp_setting['decoder_refresh_rate']=decoder_dt
        env_kwargs = get_VKF_env_kwargs()
        env_kwargs['obs_delay']=obs_delay
        env_kwargs['obs_hand_delay']=obs_hand_delay
        env_kwargs['noise_alpha'] = noise_alpha
        env_kwargs['exp_setting']=exp_setting

        env_kwargs['acceptance_window']=acceptance_window
        env_kwargs['n_eval_episodes']=100
        env_train = make_vec_env(CenteroutEnv,env_kwargs=env_kwargs,n_envs=16,seed=0)

        VKF_agent = PPOCAPSZ(MlpPolicy,env_train
                             ,learning_rate=self.learning_rate
                             ,smooth_coef=smooth_coef
                             ,zero_coef=zero_coef)

        if self.pretrained_mode==0:
            print("Agent is random initialized")
        elif self.pretrained_mode==1:
            pretrained_agent = "../pretrained/agents/constrained_PPO/VKF_agent.zip"
            VKF_agent.load_parameters(pretrained_agent)
        elif self.pretrained_mode==2:
            pretrained_agent = "../pretrained/agents/naive_PPO/VKF_agent.zip"
            VKF_agent.load_parameters(pretrained_agent)
        elif self.pretrained_mode==3:
            pretrained_agent = "../pretrained/agents/constrained_PPO/hand_agent.zip"
            VKF_agent.load_parameters(pretrained_agent)

        self.agent = VKF_agent

        assert VKF_agent.smooth_coef == smooth_coef
        assert VKF_agent.zero_coef == zero_coef
    
        if wandb!=None:
            VKF_agent.save(os.path.join(wandb.run.dir,"VKF_agent_ep{}.zip".format(0)))
            data_VKF = RLDataset(self.agent,{**env_kwargs,'verbose':True},deterministic=False,M1_delay=0,PMd_delay=0)
            data_VKF.PlotAll()
            fig_all = plt.gcf()
            data_VKF.PlotPSTH()
            fig_PSTH = plt.gcf()
            stats = data_VKF.Statistics().iloc[2][1:]
            wandb.log({**dict(stats),
                       "VKF_all":fig_all,
                       "VKF_PSTH":fig_PSTH,
                       "epoch":0})
    

        self.Curriculum_learning(env_kwargs)
    
    
    def RunFIT(self):
        encoder = self.encoder
        encoder_dt = self.encoder_dt
        decoder_dt = self.decoder_dt
        M1_delay=self.M1_delay
        PMd_delay=self.PMd_delay
        noise_alpha = self.noise_alpha
        obs_delay=self.obs_delay//encoder_dt
        obs_hand_delay=self.obs_hand_delay//encoder_dt
        num_channel=self.num_channel
        learning_epochs=self.learning_epochs
        wandb = self.wandb
        smooth_coef = self.smooth_coef
        zero_coef = self.zero_coef
        min_acceptance_window = self.min_acceptance_window
        acceptance_window = self.acceptance_window

        FIT = KFDecoder(type='FIT',num_channel=num_channel,name='FIT.npz')
        if self.retrain_decoder:
            ## Run hand
            data_hand = self.RunHand(task='pinball',b_retrain=False)
            data_hand.downSample(int(self.decoder_dt//self.encoder_dt))

            ## Train FIT
            FIT.run(data_hand,reach_start_time=250, reach_end_time='tFA')

        else:
            print("load FIT")
            pretrained_decoder = "../pretrained/decoders/FIT.npz"
            FIT.load(pretrained_decoder)


        FIT.save(wandb.run.dir)
        
        ## evaluate FIT
        exp_setting = get_exp_setting()
        exp_setting['b_bypass']=False
        exp_setting['b_synthesize']=True
        exp_setting['encoder_name']=encoder
        exp_setting['M1_delay']=M1_delay
        exp_setting['PMd_delay']=PMd_delay
        exp_setting['decoder_type']='FIT'
        exp_setting['decoder_name']='FIT.npz'
        exp_setting['decoder_path']= wandb.run.dir 
        exp_setting['encoder_refresh_rate']=encoder_dt
        exp_setting['decoder_refresh_rate']=decoder_dt
        env_kwargs = get_FIT_env_kwargs()
        env_kwargs['obs_delay']=obs_delay
        env_kwargs['obs_hand_delay']=obs_hand_delay
        env_kwargs['noise_alpha'] = noise_alpha
        env_kwargs['exp_setting']=exp_setting

        env_kwargs['acceptance_window']=acceptance_window
        env_kwargs['n_eval_episodes']=100
        env_train = make_vec_env(CenteroutEnv,env_kwargs=env_kwargs,n_envs=16,seed=0)

        FIT_agent = PPOCAPSZ(MlpPolicy,env_train
                             ,learning_rate=self.learning_rate
                             ,smooth_coef=smooth_coef
                             ,zero_coef=zero_coef)

        if self.pretrained_mode==0:
            print("Agent is random initialized")
        elif self.pretrained_mode==1:
            pretrained_agent = "../pretrained/agents/constrained_PPO/FIT_agent.zip"
            FIT_agent.load_parameters(pretrained_agent)
        elif self.pretrained_mode==2:
            pretrained_agent = "../pretrained/agents/naive_PPO/FIT_agent.zip"
            FIT_agent.load_parameters(pretrained_agent)
        elif self.pretrained_mode==3:
            pretrained_agent = "../pretrained/agents/constrained_PPO/hand_agent.zip"
            FIT_agent.load_parameters(pretrained_agent)

        self.agent = FIT_agent


        assert FIT_agent.smooth_coef == self.smooth_coef
        assert FIT_agent.zero_coef == self.zero_coef

        if wandb!=None:
            FIT_agent.save(os.path.join(wandb.run.dir,"FIT_agent_ep{}.zip".format(0)))
            data_FIT = RLDataset(self.agent,{**env_kwargs,'verbose':True},deterministic=False,M1_delay=0,PMd_delay=0)
            data_FIT.PlotAll()
            fig_all = plt.gcf()
            data_FIT.PlotPSTH()
            fig_PSTH = plt.gcf()
            stats = data_FIT.Statistics().iloc[2][1:]
            wandb.log({**dict(stats),
                       "FIT_all":fig_all,
                       "FIT_PSTH":fig_PSTH,
                       "epoch":0})

        self.Curriculum_learning(env_kwargs)

    def RunPVKF(self):
        encoder = self.encoder
        encoder_dt = self.encoder_dt
        decoder_dt = self.decoder_dt
        M1_delay=self.M1_delay
        PMd_delay=self.PMd_delay
        obs_delay=self.obs_delay//encoder_dt
        obs_hand_delay=self.obs_hand_delay//encoder_dt
        noise_alpha = self.noise_alpha
        num_channel=self.num_channel
        learning_epochs=self.learning_epochs
        wandb = self.wandb
        smooth_coef = self.smooth_coef
        zero_coef = self.zero_coef
        min_acceptance_window = self.min_acceptance_window
        acceptance_window = self.acceptance_window

        PVKF = KFDecoder(type='PVKF',num_channel=num_channel,name='PVKF.npz')
        if self.retrain_decoder:
            ## Run hand
            data_hand = self.RunHand(task='centerout',b_retrain=False)
            data_hand.downSample(int(self.decoder_dt//self.encoder_dt))
            ## Train PVKF
            PVKF = KFDecoder(type='PVKF',num_channel=num_channel,name='PVKF.npz')
            PVKF.run(data_hand)
        else:
            PVKF.load("../pretrained/decoders/PVKF.npz")

        PVKF.save(wandb.run.dir)
        ## evaluate PVKF
        exp_setting = get_exp_setting()
        exp_setting['b_bypass']=False
        exp_setting['b_synthesize']=True
        exp_setting['encoder_name']=encoder
        exp_setting['M1_delay']=M1_delay
        exp_setting['PMd_delay']=PMd_delay
        exp_setting['decoder_type']='PVKF'
        exp_setting['decoder_name']='PVKF.npz'
        exp_setting['decoder_path']=wandb.run.dir
        exp_setting['encoder_refresh_rate']=encoder_dt
        exp_setting['decoder_refresh_rate']=decoder_dt

        env_kwargs = get_PVKF_env_kwargs()
        env_kwargs['obs_delay']=obs_delay
        env_kwargs['obs_hand_delay']=obs_hand_delay
        env_kwargs['noise_alpha'] = noise_alpha
        env_kwargs['exp_setting']=exp_setting

        env_kwargs['acceptance_window']=acceptance_window
        env_kwargs['n_eval_episodes']=100
        env_train = make_vec_env(CenteroutEnv,env_kwargs=env_kwargs,n_envs=16,seed=0)

        PVKF_agent = PPOCAPSZ(MlpPolicy,env_train
                              ,learning_rate = self.learning_rate
                              ,smooth_coef=smooth_coef
                              ,zero_coef=zero_coef)


        if self.pretrained_mode==0:
            print("Agent is random initialized")
        elif self.pretrained_mode==1:
            pretrained_agent = "../pretrained/agents/constrained_PPO/PVKF_agent.zip"
            PVKF_agent.load_parameters(pretrained_agent)
        elif self.pretrained_mode==2:
            pretrained_agent = "../pretrained/agents/naive_PPO/PVKF_agent.zip"
            PVKF_agent.load_parameters(pretrained_agent)
        elif self.pretrained_mode==3:
            pretrained_agent = "../pretrained/agents/constrained_PPO/hand_agent.zip"
            PVKF_agent.load_parameters(pretrained_agent)

        self.agent = PVKF_agent


        assert PVKF_agent.smooth_coef == smooth_coef
        assert PVKF_agent.zero_coef == zero_coef
   
        if wandb!=None:
            PVKF_agent.save(os.path.join(wandb.run.dir,"PVKF_agent_ep{}.zip".format(0)))
            data_PVKF = RLDataset(self.agent,{**env_kwargs,'verbose':True},deterministic=False,M1_delay=0,PMd_delay=0)
            data_PVKF.PlotAll()
            fig_all = plt.gcf()
            data_PVKF.PlotPSTH()
            fig_PSTH = plt.gcf()
            stats = data_PVKF.Statistics().iloc[2][1:]
            wandb.log({**dict(stats),
                       "PVKF_all":fig_all,
                       "PVKF_PSTH":fig_PSTH,
                       "epoch":0})
    
        self.Curriculum_learning(env_kwargs)
        
   
        
    def RunReFIT(self):
        encoder = self.encoder
        encoder_dt = self.encoder_dt
        decoder_dt = self.decoder_dt
        M1_delay=self.M1_delay
        PMd_delay=self.PMd_delay
        obs_delay=self.obs_delay//encoder_dt
        obs_hand_delay=self.obs_hand_delay//encoder_dt
        noise_alpha = self.noise_alpha
        num_channel=self.num_channel
        learning_epochs=self.learning_epochs
        wandb = self.wandb
        smooth_coef = self.smooth_coef
        zero_coef = self.zero_coef
        min_acceptance_window = self.min_acceptance_window
        acceptance_window = self.acceptance_window

        ReFIT = KFDecoder(type='ReFIT',num_channel=num_channel,name='ReFIT.npz')
        if self.retrain_decoder:
            ## Run hand
            data_hand = self.RunHand(task='centerout',b_retrain=False)
            data_hand.downSample(int(self.decoder_dt//self.encoder_dt))
            ## Train PVKF
            PVKF = KFDecoder(type='PVKF',num_channel=num_channel,name='PVKF.npz')
            PVKF.run(data_hand)
            PVKF.save(wandb.run.dir)
            ## evaluate PVKF
            exp_setting = get_exp_setting()
            exp_setting['b_bypass']=False
            exp_setting['b_synthesize']=True
            exp_setting['encoder_name']=encoder
            exp_setting['M1_delay']=M1_delay
            exp_setting['PMd_delay']=PMd_delay
            exp_setting['decoder_type']='PVKF'
            exp_setting['decoder_name']='PVKF.npz'
            exp_setting['decoder_path']=wandb.run.dir
            exp_setting['encoder_refresh_rate']=encoder_dt
            exp_setting['decoder_refresh_rate']=decoder_dt

            env_kwargs = get_PVKF_env_kwargs()
            env_kwargs['obs_delay']=obs_delay
            env_kwargs['obs_hand_delay']=obs_hand_delay
            env_kwargs['noise_alpha'] = noise_alpha
            env_kwargs['exp_setting']=exp_setting

            PVKF_agent=PPOCAPSZ.load("../pretrained/agents/constrained_PPO/PVKF_agent.zip")
            PVKF_agent.save(os.path.join(wandb.run.dir,"PVKF_agent.zip"))
            data_PVKF = RLDataset(PVKF_agent,{**env_kwargs,'verbose':True},deterministic=False,M1_delay=0,PMd_delay=0)
            print(data_PVKF.Statistics())
            data_PVKF.PlotAll()
            fig_all = plt.gcf()
            wandb.log({"PVKF_all":fig_all})
            
            ##data_PVKF.downSample(int(self.decoder_dt//self.encoder_dt))
            ## Train ReFIT
            ReFIT.run(data_PVKF,reach_start_time='ttO',reach_end_time='tLA')
        else:
            ReFIT.load("../pretrained/decoders/ReFIT.npz")

        ReFIT.save(wandb.run.dir)

        ## evaluate ReFIT
        exp_setting = get_exp_setting()
        exp_setting['b_bypass']=False
        exp_setting['b_synthesize']=True
        exp_setting['encoder_name']=encoder
        exp_setting['M1_delay']=M1_delay
        exp_setting['PMd_delay']=PMd_delay
        exp_setting['decoder_type']='ReFIT'
        exp_setting['decoder_name']='ReFIT.npz'
        exp_setting['decoder_path']= wandb.run.dir 
        exp_setting['encoder_refresh_rate']=encoder_dt
        exp_setting['decoder_refresh_rate']=decoder_dt
        env_kwargs = get_ReFIT_env_kwargs()
        env_kwargs['obs_delay']=obs_delay
        env_kwargs['obs_hand_delay']=obs_hand_delay
        env_kwargs['noise_alpha'] = noise_alpha
        env_kwargs['exp_setting']=exp_setting
        env_kwargs['acceptance_window']=acceptance_window
        env_kwargs['n_eval_episodes']=100
        env_train = make_vec_env(CenteroutEnv,env_kwargs=env_kwargs,n_envs=16,seed=0)
      
        ReFIT_agent = PPOCAPSZ(MlpPolicy,env_train
                     ,learning_rate=self.learning_rate
                     ,smooth_coef=smooth_coef
                     ,zero_coef=zero_coef)

        if self.pretrained_mode==0:
            print("Agent is random initialized")
        elif self.pretrained_mode==1:
            pretrained_agent = "../pretrained/agents/constrained_PPO/ReFIT_agent.zip"
            ReFIT_agent.load_parameters(pretrained_agent)
        elif self.pretrained_mode==2:
            pretrained_agent = "../pretrained/agents/naive_PPO/ReFIT_agent.zip"
            ReFIT_agent.load_parameters(pretrained_agent)
        elif self.pretrained_mode==3:
            pretrained_agent = "../pretrained/agents/constrained_PPO/hand_agent.zip"
            ReFIT_agent.load_parameters(pretrained_agent)
        elif self.pretrained_mode==4:
            pretrained_agent = "../notebooks/CAPSZ-pvtpv-ReFIT-Real-b25-b50.zip"
            ReFIT_agent.load_parameters(pretrained_agent)


        self.agent = ReFIT_agent

        assert ReFIT_agent.smooth_coef == smooth_coef
        assert ReFIT_agent.zero_coef == zero_coef
   
        if wandb!=None:
            ReFIT_agent.save(os.path.join(wandb.run.dir,"ReFIT_agent_ep{}.zip".format(0)))
            data_ReFIT = RLDataset(self.agent,{**env_kwargs,'verbose':True},deterministic=False,M1_delay=0,PMd_delay=0)
            data_ReFIT.PlotAll()
            fig_all = plt.gcf()
            data_ReFIT.PlotPSTH()
            fig_PSTH = plt.gcf()
            stats = data_ReFIT.Statistics().iloc[2][1:]
            wandb.log({**dict(stats),
                       "ReFIT_all":fig_all,
                       "ReFIT_PSTH":fig_PSTH,
                       "epoch":0})
    
        self.Curriculum_learning(env_kwargs)
        

    def RunFORCE(self):
        encoder = self.encoder
        encoder_dt = self.encoder_dt
        decoder_dt = self.decoder_dt
        M1_delay=self.M1_delay
        PMd_delay=self.PMd_delay
        obs_delay=self.obs_delay//encoder_dt
        obs_hand_delay=self.obs_hand_delay//encoder_dt
        noise_alpha = self.noise_alpha
        num_channel=self.num_channel
        learning_epochs=self.learning_epochs
        wandb = self.wandb
        smooth_coef = self.smooth_coef
        zero_coef = self.zero_coef
        min_acceptance_window = self.min_acceptance_window
        acceptance_window = self.acceptance_window


        FORCE = FORCEDecoder(num_channel=num_channel,name='FORCE.npz',dt=decoder_dt)
        if self.retrain_decoder:
            ## Run hand
            data_hand = self.RunHand(task='centerout',b_retrain=False)
            data_hand.downSample(int(self.decoder_dt//self.encoder_dt))

            ### Train FORCE
            print("Train FORCE")
            FORCE.learnFromData(data_hand)
            FORCE.learnFromData(data_hand)
            FORCE.learnFromData(data_hand)
            FORCE.learnFromData(data_hand)
        else:
            pretrained_model = "../notebooks/wandb/run-20220128_124212-3q7w1ozq/files/FORCE.npz"
            FORCE.load(pretrained_model)

        FORCE.save(wandb.run.dir)
    
        ## evaluate FORCE
        exp_setting = get_exp_setting()
        exp_setting['b_bypass']=False
        exp_setting['b_synthesize']=True
        exp_setting['encoder_name']=encoder
        exp_setting['M1_delay']=M1_delay
        exp_setting['PMd_delay']=PMd_delay
        exp_setting['refresh_rate']=25
        exp_setting['decoder_type']='FORCE'
        exp_setting['decoder_name']='FORCE.npz'
        exp_setting['decoder_path']= wandb.run.dir 
        exp_setting['encoder_refresh_rate']=encoder_dt
        exp_setting['decoder_refresh_rate']=decoder_dt

        env_kwargs = get_FORCE_env_kwargs()
        env_kwargs['obs_delay']=obs_delay
        env_kwargs['obs_hand_delay']=obs_hand_delay
        env_kwargs['noise_alpha'] = noise_alpha
        env_kwargs['exp_setting']=exp_setting
    
        env_kwargs['acceptance_window']=acceptance_window
        env_kwargs['n_eval_episodes']=100
        env_train = make_vec_env(CenteroutEnv,env_kwargs=env_kwargs,n_envs=16,seed=0)

        FORCE_agent = PPOCAPSZ(MlpPolicy,env_train
                                ,learning_rate=self.learning_rate
                                ,smooth_coef=smooth_coef
                                ,zero_coef=zero_coef)



        if self.pretrained_mode==0:
            print("Agent is random initialized")
        elif self.pretrained_mode==1:
            pretrained_agent = "../pretrained/agents/constrained_PPO/FORCE_agent.zip"
            FORCE_agent.load_parameters(pretrained_agent)
        elif self.pretrained_mode==2:
            pretrained_agent = "../pretrained/agents/naive_PPO/FORCE_agent.zip"
            FORCE_agent.load_parameters(pretrained_agent)
        elif self.pretrained_mode==3:
            pretrained_agent = "../pretrained/agents/constrained_PPO/hand_agent.zip"
            FORCE_agent.load_parameters(pretrained_agent)

        self.agent = FORCE_agent

        assert FORCE_agent.smooth_coef == smooth_coef
        assert FORCE_agent.zero_coef == zero_coef
    
        if wandb!=None:
            FORCE_agent.save(os.path.join(wandb.run.dir,"FORCE_agent_ep{}.zip".format(0)))
            data_FORCE = RLDataset(self.agent,{**env_kwargs,'verbose':True},deterministic=False,M1_delay=0,PMd_delay=0)
            data_FORCE.PlotAll()
            fig_all = plt.gcf()
            data_FORCE.PlotPSTH()
            fig_PSTH = plt.gcf()
            stats = data_FORCE.Statistics().iloc[2][1:]
            wandb.log({**dict(stats),
                       "FORCE_all":fig_all,
                       "FORCE_PSTH":fig_PSTH,
                       "epoch":0})

        self.Curriculum_learning(env_kwargs)
 
        return data_FORCE


if __name__ == '__main__':
    #wandb = None
    wandb.init(project="BMI-simulator", entity="kenfuliang")

    parser = ArgumentParser()
    parser.add_argument('--encoder_dt', type=int, default=25)
    parser.add_argument('--encoder_date', type=str, default='20220105')
    parser.add_argument('--monkey_date', type=str, default='0512')
    parser.add_argument('--decoder_dt', type=int, default=25)
    parser.add_argument('--M1_delay', type=int, default=8)
    parser.add_argument('--PMd_delay', type=int, default=8)
    parser.add_argument('--noise_alpha', type=float, default=0)
    parser.add_argument('--obs_delay', type=int, default=0) # (ms)
    parser.add_argument('--obs_hand_delay', type=int, default=0) # (ms)
    parser.add_argument('--learning_epochs', type=int, default=0)
    parser.add_argument('--num_channel',type=int, default=192)
    parser.add_argument('--decoder', type=str, default='hand')
    parser.add_argument('--ke', type=float, default=1.0)
    parser.add_argument('--learning_rate',type=float, default=2.5e-4)
    parser.add_argument('--smooth_coef', type=float, default=7e-2)
    parser.add_argument('--zero_coef', type=float, default=3e-3)
    parser.add_argument('--retrain_decoder', type=int,default=0)
    parser.add_argument('--pretrained_mode', type=int,default=1)
    parser.add_argument('--target_radius', type=int,default=80)
    parser.add_argument('--acceptance_window', type=int,default=40)
    parser.add_argument('--min_acceptance_window', type=int,default=30)
    parser.add_argument('--max_acceptance_window', type=int,default=120)

    args = parser.parse_args()

    wandb.config.update(args) # adds all of the arguments as config variables

    args = vars(args)
    args['encoder'] = "Seq2Seq_stateful_b{}_{}_r192_decay0.01_ke{}_re0_delayed{}{}_extraBins0_{}_ep4.h5".format(args['encoder_dt'],args['encoder_date'],args['ke'],args['M1_delay'],args['PMd_delay'],args['monkey_date'])
    AgentEncoderDecoder(wandb,**args).sim()

