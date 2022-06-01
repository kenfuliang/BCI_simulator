import sys
import os
#sys.path.append("../code/")
#sys.path.append("../code/decoder/")

sys.path.append("./decoder/")
import numpy as np
import matplotlib.pyplot as plt
from metrics import mean_square_error
# from pylds.models import DefaultLDS (remove LDS)
from metrics import *
from decoder.DecoderBase import DecoderBase
import warnings
import scipy.io as sio
import copy
from util import binning

class KF(DecoderBase):
    def __init__(self,name="KFDecoder"):
        self.name = name

    def save(self,path='./',name=None):
        name = name if (name is not None) else self.name
        name = os.path.join(path,name)
        print(name)
        if self.type == 'NDF':
            np.savez(name,A         =self.A
                         ,W         =self.W
                         ,C         =self.C
                         ,Q         =self.Q
                         #,SmS       =self.SmS
                         ,Lw        =self.Lw
                         #,decodeState = self.decodeState
                         #,pk        =self.pk
                         ,isFB      =self.isFB
                         #,Cfeedback =self.Cfeedback
                         ,M1        =self.M1
                         ,M2        =self.M2
                         )
        else:
            np.savez(name,A         =self.A
                         ,W         =self.W
                         ,C         =self.C
                         ,Q         =self.Q
                         ,isFB      =self.isFB
                         ,Cfeedback =self.Cfeedback
                         ,M1        =self.M1
                         ,M2        =self.M2
                         ,dt        =self.dt
                         )

    def load(self,decoder_path):
        L = np.load(decoder_path)
        self.A=L['A']
        self.W=L['W']
        self.C=L['C']
        self.Q=L['Q']
        self.isFB = L['isFB']
        self.Cfeedback = L['Cfeedback']
        self.M1 = L['M1']
        self.M2 = L['M2']
        self.dt = L['dt']

        if self.type == 'NDF':
            #self.SmS = L['SmS']
            self.Lw  = L['Lw']
            #self.decodeState = L['decodeState']
            #self.decodeState = np.zeros(self.decodeState.shape)
            #self.pk = np.zeros(L['W'].shape)



    def loadFromMat(self,file_path):
        mat = sio.loadmat(file_path).get('M')
        if self.type == 'NDF':
            A   = mat[0][0]['Ms'][0][0]['signals'][0][0]['values']
            W   = mat[0][0]['Ws'][0][0]['signals'][0][0]['values']
            C   = mat[0][0]['L'][0][0]['signals'][0][0]['values']
            Q   = mat[0][0]['Qs'][0][0]['signals'][0][0]['values']
            #SmS = mat[0][0]['SmS'][0][0]['signals'][0][0]['values']
            Lw = mat[0][0]['Lw'][0][0]['signals'][0][0]['values']
            #self.decodeState = mat[0][0]['net'][0][0]['signals'][0][0]['values'][0][0]['x0']
            #self.decodeState = np.zeros(self.decodeState.shape)
            #self.pk = np.zeros(W.shape)

            #self.SmS = SmS
            self.Lw  = Lw

        else:
            A = mat[0][0]['A'][0][0]['signals'][0][0]['values']
            W = mat[0][0]['W'][0][0]['signals'][0][0]['values']
            C = mat[0][0]['C'][0][0]['signals'][0][0]['values']
            Q = mat[0][0]['Q'][0][0]['signals'][0][0]['values']
            A = A[[0,1,3,4,6]]
            A = A[:,[0,1,3,4,6]]
            W = W[[0,1,3,4,6]]
            W = W[:,[0,1,3,4,6]]
            C = C[:,[0,1,3,4,6]]

        ## VKF, PVKF, FIT, ReFIT, NDF
        if(self.type=='VKF' or self.type=='ReFIT' or self.type=='FIT'):
            isVel=True
        else:#PVKF
            isVel=False

        if(self.type=='ReFIT' or self.type=='FIT'):
            isFB=True
            Cfeedback = mat[0][0]['Cfeedback']['signals'][0][0]['values'][0][0]
            Cfeedback = Cfeedback[:,[0,1,3,4,6]]
        else:#FIT
            isFB=False
            Cfeedback = 0
        [self.M1,self.M2] = self.kalmanSteadyState(A,W,C,Q,isVel=isVel,isFB=isFB,Cfeedback=Cfeedback)


        self.A, self.W, self.C, self.Q = A, W, C, Q

    def train(self,X,Y):
        Y = [Y[:self.num_channel] for Y in Y]

        if self.type == 'ReFIT' or self.type=='FIT':
            self.fitKalmanVPFB2DZ(X,Y)
        elif self.type == 'PVKF':
            self.fitKalmanV2D(X,Y)
        elif self.type == 'VKF':
            self.fitKalmanVonly2D(X,Y)
        elif self.type == 'NDF':
            self.fitWienerLDSV2D(X,Y)

        A,W,C,Q,Cfeedback = self.A, self.W, self.C, self.Q, self.Cfeedback
        isVel = self.isVel
        isFB = self.isFB

        [self.M1,self.M2] = self.kalmanSteadyState(A,W,C,Q,isVel=isVel,isFB=isFB,Cfeedback=Cfeedback)

    def decode(self,X):
        X = X[:self.num_channel]
        X = X.reshape(-1)
        lastState = self.lastState
        M1=self.M1
        M2=self.M2
        nextState = M1.dot(lastState)+M2.dot(X)
        self.lastState = nextState
        return nextState


    def decodeFromData(self,data,spike,trial):
        dt=self.dt
        alpha = self.alpha
        M1=self.M1
        M2=self.M2
        assert self.dt % data.dt==0
        n = self.dt//data.dt
        X = binning(spike,n,'sum')
        X = X[1:].T
        if(self.type=='NDF'):
            raise NotImplementedError
            #decodeState = self.lastState.T
            #for j in range(X.shape[1]):
            #    lastState   = decodeState[-1]
            #    nextState   = self.decode(lastState,X[:,j])
            #    decodeState = np.vstack((decodeState,nextState))
            #decodeState = np.hstack((decodeState,np.ones((decodeState.shape[0],1))))
            #PVs= decodeState.dot(self.Lw.T)
            #return PVs
        else:
            initP = data.df['binCursorPos'][trial][data.extraBins].T
            initV = data.df['binCursorVel'][trial][data.extraBins].T/1000
            decodedPos = [initP]
            decoderState = np.hstack([initP,initV,1])
            self.setState(decoderState)
            for j in range(0,X.shape[1]):
                Z   = self.decode(X[:,j])
                assert len(Z.shape)==1
                _P = Z[0:2]
                _V = Z[2:4]*1000
                decodedPos.append((1-alpha)*_P + (alpha)*(decodedPos[-1]+_V*dt/1000))
            return np.array(decodedPos)

    def setState(self,state):
        self.lastState = state.reshape(-1)

    def decode_NDF(self,spike):
        sk = self.sk
        pk = self.pk
        Ms = self.A
        Ws = self.W
        L  = self.C
        Qs = self.Q
        Lw = self.Lw
        LQsinv = C.T @ np.linalg.inv(self.Q)
        LQsinvL = C.T @ np.linalg.inv(self.Q)@self.C

        sk = Ms @ sk
        pk = Ms @ pk @ Ms.T + Ws
        Kk = np.linalg.inv(np.eye(pk.shape[0])+pk@LQsinvL) @ pk @ LQsinv
        nc = L @ sk

        sk = sk + Kk@(nC - L@sk)
        pk = (np.eye(pk.shape[0])-Kk@L)@pk
        z = Lw@np.hstack((sk,np.zeros(sk.shape[0],1)))
        return z[1:]


    def kalmanSteadyState(self,A,W,C,Q,isVel, isFB, Cfeedback):
        self.isVel      = isVel
        self.isFB       = isFB
        self.Cfeedback  = Cfeedback
        TOLERANCE = 1e-13
        MAX_ITERS = 1e4
        count = 0
        CQinv = C.T.dot(np.linalg.pinv(Q))
        CQinvC = CQinv.dot(C)
        if isFB:
            C = C+Cfeedback

        #initialization
        xk = np.zeros(A.shape)
        Pk = np.zeros(W.shape)
        M1_prev = np.zeros(A.shape)
        M2_prev = np.zeros(C.shape).T
        iterativeDiff = np.array([1])

        while any(iterativeDiff > TOLERANCE):
            Pk = A.dot(Pk).dot(A.T)+W

            if (isFB or isVel):
                Pk[0:2,:] = 0
                Pk[:,0:2] = 0

            Kk = Pk.dot(C.T).dot(np.linalg.pinv(C.dot(Pk).dot(C.T)+Q))
            # update uncertainty
            Pk = Pk - Pk.dot(C.T).dot(np.linalg.pinv(C.dot(Pk).dot(C.T)+Q)).dot(C).dot(Pk)
            M1 = A - Kk.dot(C).dot(A)
            M2 = Kk

            M1_diff = M1-M1_prev
            M2_diff = M2-M2_prev

            iterativeDiff = np.hstack( [np.hstack(M1_diff),np.hstack(M2_diff)])
            M1_prev = M1
            M2_prev = M2
            count+=1
            if count>MAX_ITERS:
                warnings.warn("kalman steady state is not converged")
                return (M1,M2)
        return (M1,M2)

    def fitKalmanVPFB2DZ(self,X,Y):
        # fitKalmanVPFB2DZ (ReFIT)
        dt = self.dt
        X1X1t = np.zeros((2,2))
        X2X1t = np.zeros((2,2))
        for n in range(len(X)):
            X1X1t = X1X1t + X[n][2:4,:-1].dot(X[n][2:4,:-1].T)
            X2X1t = X2X1t + X[n][2:4,1:].dot(X[n][2:4,:-1].T)
        A = np.eye(5)
        A[0,2] = dt
        A[1,3] = dt
        A[2:4,2:4] = X2X1t.dot(np.linalg.pinv(X1X1t))

        ## fit the state noise W
        W = np.zeros((5,5))
        K = 0
        for n in range(len(X)):
            W = W + (A.dot(X[n][:,0:-1]) - X[n][:,1:]).dot((A.dot(X[n][:,:-1]) - X[n][:,1:]).T)
            K = K + X[n].shape[1]-1
        W = W/K
        W = 0.5*(W+W.T)
        W[0:2,:] = 0
        W[:,0:2] = 0
        W[4,:]=0
        W[:,4]=0

        ## fit the observation matrix C
        XXt = np.zeros((3,3))
        idx2DPos = np.array([0,1,4])
        ZXt = np.zeros((self.num_channel,3))
        nPosSamples = 0
        for n in range(len(X)):
            stoppedTimes = (X[n][2,:]==0) & (X[n][3,:]==0)
            if sum(stoppedTimes)==0:
                continue
            stoppedTimes =  [i for i, val in enumerate(stoppedTimes) if val]
            XXt = XXt + X[n][idx2DPos[:,None],stoppedTimes].dot(X[n][idx2DPos[:,None],stoppedTimes].T)
            ZXt = ZXt + Y[n][:,stoppedTimes].dot(X[n][idx2DPos[:,None],stoppedTimes].T)
            nPosSamples = nPosSamples + sum(stoppedTimes)

        Cpos = ZXt.dot(np.linalg.pinv(XXt))

        idx2DVel = [2,3]
        XXt = np.zeros((2,2))
        ZXt = np.zeros((self.num_channel,2))
        for n in range(len(X)):
            XXt = XXt + X[n][idx2DVel,:].dot(X[n][idx2DVel][:].T)
            ZXt = ZXt + (Y[n]-Cpos.dot(X[n][idx2DPos,:])).dot(X[n][idx2DVel,:].T) # why idx2DPos and idx2DVel here?

        Cvel = ZXt.dot(np.linalg.pinv(XXt))

        C = np.zeros((self.num_channel,5))
        C[:,idx2DPos] = Cpos
        C[:,idx2DVel] = Cvel

        ## fit the observation noise Q
        Q = np.zeros(self.num_channel)
        K = 0
        for n in range(len(X)):
            Q = Q + (Y[n] - C.dot(X[n])).dot((Y[n] - C.dot(X[n])).T)
            K = K + X[n].shape[1]
        Q = Q/K
        Q = 0.5*(Q+Q.T)

        ## split C into C and Cfeedback
        Cfeedback = np.zeros(C.shape)
        Cfeedback[:,[0,1,4]] = C[:,[0,1,4]]
        C[:,[0,1,4]] = 0

        self.A  =   A
        self.W  =   W
        self.C  =   C
        self.Q  =   Q
        self.Cfeedback = Cfeedback

    def fitKalmanV2D(self,X,Y):# PVKF
        dt = self.dt

        ## fit the A
        X1X1t = np.zeros((2,2))
        X2X1t = np.zeros((2,2))
        for n in range(len(X)):
            X1X1t = X1X1t + X[n][2:4,:-1].dot(X[n][2:4,:-1].T)
            X2X1t = X2X1t + X[n][2:4,1:].dot(X[n][2:4,:-1].T)
        A = np.eye(5)
        A[0,2] = dt
        A[1,3] = dt
        A[2:4,2:4] = X2X1t.dot(np.linalg.pinv(X1X1t))

        ## fit the state noise W
        W = np.zeros((5,5))
        K = 0
        for n in range(len(X)):
            W = W + (A.dot(X[n][:,0:-1]) - X[n][:,1:]).dot((A.dot(X[n][:,:-1]) - X[n][:,1:]).T)
            K = K + X[n].shape[1]-1
        W = W/K
        W = 0.5*(W+W.T)
        W[0:2,:] = 0
        W[:,0:2] = 0
        W[4,:]=0
        W[:,4]=0

        ## fit the observation matrix C
        XXt = np.zeros((5,5))
        ZXt = np.zeros((self.num_channel,5))
        nPosSamples = 0
        for n in range(len(X)):
            XXt = XXt + X[n].dot(X[n].T)
            ZXt = ZXt + Y[n].dot(X[n].T)

        C = ZXt.dot(np.linalg.pinv(XXt))

        ## fit the observation noise Q
        Q = np.zeros(self.num_channel)
        K = 0
        for n in range(len(X)):
            Q = Q + (Y[n] - C.dot(X[n])).dot((Y[n] - C.dot(X[n])).T)
            K = K + X[n].shape[1]
        Q = Q/K
        Q = 0.5*(Q+Q.T)

        self.A  =   A
        self.W  =   W
        self.C  =   C
        self.Q  =   Q
        self.Cfeedback = 0

    def fitKalmanVonly2D(self,X,Y):# VKF
        dt = self.dt
        X1X1t = np.zeros((2,2))
        X2X1t = np.zeros((2,2))
        for n in range(len(X)):
            X1X1t = X1X1t + X[n][2:4,:-1].dot(X[n][2:4,:-1].T)
            X2X1t = X2X1t + X[n][2:4,1:].dot(X[n][2:4,:-1].T)
        A = np.eye(5)
        A[0,2] = dt
        A[1,3] = dt
        A[2:4,2:4] = X2X1t.dot(np.linalg.pinv(X1X1t))

        ## fit the state noise W
        W = np.zeros((5,5))
        K = 0
        for n in range(len(X)):
            W = W + (A.dot(X[n][:,0:-1]) - X[n][:,1:]).dot((A.dot(X[n][:,:-1]) - X[n][:,1:]).T)
            K = K + X[n].shape[1]-1
        W = W/K
        W = 0.5*(W+W.T)

        ## fit the observation matrix C
        XXt = np.zeros((3,3))

        idx2DVel = [2,3,4]
        XXt = np.zeros((3,3))
        ZXt = np.zeros((self.num_channel,3))
        for n in range(len(X)):
            XXt = XXt + X[n][idx2DVel,:].dot(X[n][idx2DVel][:].T)
            ZXt = ZXt + Y[n].dot(X[n][idx2DVel,:].T)

        Cvel = ZXt.dot(np.linalg.pinv(XXt))

        C = np.zeros((self.num_channel,5))
        C[:,idx2DVel] = Cvel

        ## fit the observation noise Q
        Q = np.zeros(self.num_channel)
        K = 0
        for n in range(len(X)):
            Q = Q + (Y[n] - C.dot(X[n])).dot((Y[n] - C.dot(X[n])).T)
            K = K + X[n].shape[1]
        Q = Q/K
        Q = 0.5*(Q+Q.T)

        self.A  =   A
        self.W  =   W
        self.C  =   C
        self.Q  =   Q
        self.Cfeedback = 0

    def fitWienerLDSV2D(self,X,Y): # NDF
        D_obs = self.num_channel     # Observed data dimension
        D_latent = 20   # Latent state dimension
        D_input = 0     # Exogenous input dimension
        N_samples = 100
        LDS = DefaultLDS(D_obs, D_latent, D_input)
        for y in Y:
            LDS.add_data(y)
        for ii in range(N_samples):
            print(ii)
            LDS.resample_model()

        A,W,C,Q = LDS.A, LDS.sigma_states, LDS.C, LDS.sigma_obs
        M1,M2 = self.kalmanSteadyState(A,W,C,Q,isVel=True, isFB=False, Cfeedback=0)

        # sampling the init state
        mu = LDS.mu_init
        sigma = LDS.sigma_init
        s = np.random.multivariate_normal(mu, sigma)

        Y = np.concatenate(Y)
        # estimate state sequence with M1, M2 and real neural data
        S = [s]
        for t in range(len(Y)):
            lastState = S[-1]
            nextState = M1 @ lastState + M2 @ Y[t]
            S.append(nextState)

        S = S[1:]

        S = np.hstack([S,np.ones((len(S),1))])
        X = np.concatenate(X)

        S = S.T
        X = X.T

        Lw = X.dot(S.T).dot(np.linalg.pinv(S.dot(S.T)))

        self.A = A
        self.W = W
        self.C = C
        self.Q = Q

        self.M1 = M1
        self.M2 = M2
        self.Lw = Lw

class KFDecoder(KF):
    def __init__(self
                ,num_channel=192
                ,name='KFDecoder'
                ,type=None):


        if type not in ['ReFIT','FIT','NDF','VKF','PVKF']:
            raise ValueError

        self.type = type
        self.num_channel=192
        KF.__init__(self,name)
        if type in ['FIT','ReFIT','VKF']:
            alpha = 1.0
        elif type == 'PVKF':
            alpha = 0.0
        self.alpha = alpha

        self.reset()

        if(self.type=='VKF' or self.type=='ReFIT' or self.type=='FIT'):
            isVel=True
        elif (self.type=='PVKF' or self.type=='NDF'):
            isVel=False
        self.isVel = isVel

        if(self.type=='ReFIT' or self.type=='FIT'):
            isFB=True
        elif (self.type=='FIT' or self.type=='VKF' or self.type=='PVKF' or self.type=='NDF'):
            isFB=False
        self.isFB = isFB

    def run(self, data, reach_start_time='ttO',reach_end_time='tFA'):
        extraBins = data.extraBins
        self.dt = data.dt

        if self.type == 'ReFIT' or self.type=='FIT':

            start_index = 0 if reach_start_time=='ttO' else reach_start_time//self.dt
            if self.type == 'ReFIT':
                isHand = False
            elif self.type =='FIT':
                isHand = True

            X = []
            Y = []
            for trial in data.trainingTrials:
                assert data.df['isSuccessful'][trial]==True

                #ttO = data.df['timeTargetOn'][trial]
                tFA = int((data.df['timeFirstTargetAcquire'][trial])//self.dt)
                tLA = int((data.df['timeLastTargetAcquire'][trial])//self.dt)
                if reach_end_time=='tFA':
                    end_index = tFA
                elif reach_end_time=='tLA':
                    end_index = tLA
                else:
                    end_index = reach_end_time//self.dt

                P = copy.deepcopy(data.df['binCursorPos'][trial][extraBins:])
                V = data.df['binCursorVel'][trial][extraBins:]
                T = data.df['target'][trial]

                new_vel = np.linalg.norm(V,axis=1,keepdims=True)*(T-P)/np.linalg.norm((T-P),axis=1,keepdims=True)/1000
                new_vel[tFA:]=0
                P[tFA:] = T


                x = np.concatenate((P,new_vel),axis=1)
                y = data.df['binSpike'][trial][extraBins:]

                X.append(np.concatenate([x[start_index:end_index],x[tLA:]]))    #
                Y.append(np.concatenate([y[start_index:end_index],y[tLA:]]).T)  #
                assert len(X)!=0
                assert len(Y)!=0
                assert (X[-1].shape[0]== Y[-1].shape[1]) , "Check trial %d" % (trial)

            X = [ np.hstack( [X[ii],np.ones((len(X[ii]),1))]).T for ii in range(len(X))]

            self.train(X,Y)


        elif self.type in ['VKF','PVKF']:
            X = ([np.concatenate((data.df['binHandPos'][trial][extraBins:],data.df['binHandVel'][trial][extraBins:]/1000),axis=1) for trial in data.trainingTrials])
            X = [ np.hstack( [X[ii],np.ones((len(X[ii]),1))]).T for ii in range(len(X))]
            Y = [data.df['binSpike'][trial][extraBins:].T for trial in data.trainingTrials]
            self.train(X,Y)

        elif self.type=='NDF':
            raise NotImplementedError
            X = ([np.concatenate((data.df['binHandPos'][trial][extraBins:],data.df['binHandVel'][trial][extraBins:]/1000),axis=1) for trial in data.trainingTrials])
            Y = [data.df['binSpike'][trial][extraBins:] for trial in data.trainingTrials]
            X = [np.concatenate(X)]
            Y = [np.concatenate(Y)]
            self.train(X,Y)
        else:
            raise ValueError("{} is not supported".format(self.type))

    def reset(self):
        if self.type =='NDF':
            self.lastState = np.zeros((20,1))
        else:
            self.lastState = np.array([0,0,0,0,1]).reshape(5,1)
