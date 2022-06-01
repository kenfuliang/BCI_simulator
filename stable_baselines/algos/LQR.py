
import numpy as np
from scipy import linalg

def getABfromData(X,U,iters=10):
    '''
    X is the state from time 0 to time T. Dimension is (T,D_state) where D_state is the dimension of state vector.
    U is the input from time 0 to time T. Dimension is (T,D_input) where D_input is the dimension of input vector.
    '''
    X1 = X[1:].T
    X0 = X[:-1].T
    U = U[:-1].T
    assert X1.shape[1]==X0.shape[1]
    assert U.shape[1]==X1.shape[1]
    B = np.zeros((X.shape[1],2))
    for ii in range(iters):
        D = X1-np.dot(B,U)
        A = np.dot(np.dot(D,X0.T), np.linalg.pinv(np.dot(X1,X1.T)))

        D = X1-np.dot(A,X0)
        B = np.dot(np.dot(D,U.T), np.linalg.pinv(np.dot(U,U.T)))
    return A,B

class NaiveAgent():
    def __init__(self,position_weight=1/10,velocity_weight=1/2):
        self.position_weight = position_weight
        self.velocity_weight = velocity_weight

    def predict(self,obs):
        position_weight = self.position_weight
        velocity_weight = self.velocity_weight
        cursor_position = obs[0,0:2,-1]
        cursor_velocity = obs[0,2:4,-1]
        target_position = obs[0,4:6,-1]
        action = (target_position-cursor_position)*position_weight-cursor_velocity*velocity_weight
        action = action.reshape(1,2)
        return action

class LQRAgentVel():
    def __init__(self,R=1):
        ### solve in continuous linear dynamic form
        ### x' = Ax+Bu
        #A = np.array([[0,0],
        #              [0,0]])
        ## discrete time, therefore, no need to have dt
        #B = np.array([[1,0],
        #              [0,1]])
        #Q = np.eye(2)*1
        #R = np.eye(2)*R
        #P = linalg.solve_continuous_are(A, B, Q, R)

        ## calculate optimal controller gain
        #K = np.dot(np.linalg.inv(R), np.dot(B.T, P))


        ## Solve in discrete linear dynamic form, X[t+1] = A X[t] + B U[t]
        A = np.array([[1,0],
                      [0,1]])
        # discrete time, therefore, no need to have dt
        B = np.array([[1,0],
                      [0,1]])
        Q = np.eye(2)*1
        R = np.eye(2)*R
        P = linalg.solve_discrete_are(A, B, Q, R)

        # calculate optimal controller gain
        K = np.linalg.inv(B.T@P@B+R)@(B.T@P@A)

        self.K = -K
        self.A = A
        self.B = B
        print(self.A,self.B,self.K)
    def predict(self,obs):
        cursor_position = obs[0,0:2,-1]
        cursor_velocity = obs[0,2:4,-1]
        target_position = obs[0,4:6,-1]

        x = obs[0,0:2,-1]
        x_star = np.concatenate([target_position])
        
        action = np.dot(self.K,x-x_star)
        action = action.reshape(1,2)
        
        return action


class LQRAgentAcc():
    def __init__(self,A=None,B=None,Q=1,R=1):
        ##dt = dt
        #dt = dt
        ##if A is None:
        ##    A = np.array([[1, 0, 1, 0,],
        ##                  [0, 1, 0, 1,],
        ##                  [0, 0, 1, 0,],
        ##                  [0, 0, 0, 1,]])
        ##if B is None:
        ##    B = np.array([[ (dt**2)/2,          0],
        ##                  [ 0,          (dt**2)/2],
        ##                  [dt,                  0],
        ##                  [ 0,                 dt]])
        ##    #B = np.array([[ 0,          0],
        ##    #              [ 0,          0],
        ##    #              [dt,                  0],
        ##    #              [ 0,                 dt]])
        #    
        #A = np.array([[1, 0, 1, 0,],
        #              [0, 1, 0, 1,],
        #              [0, 0, 1, 0,],
        #              [0, 0, 0, 1,]])
        #B = np.array([[ 1/2,          0],
        #             [ 0,          1/2],
        #             [ 1,           0],
        #             [ 0,           1]])
        #    

        #    
        ##Q = np.eye(4)
        #Q = np.array([[1., 0., 0., 0.],
        #              [0., 1., 0., 0.],
        #              [0., 0., 1,  0.],
        #              [0., 0., 0., 1]])
        #R = np.eye(2)*R
        #P = linalg.solve_discrete_are(A, B, Q, R)
        ##P = linalg.solve_continuous_are(A, B, Q, R)

        ## calculate optimal controller gain
        #K = np.dot(np.linalg.inv(R), np.dot(B.T, P))

        ## Solve in discrete linear dynamic form, X[t+1] = A X[t] + B U[t]
        # p = p +v + 1/2 * a
        if A is None:
            A = np.array([[1,0,1,0],
                          [0,1,0,1],
                          [0,0,1,0],
                          [0,0,0,1]])

    
        # discrete time, therefore, no need to have dt
        if B is None:
            B = np.array([[0,0],
                          [0,0],
                          [1,0],
                          [0,1]])
        Q = np.eye(A.shape[1])*Q
        R = np.eye(B.shape[1])*R
        P = linalg.solve_discrete_are(A, B, Q, R)

        # calculate optimal controller gain
        K = np.linalg.inv(B.T@P@B+R)@(B.T@P@A)

        self.K = -K
        self.A = A
        self.B = B

    def summary(self):
        print("A:",self.A)
        print("B:",self.B)
        print("K:",self.K)

    def predict(self,obs):
        cursor_position = obs[0,0:2,-1]
        cursor_velocity = obs[0,2:4,-1] 
        target_position = obs[0,4:6,-1]
        hand_position = obs[0,0:2,-1]
        hand_velocity = obs[0,2:4,-1] 

        if self.A.shape[1] == 4:
            x = np.concatenate([cursor_position, cursor_velocity])
            x_star = np.concatenate([target_position,np.zeros(2)])
        elif self.A.shape[1] == 8:
            x = np.concatenate([cursor_position, cursor_velocity,hand_position, hand_velocity])
            x_star = np.concatenate([target_position,np.zeros(6)])

        
        #x_star = np.concatenate([target_position,np.zeros(self.A.shape[1]-2)])
        #x_star = np.concatenate([target_position,np.zeros(2), target_position, np.zeros(2)])
        #x_star = np.concatenate([cursor_position, np.zeros(2),target_position, np.zeros(2)])
        
        action = np.dot(self.K,x-x_star)
        
        action = action.reshape(1,2)
        return action

class LQRAgentJerk():
    def __init__(self,R=1):
        #dt = dt
        #A = np.array([[1, 0,dt, 0,(dt**2)/2,        0],
        #              [0, 1, 0,dt,        0,(dt**2)/2],
        #              [0, 0, 1, 0,       dt,        0],
        #              [0, 0, 0, 1,        0,        dt],
        #              [0, 0, 0, 0,        1,        0],
        #              [0, 0, 0, 0,        0,        1]])
        #B = np.array([[ (1/6)*(dt**3),      0],
        #              [ 0,                  (1/6)*(dt**3)],
        #              [ (1/2)*(dt**2),      0],
        #              [ 0,                  (1/2)*(dt**2)],
        #              [dt,                  0],
        #              [ 0,                  dt]])

        #A = np.array([[1, 0, 1, 0, 1/2,        0],
        #              [0, 1, 0, 1,        0,1/2],
        #              [0, 0, 1, 0,       1,        0],
        #              [0, 0, 0, 1,        0,        1],
        #              [0, 0, 0, 0,        1,        0],
        #              [0, 0, 0, 0,        0,        1]])
        #B = np.array([[ 1/6,   0],
        #              [ 0,     1/6],
        #              [ 1/2,   0],
        #              [ 0,     1/2],
        #              [1,      0],
        #              [ 0,     1]])

        ## p v as obs
        ## j as input 
        #A = np.array([[1, 0, 1, 0],
        #              [0, 1, 0, 1],
        #              [0, 0, 1, 0],
        #              [0, 0, 0, 1]])

        #B = np.array([[ 1/6,   0],
        #              [ 0,     1/6],
        #              [ 1/2,   0],
        #              [ 0,     1/2]])

        ##Q = np.array([[1, 0., 0., 0., 0., 0.],
        ##             [0., 1, 0., 0., 0., 0.],
        ##             [0., 0., 0, 0., 0., 0.],
        ##             [0., 0., 0., 0, 0., 0.],
        ##             [0., 0., 0., 0., 0, 0.],
        ##             [0., 0., 0., 0., 0., 0]])

        ##Q = np.eye(4)
        ##R = np.eye(2)*R
        #
        ##P = linalg.solve_discrete_are(A, B, Q, R)

        ### calculate optimal controller gain
        ##K = np.dot(np.linalg.inv(R), np.dot(B.T, P))

        #X = (linalg.solve_discrete_are(A, B, Q, R))

        ## calculate optimal controller gain
        #K = np.linalg.inv(B.T@X@B+R)@(B.T@X@A)

        ## Solve in discrete linear dynamic form, X[t+1] = A X[t] + B U[t]
        # p = p +v + 1/2 * a + 1/6 * J
        # v = v + a + 1/2* J
        # a = a + J
        #A = np.array([[1,0,1,0],
        #              [0,1,0,1],
        #              [0,0,1,0],
        #              [0,0,0,1]])
        A = np.array([[1,0,1,0,1/2,   0],
                      [0,1,0,1,0,   1/2],
                      [0,0,1,0,1,     0],
                      [0,0,0,1,0,     1],
                      [0,0,0,0,1,     0],
                      [0,0,0,0,0,     1]])



        # discrete time, therefore, no need to have dt
        B = np.array([[1/6,  0],
                      [0  ,1/6],
                      [1/2,  0],
                      [0  ,1/2],
                      [1,    0],
                      [0,    1]])
        Q = np.array([[1,  0,  0,  0, 0., 0.],
                      [0., 1,  0,  0, 0., 0.],
                      [0., 0., 0,  0, 0., 0.],
                      [0., 0., 0., 0, 0., 0.],
                      [0., 0., 0., 0., 0, 0.],
                      [0., 0., 0., 0., 0., 0]])

        #Q = np.eye(6)*1
        R = np.eye(2)*R
        P = linalg.solve_discrete_are(A, B, Q, R)

        # calculate optimal controller gain
        K = np.linalg.inv(B.T@P@B+R)@(B.T@P@A)
        
        self.K = -K
        self.A = A
        self.B = B

    def summary(self):
        print("A:",self.A)
        print("B:",self.B)
        print("K:",self.K)


    def predict(self,obs):
        assert obs.shape[1]==8
        cursor_position = obs[0,0:2,-1]
        cursor_velocity = obs[0,2:4,-1]
        cursor_acc = obs[0,4:6,-1]
        target_position = obs[0,6:8,-1]

        #x = np.concatenate([cursor_position, cursor_velocity])
        x = np.concatenate([cursor_position, cursor_velocity, cursor_acc])
        
        x_star = np.concatenate([target_position,np.zeros(4)])
        
        action = np.dot(self.K,x-x_star)
        
        action = action.reshape(1,2)
        return action
