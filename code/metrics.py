import numpy as np

import scipy

########### PSTH Pearson's correlation ##########
#def PSTHcorrcoef(PSTH1,PSTH2):
#    PSTH1 = np.transpose(PSTH1,[2,0,1])
#    PSTH1 = np.vstack( [np.hstack(PSTH1[i]) for i in range(len(PSTH1))])
#    PSTH2 = np.transpose(PSTH2,[2,0,1])
#    PSTH2 = np.vstack( [np.hstack(PSTH2[i]) for i in range(len(PSTH2))])
#    rt_corr=np.zeros(len(PSTH1))
#    for i in range(len(PSTH1)):
#        rt_corr[i] = np.corrcoef(PSTH1[i],PSTH2[i])[0,1]
#    return rt_corr


def NormalizedMeanDiff(y,ypred,axis=0):
    #print(x1.shape)
    #print(x1[0:10],x2[0:10])
    #print(np.power(x1-x2,2))
    #print(np.sum(np.power(x1-x2,2),axis=1))
    #print(np.sqrt(np.sum(np.power(x1-x2,2),axis=1)))
    #rt = np.mean(np.sqrt(np.sum(np.power(x1-x2,2),axis=axis)))
    e = np.linalg.norm(y-ypred,axis=axis)
    rt = np.mean(e)/np.std(ypred)
    return  rt

########## Mean Square Error (MSE) ##########
def mean_square_error(x1,x2,axis=0):
    e = np.power(x1-x2,2)
    s = np.sum(e,axis=axis)
    rt = np.mean(s)
    return rt

def root_mean_square_error(x1,x2,axis=0):
    e = np.power(x1-x2,2)
    s = np.sum(e,axis=axis)
    rt = np.sqrt(np.mean(s))
    return rt



########## Normalized root mean square error (NRMSE) ##########
def get_NRMSE(y_test,y_test_pred):
    """
    Function to get NRMSE

    Parameters
    ----------
    y_test - the true outputs (a matrix of size number of examples x number of outputs)
    y_test_pred - the predicted outputs (a matrix of size number of examples x number of outputs)

    Returns
    -------
    NRMSE_array: An array of NRMSEs for each output
    """
    y_test = np.vstack(y_test)
    y_test_pred = np.vstack(y_test_pred)

    NRMSE_list=[] #Initialize a list that will contain the R2s for all the outputs
    for i in range(y_test.shape[1]): #Loop through outputs
        #Compute R2 for each output
        y_std=np.std(y_test[:,i])
        if y_std==0:
            NRMSE=np.nan
        else:
            NRMSE=root_mean_square_error(y_test[:,i],y_test_pred[:,i])/y_std
        NRMSE_list.append(NRMSE) #Append R2 of this output to the list
    NRMSE_array=np.array(NRMSE_list)
    return NRMSE_array #Return an array of R2s


########## R-squared (R2) ##########

def get_R2(y_test,y_test_pred):

    """
    Function to get R2

    Parameters
    ----------
    y_test - the true outputs (a matrix of size number of examples x number of outputs)
    y_test_pred - the predicted outputs (a matrix of size number of examples x number of outputs)

    Returns
    -------
    R2_array: An array of R2s for each output
    """
    y_test = np.vstack(y_test)
    y_test_pred = np.vstack(y_test_pred)

    R2_list=[] #Initialize a list that will contain the R2s for all the outputs
    for i in range(y_test.shape[1]): #Loop through outputs
        #Compute R2 for each output
        y_mean=np.mean(y_test[:,i])
        if np.sum((y_test[:,i]-y_mean)**2)==0:
            R2=np.nan
        else:
            R2=1-np.sum((y_test_pred[:,i]-y_test[:,i])**2)/np.sum((y_test[:,i]-y_mean)**2)
        R2_list.append(R2) #Append R2 of this output to the list
    R2_array=np.array(R2_list)
    return R2_array #Return an array of R2s




########## Pearson's correlation (rho) ##########

def get_rho(y_test,y_test_pred):

    """
    Function to get Pearson's correlation (rho)

    Parameters
    ----------
    y_test - the true outputs (a matrix of size number of examples x number of outputs)
    y_test_pred - the predicted outputs (a matrix of size number of examples x number of outputs)

    Returns
    -------
    rho_array: An array of rho's for each output
    """

    y_test = np.vstack(y_test)
    y_test_pred = np.vstack(y_test_pred)

    rho_list=[] #Initialize a list that will contain the rhos for all the outputs
    for i in range(y_test.shape[1]): #Loop through outputs
        #Compute rho for each output
        y_mean=np.mean(y_test[:,i])
        rho=np.corrcoef(y_test[:,i].T,y_test_pred[:,i].T)[0,1]
        rho_list.append(rho) #Append rho of this output to the list
    rho_array=np.array(rho_list)
    return rho_array #Return the array of rhos
