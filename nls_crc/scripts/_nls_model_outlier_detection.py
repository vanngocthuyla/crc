"""
functions to fit the non-linear regression model of dose-response curve with outlier detection
"""

import pandas as pd
import numpy as np
import os

from _nls_model import f_curve_vec, parameter_estimation, parameter_estimation_control


def incremental_array(n_obs):
    """
    This function can be used while outlier detection. 
    If one of the position within vector n_obs equal to 0, that position will be removed. 
    """
    new_array = []
    new_array.append(n_obs[0])
    sum = n_obs[0]
    for i in range(1, len(n_obs)):
        sum = sum + n_obs[i]
        new_array.append(sum)
    return new_array


def check_outlier(data, theta, sigma_c=None, cut_off=2.5):
    """
    This function is used to detect outlier(s) based on data and curve fitting.
    If outlier(s) exist, they will be removed and some information in data will be updated. 

    Parameters:
    ----------
    data      : list of four element: x, y, c1, c2.
                x is vector of concentration
                y is vector of response
                c1 is control at bottom. c1 can be vector or None or 'None'
                c2 is control at top. c2 can be vector or None or 'None'
                n_obs is vector of nummber of replication at each concentration
                x and y must be same length 
    theta     : vector of 4 parameters (bottom response, top response, logIC50, hill slope)
    sigma_c   : float, optional
                sample standard deviation of curve
    cut_off   : float, optional, criteria to remove outlier(s)
    ----------

    return [outlier position, y_update, x_update, n_obs_update]
    """
    y = data['y']
    x = data['x']
    n_obs = data['n_obs']
    resid = y - f_curve_vec(x, *theta) #residual

    if sigma_c is None: 
        sigma_c = np.sqrt(np.sum((resid)**2)/len(y))

    if np.sum(abs(resid/sigma_c)>cut_off):
        filter_pos = abs(resid/sigma_c)<cut_off
        y_update = y[filter_pos]
        x_update = x[filter_pos]
        try: 
            n_obs_update = np.sum(np.split(filter_pos*1, incremental_array(n_obs)[:-1]), axis=1)
        except: 
            print('One of the position equal to 0')
            temp_n_obs = np.split(filter_pos*1, incremental_array(n_obs)[:-1])
            n_obs_update = []
            
            for i in temp_n_obs:
                n_obs_update.append(np.sum(i))
            n_obs_update = np.array(n_obs_update)
        
        outlier_position = [pos for pos, e in enumerate(filter_pos) if e==False]
        return [outlier_position, y_update, x_update, n_obs_update]
    
    else: 
        outlier_position = []
        return [outlier_position, None, None, None]


def one_expt(data, outlier_detection=False, cut_off=2.5):
    """
    Estimation of parameters for both control and no control procedures with outlier detection

    Parameters:
    ----------
    data             : list of four element: x, y, c1, c2.
                      x is vector of concentration
                      y is vector of response
                      c1 is control at bottom. c1 can be vector or None or 'None'
                      c2 is control at top. c2 can be vector or None or 'None'
                      n_obs is vector of nummber of replication at each concentration
                      x and y must be same length
    outlier_detection: bool, optional, fit the model with/without outlier detection
    cut_off          : float, optional, criteria to remove outlier(s)
    ----------

    return [theta, ASE, variance, outlier_position]
    """
    if (data['c1'] == 'None' and data['c2'] == 'None') or (data['c1'] is None and data['c2'] is None): 
        [theta, ASE, variances] = parameter_estimation(data)
    else: 
        [theta, ASE, variances] = parameter_estimation_control(data)

    if outlier_detection:

        [outlier_position, y_update, x_update, n_obs_update] = check_outlier(data, theta, np.sqrt(variances[0]))

        if len(outlier_position)>0: 
            data['y'] = y_update
            data['x'] = x_update
            data['n_obs'] = n_obs_update

            if (data['c1'] == 'None' and data['c2'] == 'None') or (data['c1'] is None and data['c2'] is None): 
                [theta, ASE, variances] = parameter_estimation(data, theta)
            else: 
                [theta, ASE, variances] = parameter_estimation_control(data, theta, variances)
        
        if data['c1']=="None" and data['c1'] is not None and len(variances)>1:
            variances = np.delete(variances, 1)
        if data['c2']=="None" and data['c2'] is not None and len(variances)>1:
            variances = np.delete(variances, 2)
    else: 
        outlier_position = 'None'

    return [theta, ASE, variances, outlier_position]


def multi_expt(experiments, outlier_detection=False, cut_off=2.5,  
               OUT_DIR=None, file_name_to_save=None):
    """
    This functions fits non-linear regression model based on available information of the experiment. 
    IDs of experiments not able to be fit would be save in problem set. Experiments with only one concentration would be noticed as well. 
    
    Parameters:
    ----------
    experiments       : list of experiments that were need to be fit
                        ID            : string, identify name of experiment
                        Molecular_Name: string, name of molecule
                        logLtot       : vector, series of log of concentration of compound
                        n_obs         : vector, nummber of replication at each concentration
                        raw_R         : vector, response
                        Negative Control : vector, control of the top
                        Positive Control : vector, control of the bottom
                        Covalent Warhead : bool
    outlier_detection : bool, optional, fit the model with/without outlier detection
    OUT_DIR           : directory for output
    file_name_to_save : name of final result file
    ----------
    
    return dataframe of multiple fitting and vector of IDs of curves that cannot be fit

    """
    if OUT_DIR is None:
        OUT_DIR = os.getcwd()
    else:
        os.chdir(OUT_DIR)

    id_array = []
    name_array = []
    logConc_array = []
    raw_array = []
    neg_control_array = []
    pos_control_array = []
    n_obs_array = []
    covalent = []

    for key in experiments.keys():
        for expt in experiments[key]:
            id_array.append(expt['ID'])
            name_array.append(expt['Molecular_Name'])
            logConc_array.append(np.repeat(expt['logLtot'], expt['n_obs'], axis=0))
            raw_array.append(expt['raw_R'])
            neg_control_array.append(expt['Negative Control'])
            pos_control_array.append(expt['Positive Control'])
            n_obs_array.append(expt['n_obs'])
            covalent.append(expt['Covalent Warhead'])
    M = len(raw_array)

    CVD_init = pd.DataFrame([id_array, n_obs_array, raw_array, logConc_array, neg_control_array, pos_control_array], 
                            index=['ID','n_obs','y', 'x', 'c2', 'c1']).T
    CVD = pd.DataFrame([id_array, name_array, covalent, n_obs_array, raw_array, logConc_array, neg_control_array, pos_control_array, np.zeros(M), np.zeros(M), np.zeros(M), np.zeros(M)], 
                       index=['ID', 'Molecular_Name', 'Covalent Warhead', 'n_obs', 'y', 'LogConcentration', 'Negative', 'Positive', 'theta', 'ASE', 'Variances', 'Outlier_Pos']).T
    
    problem_set = []
    unique_conc = []
    exper_names = list(CVD_init.ID)

    for i in range(M): 
        name = exper_names[i]
        dat = CVD_init.iloc[i]
        print(name)

        if len(np.unique(dat['x']))==1:
            unique_conc.append(i)
        else:
            # try: 
            [theta, ASE, variances, outlier_position] = one_expt(dat, outlier_detection)
            CVD['theta'][i] = theta
            CVD['ASE'][i] = ASE
            CVD['Variances'][i] = np.array(variances)
            CVD['Outlier_Pos'][i] = outlier_position
            # except:
                # print(f'Problem with {i}')
                # problem_set.append(i)
            
    if len(unique_conc)>0:
        print("There are", len(unique_conc), "experiment with only 1 inhibitor concentration.")
    
    CVD.to_csv(os.path.join(OUT_DIR, file_name_to_save+'.csv'))
    pickle.dump(CVD.to_dict(), open(os.path.join(OUT_DIR, file_name_to_save+'.pickle'), "wb"))

    return CVD, problem_set


def scaling_data(y, bottom, top):
    """
    This function is used to normalize data by mean of top and bottom control
    
    Parameters:
    ----------
    y     : vector of response
    bottom: mean of vector of control on the bottom
    top   : mean of vector of control on the top
    ----------

    return vector of normalized response 
    """
    min_y = min(bottom, top)
    max_y = max(bottom, top)
    return (y-min_y)/abs(max_y - min_y)*100