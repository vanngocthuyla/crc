import pandas as pd
import warnings
import numpy as np
import sys
import os
import shutil
import argparse

import pickle
import arviz as az
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
from jax import random
import jax.random as random

import numpyro
import numpyro.distributions as dist
from numpyro.distributions import LogNormal, Normal, Uniform
from numpyro.infer import MCMC, NUTS

from _bdrc_model import f_curve_vec, drc_fitting_control, plot_drc


def bdrc_numpyro_fitting(rng_key, data, niters=10000, nburn=2000, nchain=4, nthin=10, 
                         name=None, OUT_DIR=None):
    """
    Fitting the dose-response curve with available information of control

    Parameters:
    ----------
    rng_key : integer,   random key for numpyro model
    data    : list of information of experiment: x, y, c1, c2, n_obs.
                    x is vector of log10 of concentration
                    y is vector of response
                    c1 is control at bottom. c1 can be vector or None or 'None'
                    c2 is control at top. c2 can be vector or None or 'None'
                    n_obs is vector of nummber of replication at each concentration
                    x and y must be same length 
    niters  : optional,  integer,number of iterations
    nburn   : optional,  integer,number of warm-up
    nchain  : optional,  integer,number of sampling chains
    nthin   : optional,  integer,number of thinning
    name    : optional,  string, name of experiment
    OUT_DIR : optional,  string, output directory. 
              If OUT_DIR is declared, saving trace under .pickle file and plot of trace
    ----------
    return sampling trace and plot of trace 
    """

    x = jnp.array(data['x'])
    response = jnp.array(data['y'])
    
    if data['c1'] is None or data['c1'] == 'None':
        positive = None
    else: 
        positive = jnp.array(data['c1'])        
    
    if  data['c2'] is None or data['c2'] == 'None':
        negative = None        
    else: 
        negative = jnp.array(data['c2'])

    assert np.sum(x>0)==0, "LogConcentration should be lower than 0"
    
    kernel = NUTS(drc_fitting_control)
    nuts = MCMC(kernel, num_warmup=nburn, num_samples=niters, num_chains=nchain, 
                thinning=nthin, progress_bar=False)
    nuts.run(rng_key, x, response, positive, negative)
    nuts.print_summary()

    trace = nuts.get_samples(group_by_chain=False)

    if OUT_DIR is not None:
        pickle.dump(trace, open(os.path.join(OUT_DIR, 'traces.pickle'), "wb"))

        sample = az.convert_to_inference_data(nuts.get_samples(group_by_chain=True))
        az.plot_trace(sample)
        plt.tight_layout();
        plt.savefig(os.path.join(OUT_DIR,'trace_plot'))
        plt.ioff()

        theta = np.zeros(4)
        params = ["R_b", "R_t", "x_50", "H"]
        for i in range(len(params)): 
            theta[i] = np.mean(trace[params[i]])

        plot_drc(data, theta, title_name=name, outfile=os.path.join(OUT_DIR, 'drc_plot'))
    else: 
        sample = az.convert_to_inference_data(nuts.get_samples(group_by_chain=True))
        az.plot_trace(sample)
        return trace


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
                x is vector of log10 of concentration
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


def one_expt(rng_key, expt, niters=10000, nburn=2000, nchain=4, nthin=10, 
             name=None, OUT_DIR=None):
    """
    This functions fits numpyro model based on available information of the experiment 

    Parameters:
    ----------
    rng_key : integer,   random key for numpyro model
    expt    : list of experiment information: x, y, c1, c2.
                x is vector of log10 of concentration
                y is vector of response
                c1 is control at bottom. c1 can be vector or None or 'None'
                c2 is control at top. c2 can be vector or None or 'None'
                n_obs is vector of nummber of replication at each concentration
                x and y must be same length 
    niters  : optional,  integer,number of iterations
    nburn   : optional,  integer,number of warm-up
    nchain  : optional,  integer,number of sampling chains
    nthin   : optional,  integer,number of thinning
    name    : optional,  string, name of the experiment
    OUT_DIR : optional,  string, output directory. 
              If OUT_DIR is declared, saving trace under .pickle file and plot of trace
    ----------
    return [theta, std, variances]
    
    """
    bdrc_numpyro_fitting(rng_key, expt, niters=niters, nburn=nburn, nchain=nchain, nthin=nthin,
                         name=name, OUT_DIR=OUT_DIR)
    
    traces = pickle.load(open(os.path.join(OUT_DIR, 'traces.pickle'), "rb"))
    theta = [np.mean(traces[x]) for x in ["R_b", "R_t", "x_50", "H"]]
    std = [np.std(traces[x]) for x in ["R_b", "R_t", "x_50", "H"]]
    sigma_c_2 = (np.mean(np.exp(traces['log_sigma_c'])))**2
    
    if expt['c1'] is None or expt['c1']=='None':
        sigma_b_2 = 'None'
    else: 
        sigma_b_2 = (np.mean(np.exp(traces['log_sigma_b'])))**2       
    
    if expt['c2'] is None or expt['c2']=='None':
        sigma_t_2 = 'None'
    else: 
        sigma_t_2 = (np.mean(np.exp(traces['log_sigma_t'])))**2
    
    variances = [sigma_c_2, sigma_b_2, sigma_t_2]
    return [theta, std, variances]


def extract_theta_std_variance(data, DATA_DIR): 
    """
    Loading all the trace file in DATA_DIR directory
    Returning theta, standard deviance for 4 parameters of dose-response curve, together with variance of curves and controls
    """
    traces = pickle.load(open(os.path.join(DATA_DIR, 'traces.pickle'), "rb"))
    theta = [np.mean(traces[x]) for x in ["R_b", "R_t", "x_50", "H"]]
    std = [np.std(traces[x]) for x in ["R_b", "R_t", "x_50", "H"]]
    
    if (data['c1'] is None and data['c2'] is None) or (data['c1']=='None' and data['c2']=='None'):
        variances = [np.mean(np.exp(traces['log_sigma_c']))**2]        
    else:
        variances = [np.mean(np.exp(traces[x]))**2 for x in ["log_sigma_c", "log_sigma_b", "log_sigma_t"]]
    
    return [theta, std, variances]


def one_expt_outlier_detection(rng_key, expt, outlier_detection=False, 
                               niters=10000, nburn=2000, nchain=4, nthin=10, 
                               name=None, OUT_DIR=None):
    """
    This functions fits numpyro model based on available information of the experiment.
    The values of 4 paramters and properly the variance sigma**2 of the curve estimated by bayesian model are used for outlier detection.
    If there is any outlier(s), old results of fitting would be placed in subfolder. Outlier(s) was removed before re-fitting. 

    Parameters:
    ----------
    rng_key : integer,   random key for numpyro model
    expt    : list of experiment information: x, y, c1, c2.
                x is vector of log10 of concentration
                y is vector of response
                c1 is control at bottom. c1 can be vector or None or 'None'
                c2 is control at top. c2 can be vector or None or 'None'
                n_obs is vector of nummber of replication at each concentration
                x and y must be same length 
    outlier_detection: boolean, if outlier_detection=True, detect the outlier(s) after fitting
    niters  : optional,  integer,number of iterations
    nburn   : optional,  integer,number of warm-up
    nchain  : optional,  integer,number of sampling chains
    nthin   : optional,  integer,number of thinning
    name    : optional,  string, name of the experiment
    OUT_DIR : optional,  string, output directory. 
              If OUT_DIR is declared, saving trace under .pickle file and plot of trace
    ----------
    return [theta, std, variances, outlier_position]
    
    """ 

    [theta, std, variance] = one_expt(rng_key=rng_key, expt=expt, niters=niters, 
                                      nburn=nburn, nchain=nchain, nthin=nthin, 
                                      name=name, OUT_DIR=OUT_DIR)

    if outlier_detection: 
        #check outliers
        [outlier_position, y_update, x_update, n_obs_update] = check_outlier(expt, theta)#, np.sqrt(variance[0]))
        if len(outlier_position)>0: 
            expt['y'] = y_update
            expt['x'] = x_update
            expt['n_obs'] = n_obs_update
            os.mkdir(os.path.join(OUT_DIR, 'no_outlier_detection'))
            for file in ['traces.pickle', 'trace_plot.png', 'drc_plot.png']:
                shutil.move(OUT_DIR+'/'+file, os.path.join(OUT_DIR,'no_outlier_detection',file))
            print("Detect outlier(s) and fit again.")
            [theta, std, variance] = one_expt(rng_key=rng_key, expt=expt, niters=niters, 
                                              nburn=nburn, nchain=nchain, nthin=nthin, 
                                              name=name, OUT_DIR=OUT_DIR)
    else:
        outlier_position = None

    return [theta, std, variance, outlier_position]


def multi_expt(experiments, rng_key, niters=10000, nburn=2000, nchain=4, nthin=10, 
               outlier_detection=False, OUT_DIR=None, file_name_to_save=None): 
    """
    This functions fits numpyro model based on available information of the experiment. 
    IDs of experiments not able to be fit would be save in problem set. Experiments with only one concentration would be noticed as well. 
    
    Parameters:
    ----------
    experiments: list of experiments that were need to be fit
    rng_key : integer,   random key for numpyro model
    data    : list of experiment information: x, y, c1, c2.
                x is vector of log10 of concentration
                y is vector of response
                c1 is control at bottom. c1 can be vector or None or 'None'
                c2 is control at top. c2 can be vector or None or 'None'
                n_obs is vector of nummber of replication at each concentration
                x and y must be same length 
    outlier_detection:   boolean, if outlier_detection=True, detect the outlier(s) after fitting
    niters  : optional,  integer,number of iterations
    nburn   : optional,  integer,number of warm-up
    nchain  : optional,  integer,number of sampling chains
    nthin   : optional,  integer,number of thinning
    name    : optional,  string, name of the experiment
    OUT_DIR : optional,  string, output directory. 
              If OUT_DIR is declared, saving trace under .pickle file and plot of trace
    file_name_to_save:   name of final result file
    ----------
    """
    if OUT_DIR is None:
        OUT_DIR = os.getcwd()
    else:
        os.chdir(OUT_DIR)
    
    if file_name_to_save is None:
        file_name_to_save = 'CVD_Final'

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
    
    M = 2 #len(raw_array)

    CVD_init = pd.DataFrame([id_array, n_obs_array, raw_array, logConc_array, neg_control_array, pos_control_array], 
                            index=['ID', 'n_obs', 'y', 'x', 'c2', 'c1']).T
    CVD = pd.DataFrame([id_array, name_array, covalent, n_obs_array, raw_array, logConc_array, neg_control_array, pos_control_array, np.zeros(M), np.zeros(M), np.zeros(M), np.zeros(M)], 
                       index=['ID', 'Molecular_Name', 'Covalent Warhead', 'n_obs', 'y', 'LogConcentration', 'Negative', 'Positive', 'theta', 'ASE', 'Variance', 'Outlier_Pos']).T
    
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
            if not os.path.exists(name): 
                os.mkdir(name)

            if not os.path.exists(os.path.join(OUT_DIR, name,'traces.pickle')):
                print("Running", name)
                if outlier_detection:
                    [theta, std, variance, outlier_position] = one_expt_outlier_detection(rng_key=rng_key, expt=dat, 
                                                                                          niters=niters, nburn=nburn, nchain=nchain, nthin=nthin, 
                                                                                          name=name, OUT_DIR=os.path.join(OUT_DIR, name))
                else:
                    [theta, std, variance] = one_expt(rng_key=rng_key, expt=dat, 
                                                      niters=niters, nburn=nburn, nchain=nchain, nthin=nthin, 
                                                      name=name, OUT_DIR=os.path.join(OUT_DIR, name))
                    outlier_position = 'None'
            else:
                print("Files exist. Loading from traces.")
                [theta, std, variance] = extract_theta_std_variance(data=dat, DATA_DIR=os.path.join(OUT_DIR, name))
                outlier_position = 'None'

            CVD['theta'][i] = np.array(theta)
            CVD['ASE'][i] = np.array(std)
            CVD['Variance'][i] = np.array(variance)
            CVD['Outlier_Pos'][i] = outlier_position

            # except:
            #     print(f'Problem with {i}')
            #     problem_set.append(i)
    
    if len(unique_conc)>0:
        print("There are", len(unique_conc), "experiment with only 1 inhibitor concentration.")

    CVD.to_csv(os.path.join(OUT_DIR, file_name_to_save+'.csv'))
    pickle.dump(CVD.to_dict(), open(os.path.join(OUT_DIR, file_name_to_save+'.pickle'), "wb"))
    return CVD, problem_set


def scaling_data(ydata, negative, positive):
    min_y = min(negative, positive)
    max_y = max(negative, positive)
    return (ydata-min_y)/abs(max_y - min_y)*100