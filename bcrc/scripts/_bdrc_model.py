import pandas as pd
import warnings
import numpy as np
import sys
import os
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


def f_curve_vec(x, R_b, R_t, x_50, H):
    """
    Dose-response curve function
    
    Parameters:
    ----------
    x   : array, log_10 of concentration of inhibitor
    R_b : float, bottom response
    R_t : float, top response
    x_50: float, log_10_IC50
    H   : float, hill slope
    ----------
    return an array of response
    """
    return R_b+(R_t-R_b)/(1+10**(x*H-x_50*H))


def uniform_prior(name, lower, upper):
    """
    Parameters:
    ----------
    name: string, name of variable
    lower: float, lower value of uniform distribution
    upper: float, upper value of uniform distribution
    ----------
    return numpyro.Uniform
    """
    name = numpyro.sample(name, dist.Uniform(low=lower, high=upper))
    return name


def logsigma_guesses(response):
    """
    Parameters:
    ----------
    response: jnp.array, observed data of concentration-response dataset
    ----------
    return range of log of sigma
    """
    log_sigma_guess = jnp.log(response.std()) # jnp.log(response.std())
    log_sigma_min = log_sigma_guess - 10 #log_sigma_min.at[0].set(log_sigma_guess - 10)
    log_sigma_max = log_sigma_guess + 5 #log_sigma_max.at[0].set(log_sigma_guess + 5)
    return log_sigma_min, log_sigma_max


def prior(response, fixed_Rb = False, fixed_Rt = False):
    """
    Parameters:
    ----------
    response: jnp.array, observed data of concentration-response dataset
    fixed_Rb: boolean,   if fixed_Rb = True, fixed R_b = 0
    fixed_Rt: boolean,   if fixed_Rt = True, fixed R_t = 100
    ----------
    return prior distributions for parameters of Bayesian dose-response curve fitting
    """
    if fixed_Rb: 
        R_b = 0
    else: 
        R_b = uniform_prior("R_b", lower=-2*abs(min(response)), upper=max(response))
    
    if fixed_Rt: 
        R_t = 100
    else:
        R_t = uniform_prior("R_t", lower=min(response), upper=2*max(response))
    
    x_50 = uniform_prior("x_50", lower=-50, upper=50)
    H = uniform_prior("H", lower=0, upper=50)

    log_sigma_c_min, log_sigma_c_max = logsigma_guesses(response)
    log_sigma_c = uniform_prior("log_sigma_c", lower=log_sigma_c_min, upper=log_sigma_c_max)
    return R_b, R_t, x_50, H, log_sigma_c


def dose_response_curve(x, R_b, R_t, x_50, H):
    """
    Dose-response curve function
    
    Parameters:
    ----------
    x   : array, log_10 of concentration of inhibitor
    R_b : float, bottom response
    R_t : float, top response
    x_50: float, log_10_IC50
    H   : float, hill slope
    ----------
    return jnp.array of response
    """
    response_model = jnp.zeros([len(x)], dtype=jnp.float64)
    for n in range(len(x)): 
        response_model = jax.ops.index_add(response_model, 
                                           jax.ops.index[n], 
                                           R_b+(R_t-R_b)/(1+10**(x[n]*H-x_50*H)))
    return response_model


def drc_fitting(x, response, fixed_Rb=False, fixed_Rt=False):
    """
    Creating the Bayesian model
    ----------
    x       : array, log_10 of concentration of inhibitor
    response: jnp.array, observed data of concentration-response dataset
    fixed_Rb: boolean,   if fixed_Rb = True, fixed R_b = 0
    fixed_Rt: boolean,   if fixed_Rt = True, fixed R_t = 100
    ----------
    return an instance of numpyro.model
    """
    R_b, R_t, x_50, H, log_sigma_c = prior(response, fixed_Rb, fixed_Rt) # prior
    response_model = dose_response_curve(x, R_b, R_t, x_50, H)
    sigma_c = jnp.exp(log_sigma_c)
    numpyro.sample('response', dist.Normal(loc=response_model, scale=sigma_c), obs=response)


def prior_control(response, positive=None, negative=None, fixed_Rb=False, fixed_Rt=False):
    """
    Parameters:
    ----------
    response: jnp.array, observed data of concentration-response dataset
    positive: optional,  jnp.array of the positive control
    negative: optional,  jnp.array of the negative control
    fixed_Rb: boolean,   if fixed_Rb = True, fixed R_b = 0
    fixed_Rt: boolean,   if fixed_Rt = True, fixed R_t = 100
    ----------
    return prior distributions for parameters of Bayesian dose-response curve fitting with control
    """
    if fixed_Rb: 
        R_b = 0
    else: 
        R_b = uniform_prior("R_b", lower=-2*abs(min(response)), upper=max(response))
    
    if fixed_Rt: 
        R_t = 100
    else:
        R_t = uniform_prior("R_t", lower=min(response), upper=2*max(response))

    x_50 = uniform_prior("x_50", lower=-50, upper=50)
    H = uniform_prior("H", lower=0, upper=50)

    log_sigma_c_min, log_sigma_c_max = logsigma_guesses(response)
    log_sigma_c = uniform_prior("log_sigma_c", lower=log_sigma_c_min, upper=log_sigma_c_max)

    if positive is not None: 
        log_sigma_b_min, log_sigma_b_max = logsigma_guesses(positive)
        log_sigma_b = uniform_prior("log_sigma_b", lower=log_sigma_b_min, upper=log_sigma_b_max)
    else: 
        log_sigma_b = None

    if negative is not None:
        log_sigma_t_min, log_sigma_t_max = logsigma_guesses(negative)
        log_sigma_t = uniform_prior("log_sigma_t", lower=log_sigma_t_min, upper=log_sigma_t_max)
    else:
        log_sigma_t = None

    return R_b, R_t, x_50, H, log_sigma_c, log_sigma_b, log_sigma_t


def drc_fitting_control(x, response, positive=None, negative=None, fixed_Rb=False, fixed_Rt=False):
    """
    Creating the Bayesian model of fitting with control
    
    Parameters:
    ----------
    x       : array, log_10 of concentration of inhibitor
    response: jnp.array, observed data of concentration-response dataset
    positive: optional,  jnp.array of the positive control
    negative: optional,  jnp.array of the negative control
    fixed_Rb: boolean,   if fixed_Rb = True, fixed R_b = 0
    fixed_Rt: boolean,   if fixed_Rt = True, fixed R_t = 100
    ----------
    return an instance of numpyro.model
    """
    R_b, R_t, x_50, H, log_sigma_c, log_sigma_b, log_sigma_t = prior_control(response, positive, negative, fixed_Rb, fixed_Rt) #prior
    response_model = dose_response_curve(x, R_b, R_t, x_50, H)
    
    sigma_c = jnp.exp(log_sigma_c)
    numpyro.sample('response', dist.Normal(loc=response_model, scale=sigma_c), obs=response)
    
    if log_sigma_b is not None:
        sigma_b = jnp.exp(log_sigma_b)
        numpyro.sample('positive', dist.Normal(loc=R_b, scale=sigma_b), obs=positive)
    
    if log_sigma_t is not None:
        sigma_t = jnp.exp(log_sigma_t)
        numpyro.sample('negative', dist.Normal(loc=R_t, scale=sigma_t), obs=negative)


def bdrc_numpyro_fitting(rng_key, data, fitting_by_control=False, 
                         fixed_Rb=False, fixed_Rt=False, 
                         niters=10000, nburn=2000, nchain=4, nthin=1, 
                         name=None, OUT_DIR=None):
    """
    Fitting the dose-response curve

    Parameters:
    ----------
    rng_key : integer,   random key for numpyro model
    data    : list of information of experiment: x, y, c1, c2, n_obs.
                    x is vector of log10 of concentration
                    y is vector of response
                    c1 is control at bottom. c1 can be vector or None or 'None'
                    c2 is control at top. c2 can be vector or None or 'None'
                    n_obs is vector of nummber of replication at each concentration
                    x and y must be same length response: jnp.array, observed data of concentration-response dataset
    fitting_by_control:  boolean, if fitting_by_control, fitting the dose-response curve with control
    fixed_Rb: boolean,   if fixed_Rb = True, fixed R_b = 0
    fixed_Rt: boolean,   if fixed_Rt = True, fixed R_t = 100
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

    assert np.sum(x>0)==0, "LogConcentration should be lower than 0"
            
    if fitting_by_control: 
        if data['c1'] is None or data['c1'] == 'None':
            positive = None
        else: 
            positive = jnp.array(data['c1'])        
        
        if data['c2'] is None or data['c2'] == 'None':
            negative = None        
        else: 
            negative = jnp.array(data['c2'])

        kernel = NUTS(drc_fitting_control)
        nuts = MCMC(kernel, num_warmup=nburn, num_samples=niters, num_chains=nchain, 
                    thinning=nthin, progress_bar=False)
        nuts.run(rng_key, x, response, positive, negative, fixed_Rb, fixed_Rt)
        nuts.print_summary()
    
    else:
        kernel =  NUTS(drc_fitting)
        nuts = MCMC(kernel, num_warmup=nburn, num_samples=niters, num_chains=nchain, 
                    thinning=nthin, progress_bar=False)
        nuts.run(rng_key, x, response, fixed_Rb, fixed_Rt)
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
        params = ['R_b', 'R_t', 'x_50', 'H']
        
        if fixed_Rb:
            theta[0] = 0
        else: 
            theta[0] = np.mean(trace['R_b'])
        
        if fixed_Rt:
            theta[1] = 100
        else: 
            theta[1] = np.mean(trace['R_t'])

        for i, var in enumerate(params):
            if var in ['x_50', 'H']: 
                theta[i] = np.mean(trace[var])

        plot_drc(data=data, theta=theta, plot_control=fitting_by_control*False, 
                 title_name=name, outfile=os.path.join(OUT_DIR, 'drc_plot'))
    else: 
        sample = az.convert_to_inference_data(nuts.get_samples(group_by_chain=True))
        az.plot_trace(sample)
        return trace


def plot_drc(data, theta=None, positive=None, negative=None, outlier_pos=None, plot_control=True,
             ax=None, line_color='b', line_style = '-', label_curve='Fitting',  
             title_name=None, xlabel='$Log_{10}Concentration$', ylabel='Response', 
             figure_size=(6.4, 4.8), dpi=80, outfile=None):
    """
    Parameters:
    ----------
    data          : list of experiment information: x, y, c1, c2, n_obs, outlier_pos.
                      x is vector of log10 of concentration
                      y is vector of response
                      c1 is control at bottom. c1 can be vector or None or 'None'
                      c2 is control at top. c2 can be vector or None or 'None'
                      n_obs is vector of nummber of replication at each concentration
                      outlier_pos is the position of outlier detected after first fitting
                      x and y must be same length
    theta         : vector, if no 'theta' in data, can using this plug-in theta
    positive      : vector, if no 'c1' in data, can using this plug-in positive control
    negative      : vector, if no 'c2' in data, can using this plug-in negative control
    outlier_pos   : vector, if no 'outlier_pos' in data, can using this plug-in outlier_pos
    plot_control  : boolean, if plot_control=False, no plotting the control despite of data has the control information
    ax            : if None, generating new plot
    line_color    : string, color of fitting line
    line_style    : string, line_style of fitting line
    label_curve   : string, label of fitting line
    title_name    : string, title of the plot
    xlabel, ylabel: string, label of two axes of plot
    figure_size   : (width, height) size of plot
    dpi           : quality of plot
    outfile       : optional, string, output file for saving plot
    ----------
    return the plot of curve
    """
    try: 
        y = data['y']
    except:
        y = data['response']

    try:
        x = data['x']
    except: 
        x = data['LogConcentration']

    if positive is None:
        if 'c1' in data.index or 'Positive' in data.index:
            try: 
                c1 = data['c1']
            except: 
                c1 = data['Positive']
        else:
            c1 = None
    else:
        c1 = positive

    if negative is None:
        if 'c2' in data.index or 'Negative' in data.index:
            try: 
                c2 = data['c2']
            except: 
                c2 = data['Negative']
        else:
            c2 = None
    else: 
        c2 = negative

    if theta is None: 
        if 'theta' in data.index:
            theta = data['theta']
        else:
            print("Please provide the values for 4 parameters.")

    if ax is None:
        plt.figure(figsize=figure_size, dpi=dpi)
        ax = plt.axes()
    
    conc = np.linspace(min(x), max(x), num=50)
    fitted = f_curve_vec(conc, *theta)

    ax.plot(x, y, 'k.', label='Observed Data')
    ax.plot(conc, fitted, color=line_color, linestyle=line_style, label=label_curve)


    if plot_control and c1 !="None" and c1 is not None:
        ax.plot(np.repeat(max(x+0.5), len(c1)), c1, 'g^', label="$c_b$")
    
    if plot_control and c2 !="None" and c2 is not None:
        ax.plot(np.repeat(min(x-0.5), len(c2)), c2, 'gv', label="$c_t$")

    if outlier_pos is not None:
        ax.plot(x[outlier_pos], y[outlier_pos], 'rx')
    else: 
        if 'Outlier_Pos' in data.index and data['Outlier_Pos'] is not None and data['Outlier_Pos']!='None': 
            outlier_pos = data['Outlier_Pos']
            ax.plot(x[outlier_pos], y[outlier_pos], 'rx')

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if title_name is not None:
        ax.set_title(title_name)

    plt.legend()
    plt.tight_layout();
    
    if outfile is not None: 
        plt.savefig(outfile)
        plt.ioff()
    else:
        return ax


def bend_point(theta):
    """
    Return the bend_point of dose-response curve
    Check Sebaugh, J. L. Pharmaceutical Statistics 2011 for more information
    """
    k = 4.6805
    b = theta[3]
    c = 10**(theta[2])
    x_lower = c*(1/k)**(1/b)
    x_upper = c*(k)**(1/b)
    return [np.log10(x_lower), np.log10(x_upper)]


def plot_incomplete_curve(dat, theta, pos_incomplete=None, plot_control=False, plot_bend_point=False, 
                          line_color='b', line_style='-', label_curve='Fitting', plt_name=None, ax=None, 
                          xlable='$Log_{10}Concentration$', ylabel='Response', 
                          figure_size=(5, 3), dpi=80, legend_position=(1.4, 0.75), outfile=None):
    
    """
    Dose-reponse curve plotting
    """
    if ax is None: 
        fig = plt.figure(figsize=figure_size, dpi=dpi)
        ax = fig.add_axes([0, 0, 1, 1])

    if theta is None: 
        if 'theta' in data.index:
            theta = data['theta']
        else:
            print("Please provide the values for 4 parameters.")

    conc = np.linspace(min(dat['x']), max(dat['x']), num=50)
    fitted = f_curve_vec(conc, *theta)

    if pos_incomplete is not None:
        ax.plot(np.delete(dat['x'], pos_incomplete), np.delete(dat['y'], pos_incomplete), 'k.', label="Observed data")
        handles, labels = ax.get_legend_handles_labels()
        ax.plot(dat['x'][pos_incomplete], dat['y'][pos_incomplete], 'rx', label="Missing data")
        handles, labels = ax.get_legend_handles_labels()
    else:
        ax.plot(dat['x'], dat['y'], 'k.', label="Observed data")
        handles, labels = ax.get_legend_handles_labels()
        
    if plot_control is True and 'c1' in dat.index and dat['c1']!="None":
        ax.plot(np.repeat(max(dat['x'])+0.5, len(dat['c1'])), dat['c1'], 'g^', label="$c_b$")
        handles, labels = ax.get_legend_handles_labels()

    if plot_control is True and 'c2' in dat.index and dat['c2']!="None":
        ax.plot(np.repeat(min(dat['x'])-0.5, len(dat['c2'])), dat['c2'], 'gv', label="$c_t$")
        handles, labels = ax.get_legend_handles_labels()
    
    ax.plot(conc, fitted, color=line_color, ls=line_style, label=label_curve)
    handles, labels = ax.get_legend_handles_labels()

    if plot_bend_point:
        [x_lower, x_upper] = bend_point(theta)
        ax.axvline(x_upper, color = 'k', linestyle = '--')
        ax.axvline(x_lower, color = 'k', linestyle = '--')

    if pos_incomplete is not None:
        for i in pos_incomplete: 
            ax.plot(dat['x'][i], dat['y'][i], 'rx')
    
    ax.legend(handles, labels, bbox_to_anchor=legend_position)
    ax.set_xlabel(xlable)
    ax.set_ylabel(ylabel)

    if plt_name is not None:
        ax.set_title(plt_name)

    plt.tight_layout();
    
    if out is not None: 
        plt.savefig(outfile)
        plt.ioff()
    else:
        return ax