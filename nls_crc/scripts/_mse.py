import pandas as pd
import warnings
import numpy as np
import sys
import os

import matplotlib
import matplotlib.pyplot as plt

def plot_MSE(theta_t, input_file, drop_first_column=True, plot_name='MSE Plot', OUT_DIR=None): 
    """
    This function is used to plot MSE values between the estimated parameters and true parameters
    
    Parameters:
    ----------
    theta_t          : vector, true values of 4 parameters
    input_file       : string, directory of .csv file from the fitting results 
    drop_first_column: bool  , keep or drop first column in .csv file 
    plot_name        : string, name of the plot
    OUT_DIR          : string, directory to save plot
    ----------
    Saving plot of MSE in OUTDIR directory

    """
    Theta_matrix = pd.read_csv(input_file)
    if drop_first_column:
        Theta_matrix = Theta_matrix.drop(Theta_matrix.columns[0], axis=1)
    
    xlabel = {'theta1':'${\Theta}_1$', 'theta2':'${\Theta}_2$', 
              'theta3':'${\Theta}_3$', 'theta4':'${\Theta}_4$'}
    fig, axes = plt.subplots(2, 2, figsize=(9, 7), sharex=False, sharey=False)
    axes = axes.flatten()

    for i, theta in enumerate(['theta1', 'theta2', 'theta3', 'theta4']):
        if theta in Theta_matrix.columns: 
            axes[i].hist(Theta_matrix[theta])
            axes[i].set_xlabel(xlabel[theta])
            axes[i].axvline(theta_t[i], color='r', label='True ${\Theta}$')
            axes[i].axvline(np.mean(Theta_matrix[theta]),color="green", label='Simulated ${\Theta}$')
            handles, labels = axes[i].get_legend_handles_labels()
            axes[i].grid(True)
        else:
            axes[i].set_visible(False)
    by_label = dict(zip(labels, handles))
    fig.legend(by_label.values(), by_label.keys(), loc='center right', bbox_to_anchor=(1.17, 0.8))
    plt.suptitle(plot_name, x=0.5, y=1.02, fontsize='x-large')
    fig.tight_layout();

    if OUT_DIR is not None:
        plt.savefig(os.path.join(OUT_DIR, plot_name), bbox_inches='tight')
    plt.ioff()
    # plt.show();

def calculating_MSE(theta_t, input_file, drop_first_column=True, n_batch=1, 
                    output_file=None, OUT_DIR=None):
    """
    This function is used to calculate MSE/RMSE between the estimated parameters and true parameters
    
    Parameters:
    ----------
    theta_t          : vector, true values of 4 parameters
    input_file       : string, directory of .csv file from the fitting results 
    drop_first_column: bool  , keep or drop first column in .csv file
    n_batch          : integer, number of batches to calculate mean and std of MSE
    output_file      : string, name of the output file
    OUT_DIR          : string, directory to save file
    ----------
    Table of MSE/RMSE for each parameter, saving in OUTDIR directory
     
    """

    assert n_batch>0 and isinstance(n_batch, int), print("The number of batches need to be positive integer.")

    Theta_matrix = pd.read_csv(input_file)
    if drop_first_column:
        Theta_matrix = Theta_matrix.drop(Theta_matrix.columns[0], axis=1)
    colname = Theta_matrix.columns
    
    MEAN = {}
    STD  = {}
    MSE  = {}
    for i, theta in enumerate(['theta1', 'theta2', 'theta3', 'theta4']):
        if theta in Theta_matrix.columns: 
            data = Theta_matrix[theta]
            n = int(len(data)/n_batch)
            data = data.values.reshape(n_batch, n)

            mse_theta = np.mean((data - theta_t[i])**2, axis=1)

            if n_batch > 1:
                MEAN[theta] = np.mean(mse_theta)
                STD[theta]  = np.std(mse_theta, ddof=1)
            else:
                MSE[theta] = mse_theta[0]
        else: 
            MEAN[theta] = None
            STD[theta]  = None
            MSE[theta]  = None

    if n_batch > 1:
        output_mean = pd.DataFrame(MEAN.items(), columns=['theta', 'mean'])
        output_std  = pd.DataFrame(STD.items(), columns=['theta', 'std'])
        output = pd.merge(output_mean[['theta', 'mean']], output_std[['theta', 'std']], on='theta')
        output.insert(len(output.columns), 'lower', output['mean']-output['std'])
        output.insert(len(output.columns), 'upper', output['mean']+output['std'])
    else:
        output = pd.DataFrame(MSE.items(), columns=['theta', 'MSE'])
    
    if OUT_DIR is not None: 
        output.to_csv(os.path.join(OUT_DIR, output_file+'.csv'))
    
    return output