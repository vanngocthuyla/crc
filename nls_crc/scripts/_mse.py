import pandas as pd
import warnings
import numpy as np
import sys
import os
import pickle

import matplotlib
import matplotlib.pyplot as plt

from _nls_model_outlier_detection import scaling_data

def plot_MSE(theta_t, input_file, control_file=None, normalized_data=False,
             drop_first_column=True, plot_name='MSE Plot', OUT_DIR=None): 
    """
    This function is used to plot MSE values between the estimated parameters and true parameters
    
    Parameters:
    ----------
    theta_t          : vector, true values of 4 parameters
    input_file       : string, directory of .csv file from the fitting results
    control_file     : string, file contains the information of fitting curves, including both controls
    normalized_data  : bool  , theta was normalized (True) or unnormalized (False)
    drop_first_column: bool  , keep or drop first column in .csv file 
    plot_name        : string, name of the plot
    OUT_DIR          : string, directory to save plot
    ----------
    Saving plot of MSE in OUTDIR directory

    """
    Theta_matrix_init = pd.read_csv(input_file)
    if drop_first_column:
        Theta_matrix_init = Theta_matrix_init.drop(Theta_matrix_init.columns[0], axis=1)

    if not normalized_data:
        print("Scaling data to 0 to 100%.")
        try: 
            expt_dict = pickle.load(open(control_file+'.pickle', "rb"))
            expt = pd.DataFrame.from_dict(expt_dict)
        except:
            expt = pd.read_csv(control_file+'.csv', index_col=0)

        theta1_scaled = []
        theta2_scaled = []
        for i in range(len(expt)):
            c1 = expt['c1'][i]
            c2 = expt['c2'][i]
            theta1_scaled.append(scaling_data(Theta_matrix_init.theta1[i], np.mean(c1), np.mean(c2)))
            theta2_scaled.append(scaling_data(Theta_matrix_init.theta2[i], np.mean(c1), np.mean(c2)))
        Theta_matrix = pd.DataFrame([theta1_scaled, theta2_scaled, Theta_matrix_init.theta3, Theta_matrix_init.theta4], 
                                     index=['theta1', 'theta2', 'theta3', 'theta4']).T
    else:
        Theta_matrix = Theta_matrix_init

    
    xlabel = {'theta1':'$R_b$', 'theta2':'$R_t$', 
              'theta3':'$x_{50}$', 'theta4':'$H$'}
    if len(Theta_matrix.columns) > 2: 
        fig, axes = plt.subplots(2, 2, figsize=(9, 7), sharex=False, sharey=False)
    else:
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), sharex=False, sharey=False)
    axes = axes.flatten()

    # for i, theta in enumerate(['theta1', 'theta2', 'theta3', 'theta4']):
    #     if theta in Theta_matrix.columns: 
    #         axes[i].hist(Theta_matrix[theta])
    #         axes[i].set_xlabel(xlabel[theta])
    #         axes[i].axvline(theta_t[i], color='r', label='True ${\Theta}$')
    #         axes[i].axvline(np.mean(Theta_matrix[theta]),color="green", label='Simulated ${\Theta}$')
    #         handles, labels = axes[i].get_legend_handles_labels()
    #         axes[i].grid(True)
    #     else:
    #         axes[i].set_visible(False)

    for i, theta in enumerate(Theta_matrix.columns):
        axes[i].hist(Theta_matrix[theta], color="lightblue")
        axes[i].set_xlabel(xlabel[theta])
        axes[i].axvline(theta_t[np.where(np.array(['theta1', 'theta2', 'theta3', 'theta4'])==theta)], 
                        color='r', label='True ${\Theta}$', linewidth=3, linestyle='-')
        axes[i].axvline(np.mean(Theta_matrix[theta]), color="green", 
                        label='Simulated ${\Theta}$', linewidth=3, linestyle='--')
        handles, labels = axes[i].get_legend_handles_labels()
        axes[i].grid(True)
    if len(axes) > (i+1):
        axes[int(i+1)].set_visible(False)

    by_label = dict(zip(labels, handles))
    #fig.legend(by_label.values(), by_label.keys(), loc='center right', bbox_to_anchor=(1.17, 0.8))
    #plt.suptitle(plot_name, x=0.5, y=1.02, fontsize='x-large')
    fig.tight_layout();

    if OUT_DIR is not None:
        plt.savefig(os.path.join(OUT_DIR, plot_name))#, bbox_inches='tight')
    plt.ioff()
    # plt.show();

def calculating_MSE(theta_t, input_file, control_file=None, normalized_data=True, 
                    drop_first_column=True, n_batch=1, 
                    output_file=None, OUT_DIR=None):
    """
    This function is used to calculate MSE/RMSE between the estimated parameters and true parameters
    
    Parameters:
    ----------
    theta_t          : vector, true values of 4 parameters
    input_file       : string, directory of .csv file from the fitting results 
    control_file     : string, file contains the information of fitting curves, including both controls
    normalized_data  : bool  , theta was normalized (True) or unnormalized (False)
    drop_first_column: bool  , keep or drop first column in .csv file
    n_batch          : integer, number of batches to calculate mean and std of MSE
    output_file      : string, name of the output file
    OUT_DIR          : string, directory to save file
    ----------
    Table of MSE/RMSE for each parameter, saving in OUTDIR directory
     
    """

    assert n_batch>0 and isinstance(n_batch, int), print("The number of batches need to be positive integer.")

    Theta_matrix_init = pd.read_csv(input_file)
    if drop_first_column:
        Theta_matrix_init = Theta_matrix_init.drop(Theta_matrix_init.columns[0], axis=1)

    if not normalized_data:
        print("Scaling data to 0 to 100%.")
        try: 
            expt_dict = pickle.load(open(control_file+'.pickle', "rb"))
            expt = pd.DataFrame.from_dict(expt_dict)
        except:
            expt = pd.read_csv(control_file+'.csv', index_col=0)

        theta1_scaled = []
        theta2_scaled = []
        for i in range(len(expt)):
            c1 = expt['c1'][i]
            c2 = expt['c2'][i]
            theta1_scaled.append(scaling_data(Theta_matrix_init.theta1[i], np.mean(c1), np.mean(c2)))
            theta2_scaled.append(scaling_data(Theta_matrix_init.theta2[i], np.mean(c1), np.mean(c2)))
        Theta_matrix = pd.DataFrame([theta1_scaled, theta2_scaled, Theta_matrix_init.theta3, Theta_matrix_init.theta4], 
                                     index=['theta1', 'theta2', 'theta3', 'theta4']).T
    else:
        Theta_matrix = Theta_matrix_init

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