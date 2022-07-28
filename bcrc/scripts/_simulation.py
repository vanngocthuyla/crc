import numpy as np
import pandas as pd

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


def generate_one_curve(logConc, theta, no_replication, var_noise):
    """
    This function is used to generate dataset of concentration-response curve
    
    Parameters:
    ----------
    logConc       : vector, series of log of concentration of inhibitors
    theta         : vector, parameters (bottom response, top response, logIC50, hill slope)
    no_replication: integer, number of replication for each concentration
    var_noise     : float, noise added to response
    ----------
    return array of response
    """
    x = logConc
    n = len(x)
    [R_b, R_t, x_50, H] = theta
    mean_y = R_b+(R_t-R_b)/(1+10**(x*H-x_50*H))
    total_n = n*no_replication
    noise = np.random.normal(loc=0, scale=np.sqrt(var_noise), size=total_n)
    input = np.repeat(logConc, repeats=no_replication)
    output = np.repeat(mean_y, repeats=no_replication)+noise
    experiments = pd.DataFrame([input, output], ['x','y']).T
    return experiments


def generate_drc_dataset(B, logConc, theta, no_replication, var_noise, 
                         var_control, size_of_control, 
                         seed=None, scaling=False, pos_incomplete=None):
    """
    This function is used to generate multiple datasets of concentration-response curve.
    If pos_incomplete 
    
    Parameters:
    ----------
    B             : number of simulation
    logConc       : vector , series of log of concentration of inhibitors
    theta         : vector , parameters (bottom response, top response, logIC50, hill slope)
    no_replication: integer, number of replication for each concentration
    var_noise     : float  , noise added to response
    var_control   : vector , variance of two controls
    seed          : integer, optional, used to for random.seed
    scaling       : boolean, optional, scaling data or not
    pos_incomplete: vector, optional, removed position to generate incomplete curve
    ----------
    return array of response
    """
    if seed==None: 
        np.random.seed(0)
    else: 
        np.random.seed(seed)
    
    id_array = []
    x_array = []
    raw_array = []
    neg_control_array = []
    pos_control_array = []

    for i in range(B): 
        dat = generate_one_curve(logConc, theta=theta, no_replication=no_replication, var_noise=var_noise)
        c1 = np.random.normal(loc=theta[0], scale=np.sqrt(var_control), size=size_of_control)
        c2 = np.random.normal(loc=theta[1], scale=np.sqrt(var_control), size=size_of_control)

        if not pos_incomplete==None: 
            x_update = np.delete(np.asarray(dat['x']), pos_incomplete)
            y_update = np.delete(np.asarray(dat['y']), pos_incomplete)
        else: 
            x_update = dat['x']
            y_update = dat['y']

        if scaling == True:
            y_update = scaling_data(np.asarray(y_update), np.mean(c2), np.mean(c1))
            c2_update = scaling_data(c2, np.mean(c2), np.mean(c1))
            c1_update = scaling_data(c1, np.mean(c2), np.mean(c1))
        else:
            c2_update = c2
            c1_update = c1
        
        id_array.append(str(i)+'_sim')
        x_array.append(x_update)
        raw_array.append(y_update)
        neg_control_array.append(c2_update)
        pos_control_array.append(c1_update)

    expt = pd.DataFrame([id_array, np.array(raw_array), np.array(x_array), neg_control_array, pos_control_array], 
                        index=['ID', 'y', 'x', 'c2', 'c1']).T
    return expt