import numpy as np
from _nls_model import f_curve_vec
from _nls_model_outlier_detection import scaling_data

def plot_crc_normalizing(data, theta=None, positive=None, negative=None, outlier_pos=None, 
                         plot_control=True, data_normalizing=False, 
                         ax=None, line_color='b', line_style = '-', label_curve='Fitting',  
                         title_name=None, xlabel='$Log_{10}Concentration$', ylabel='Response', 
                         figure_size=(6.4, 4.8), dpi=80, outfile=None):
    """
    Parameters:
    ----------
    data            : list of experiment information: x, y, c1, c2, n_obs, outlier_pos.
                        x is vector of log10 of concentration
                        y is vector of response
                        c1 is control at bottom. c1 can be vector or None or 'None'
                        c2 is control at top. c2 can be vector or None or 'None'
                        n_obs is vector of nummber of replication at each concentration
                        outlier_pos is the position of outlier detected after first fitting
                        x and y must be same length
    theta           : vector, if no 'theta' in data, can using this plug-in theta
    positive        : vector, if no 'c1' in data, can using this plug-in positive control
    negative        : vector, if no 'c2' in data, can using this plug-in negative control
    outlier_pos     : vector, if no 'outlier_pos' in data, can using this plug-in outlier_pos
    plot_control    : boolean, if plot_control=False, no plotting the control despite of data has the control information
    data_normalizing: boolean, if negative and positive available, normalizing data
    ax              : if None, generating new plot
    line_color      : string, color of fitting line
    line_style      : string, line_style of fitting line
    label_curve     : string, label of fitting line
    title_name      : string, title of the plot
    xlabel, ylabel  : string, label of two axes of plot
    figure_size     : (width, height) size of plot
    dpi             : quality of plot
    outfile         : optional, string, output file for saving plot
    ----------
    return the plot of curve
    """
    #Load data
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

    if data_normalizing: 
        if len(c1)>=0 and len(c2)>=0:
            y_update = scaling_data(y, np.mean(c1), np.mean(c2))
            c1_update = scaling_data(c1, np.mean(c1), np.mean(c2))
            c2_update = scaling_data(c2, np.mean(c1), np.mean(c2))
            Rb_update = scaling_data(theta[0], np.mean(c1), np.mean(c2))
            Rt_update = scaling_data(theta[1], np.mean(c1), np.mean(c2))

            y = y_update
            c1 = c1_update
            c2 = c2_update
            theta[0] = Rb_update
            theta[1] = Rt_update
        else: 
            print("Please provide negative/positive controls if you want to normalize data.")

    if ax is None:
        plt.figure(figsize=figure_size, dpi=dpi)
        ax = plt.axes()

    conc = np.linspace(min(x), max(x), num=50)
    fitted = f_curve_vec(conc, *theta)

    ax.plot(x, y, 'k.', label='Observed Data')
    ax.plot(conc, fitted, color=line_color, linestyle=line_style, label=label_curve)
    handles, labels = ax.get_legend_handles_labels()


    if plot_control and c1 !="None" and c1 is not None:
        ax.plot(np.repeat(max(x+0.5), len(c1)), c1, 'g^', label="$c_b$")
        handles, labels = ax.get_legend_handles_labels()
    
    if plot_control and c2 !="None" and c2 is not None:
        ax.plot(np.repeat(min(x-0.5), len(c2)), c2, 'gv', label="$c_t$")
        handles, labels = ax.get_legend_handles_labels()

    if outlier_pos is not None:
        ax.plot(x[outlier_pos], y[outlier_pos], 'rx')
        handles, labels = ax.get_legend_handles_labels()
    else: 
        if 'Outlier_Pos' in data.index and data['Outlier_Pos'] is not None and data['Outlier_Pos']!='None': 
            outlier_pos = data['Outlier_Pos']
            ax.plot(x[outlier_pos], y[outlier_pos], 'rx')
            handles, labels = ax.get_legend_handles_labels()

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if title_name is not None:
        ax.set_title(title_name)

    ax.legend(handles, labels, bbox_to_anchor=(1, 1))
    plt.tight_layout();
    
    if outfile is not None: 
        plt.savefig(outfile)
        plt.ioff()
    else:
        return ax


def plot_incomplete_curve(dat, theta, plot_data=False, pos_incomplete=None, 
                          plot_control=False, plot_bend_point=False, 
                          line_color='b', linestyle = '-', line_label=None, ax=None,
                          xlabel='$Log_{10}Concentration$', ylabel='% Activity',):
    
    if ax is None: 
        fig = plt.figure(figsize=(5, 3), dpi=400)
        ax = fig.add_axes([0, 0, 1, 1])
    
    conc = np.linspace(min(dat['x']), max(dat['x']), num=50)
    fitted = f_curve_vec(conc, *theta)

    if plot_data: 
        ax.plot(dat['x'], dat['y'], 'k.', label="Observed data")
        handles, labels = ax.get_legend_handles_labels()        

    if pos_incomplete is not None:
        ax.plot(np.delete(dat['x'], pos_incomplete), np.delete(dat['y'], pos_incomplete), 'k.', label="Observed data")
        handles, labels = ax.get_legend_handles_labels()
        ax.plot(dat['x'][pos_incomplete], dat['y'][pos_incomplete], 'rx', label="Missing data")
        handles, labels = ax.get_legend_handles_labels()
        
    if plot_control is True and 'c1' in dat.index and dat['c1']!="None":
        ax.plot(np.repeat(max(dat['x'])+0.5, len(dat['c1'])), dat['c1'], 'g^', label="Control data")
        ax.plot(np.repeat(min(dat['x'])-0.5, len(dat['c2'])), dat['c2'], 'g^')
        handles, labels = ax.get_legend_handles_labels()
    
    ax.plot(conc, fitted, color=line_color, ls=linestyle, label=line_label)
    handles, labels = ax.get_legend_handles_labels()
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if plot_bend_point is True:
        [x_lower, x_upper] = bend_point_2(theta)
        ax.axvline(x_upper, color = 'k', linestyle = '--')
        ax.axvline(x_lower, color = 'k', linestyle = '--')

    if pos_incomplete is not None:
        for i in pos_incomplete: 
            ax.plot(dat['x'][i], dat['y'][i], 'rx')
    if line_label is not None: 
        ax.legend(handles, labels, bbox_to_anchor=(1.4, 0.75))
    return ax


def normalized_params(theta, ASE, bottom, top):
    """
    This function is used to normalize some parameters by mean of top and bottom control
    
    Parameters:
    ----------
    theta : vector of bottom response, top response, x50 and hill slope
    ASE   : vector of ASE of theta
    bottom: mean of vector of control on the bottom
    top   : mean of vector of control on the top
    ----------

    return vector of normalized theta and their ASE 
    """
    Rb = ufloat(theta[0], ASE[0])
    Rt = ufloat(theta[1], ASE[1])
    Rb_update = scaling_data(Rb, bottom, top)
    Rt_update = scaling_data(Rt, bottom, top)
    theta_update = np.array([Rb_update.n, Rt_update.n, theta[2], theta[3]])
    ASE_update = np.array([Rb_update.s, Rt_update.s, ASE[2], ASE[3]])
    return theta_update, ASE_update


def plot_crc_params(data, theta=None, positive=None, negative=None, outlier_pos=None, 
                    plot_control=True, data_normalizing=False, 
                    ax=None, line_color='b', line_style = '-', label_curve='Fitting',  
                    title_name=None, xlabel='$Log_{10}Concentration$', ylabel='Response', 
                    figure_size=(6.4, 4.8), dpi=80, outfile=None):
    """
    Parameters:
    ----------
    data            : list of experiment information: x, y, c1, c2, n_obs, outlier_pos.
                        x is vector of log10 of concentration
                        y is vector of response
                        c1 is control at bottom. c1 can be vector or None or 'None'
                        c2 is control at top. c2 can be vector or None or 'None'
                        n_obs is vector of nummber of replication at each concentration
                        outlier_pos is the position of outlier detected after first fitting
                        x and y must be same length
    theta           : vector, if no 'theta' in data, can using this plug-in theta
    positive        : vector, if no 'c1' in data, can using this plug-in positive control
    negative        : vector, if no 'c2' in data, can using this plug-in negative control
    outlier_pos     : vector, if no 'outlier_pos' in data, can using this plug-in outlier_pos
    plot_control    : boolean, if plot_control=False, no plotting the control despite of data has the control information
    data_normalizing: boolean, if negative and positive available, normalizing data
    ax              : if None, generating new plot
    line_color      : string, color of fitting line
    line_style      : string, line_style of fitting line
    label_curve     : string, label of fitting line
    title_name      : string, title of the plot
    xlabel, ylabel  : string, label of two axes of plot
    figure_size     : (width, height) size of plot
    dpi             : quality of plot
    outfile         : optional, string, output file for saving plot
    ----------
    return the plot of curve with information of params and their ASE
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
        if 'theta' in data.index and 'ASE' in data.index:
            theta = data['theta']
            ASE = data['ASE']
        else:
            print("Please provide the values for 4 parameters.")

    if data_normalizing: 
        if len(c1)>=0 and len(c2)>=0:
            y_update = scaling_data(y, np.mean(c1), np.mean(c2))
            c1_update = scaling_data(c1, np.mean(c1), np.mean(c2))
            c2_update = scaling_data(c2, np.mean(c1), np.mean(c2))
            theta_update, ASE_update = normalized_params(theta, ASE, np.mean(c1), np.mean(c2))

            y = y_update
            c1 = c1_update
            c2 = c2_update
            theta = theta_update
            ASE = ASE_update
        else: 
            print("Please provide negative/positive controls if you want to normalize data.")

    if ax is None:
        plt.figure(figsize=figure_size, dpi=dpi)
        ax = plt.axes()

    conc = np.linspace(min(x), max(x), num=50)
    fitted = f_curve_vec(conc, *theta)

    ax.plot(x, y, 'k.', label='Observed Data')
    ax.plot(conc, fitted, color=line_color, linestyle=line_style, label=label_curve)
    handles, labels = ax.get_legend_handles_labels()

    ax.text(0.65, 0.7,
            r'$R_b$: '+ str('%3.2f' %theta[0]) + ' $\pm$ ' + str('%3.2f' %ASE[0])
            + '\n$R_t$: '+ str('%3.2f' %theta[1]) + ' $\pm$ ' + str('%3.2f' %ASE[1]) 
            + '\n$x_{50}$: '+ str('%3.2f' %theta[2]) + ' $\pm$ ' + str('%3.2f' %ASE[2])
            + '\n$H$: ' + str('%5.3f' %theta[3]) + ' $\pm$ ' + str('%3.2f' %ASE[3]), 
        fontsize=11, transform=ax.transAxes, color='k')


    if plot_control and c1 !="None" and c1 is not None:
        ax.plot(np.repeat(max(x+0.5), len(c1)), c1, 'g^', label="$c_b$")
        handles, labels = ax.get_legend_handles_labels()
    
    if plot_control and c2 !="None" and c2 is not None:
        ax.plot(np.repeat(min(x-0.5), len(c2)), c2, 'gv', label="$c_t$")
        handles, labels = ax.get_legend_handles_labels()

    if outlier_pos is not None:
        ax.plot(x[outlier_pos], y[outlier_pos], 'rx')
        handles, labels = ax.get_legend_handles_labels()
    else: 
        if 'Outlier_Pos' in data.index and data['Outlier_Pos'] is not None and data['Outlier_Pos']!='None': 
            outlier_pos = data['Outlier_Pos']
            ax.plot(x[outlier_pos], y[outlier_pos], 'rx')
            handles, labels = ax.get_legend_handles_labels()

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if title_name is not None:
        ax.set_title(title_name)

    # ax.legend(handles, labels, bbox_to_anchor=(1, 1))
    plt.tight_layout();
    
    if outfile is not None: 
        plt.savefig(outfile)
        plt.ioff()
    else:
        return ax