"""
functions to fit the non-linear regression model of dose-response curve
"""

import numpy as np
from scipy.optimize import minimize, curve_fit


def f_curve_vec(x, R_b, R_t, x_50, H):
    """
    Dose-response curve function
    
    Parameters:
    ----------
    x   : array
          log_10 of concentration of inhibitor
    R_b : float
          bottom response
    R_t : float
          top response
    x_50: float
          logIC50
    H   : float
          hill slope
    ----------
    return an array of response
    """
    return R_b+(R_t-R_b)/(1+10**(x*H-x_50*H))
    

def grad_f_vec(x, theta):
    """
    first derivatives of response by theta
    Parameters:
    ----------
    x    : concentration
    theta: vector of 4 parameters (bottom response, top response, logIC50, hill slope)
    ----------
    """
    [R_b, R_t, x_50, H] = theta
    N = len(x)
    grad = np.reshape(np.zeros(N*4), (4, N))
    temp = 10**(x*H-x_50*H)
    grad[1,:] = 1/(1+temp)
    grad[0,:] = 1-grad[1,:]
    grad[2,:] = np.log(10)*(R_t-R_b)*H*temp*(grad[1,:])**2
    grad[3,:] = -np.log(10)*(R_t-R_b)*(x-x_50)*temp*(grad[1,:])**2
    return grad


def hessian_f(x, theta):
    """
    This function returns the hessian matrix of f function for x=x_i

    Parameters:
    ----------
    x    : concentration
    theta: vector of 4 parameters (bottom response, top response, logIC50, hill slope)
    ----------
    """
    [R_b, R_t, x_50, H] = theta
    Hf = np.reshape(np.zeros(16), (4, 4))
    temp1 = R_b - R_t
    temp2 = H*x_50 - H*x
    temp3 = 10**(temp2)
    temp4 = (1+temp3)**2
    temp5 = (1+temp3)**3
    
    Hf[2,2] = (np.log(10))**2*temp1*(H**2)*temp3*(temp3-1)/temp5
    Hf[3,3] = Hf[2,2]*(x_50-x)**2/(H**2)
    Hf[0,2] = -np.log(10)*H*temp3/temp4
    Hf[2,0] = Hf[0,2]
    Hf[0,3] = Hf[0,2]*(x_50-x)/H
    Hf[3,0] = Hf[0,3]
    Hf[1,2] = -Hf[0,2]
    Hf[2,1] = Hf[1,2]
    Hf[1,3] = -Hf[0,3]
    Hf[3,1] = Hf[1,3]
    Hf[2,3] = -np.log(10)*temp1*temp3/temp5*(temp2*np.log(100)-(temp3+1)*(temp2*np.log(10)-1))
    Hf[3,2] = Hf[2,3]
    return Hf


def Moore_Penrose_discard_zero(A):
    U, S, VT = np.linalg.svd(A, full_matrices=False)
    threshold = np.finfo(float).eps * max(A.shape) * S[0]
    S = S[S > threshold]
    VT = VT[:S.size]
    pcov = np.dot(VT.T/S**2, VT)
    return pcov


def nls(theta, variances, bounds, data, itnmax=100, tol=1e-4):
    """
    Fitting non-linear regression without control
    Parameters:
    ----------
    theta     : vector of 4 parameters (bottom response, top response, logIC50, hill slope)
    variances : vector (sigma_c**2, sigma_b**2, sigma_t**2) 
                variance of the data on the curve, controls at bottom and controls at top
    bounds    : (lower, upper)
                boundary for scipy.minimize or scipy.curve_fit
    data      : list of two element: x, y.
                x is vector of concentration
                y is vector of response
                x and y must be same length 
    itnmax    : int, option
                maximum number of iterations for minimization
    tol       : float, option
                tolerance value for minimization
    ----------
    return [theta, ASE, variance]
    """
    assert len(data['x'])==len(data['y']), "Length of concentration and response must be the same."
    N = len(data['x'])

    ## Define the R function which is in replacement of the logL for the MLE
    def R_fun(theta):
        f_fit = f_curve_vec(data['x'], *theta)
        part1 = np.sum((data['y']-f_fit)**2)/(2*variances[0])
        return part1

    def hessian_Q(theta):
        N = len(data['y'])
        grad = grad_f_vec(data['x'], theta)
        resid = data['y'] - f_curve_vec(data['x'], *theta)
        part1 = np.reshape(np.zeros(16), (4, 4))
        part2 = np.reshape(np.zeros(16), (4, 4))
        for i in range(N):
            temp = grad[:,i]
            temp = temp.reshape(4, 1)
            part1 = part1 + np.matmul(temp, temp.T)
            part2 = part2 + resid[i]*hessian_f(data['x'][i], theta)
        HQ = part1-part2
        HR = HQ/variances[0]
        return HR
  
    ## optimize R_fun with current variances value
    itern = 1
    obj1 = R_fun(theta)+N/2*np.log(variances[0])
    obj0 = obj1+1
    while (itern <= itnmax and abs(obj0-obj1)>=tol):
        obj0 = obj1
        optsol = minimize(R_fun, x0=theta, bounds=bounds, #method ='trust-constr',  
                          options={'disp': False, 'maxiter': 10000})
        theta = optsol.x
        ## update the variances 
        resid = data['y'] - f_curve_vec(data['x'], *theta)
        variances[0] = np.mean(resid**2)
      
        ## update the loglikelihood function
        obj1 = R_fun(theta)+N/2*np.log(variances[0])
        itern = itern + 1

    try: 
        cov_mle = np.linalg.inv(hessian_Q(theta))
    except: 
        print("Singular matrix while extracting cov_mle!")
        # A = hessian_Q(theta)
        # U, S, VT = np.linalg.svd(A, full_matrices=False)
        # threshold = np.finfo(float).eps*max(A.shape)*S[0]
        # S = S[S > threshold]
        # VT = VT[:S.size]
        # cov_mle = np.dot(VT.T / S**2, VT)
        cov_mle = Moore_Penrose_discard_zero(hessian_Q(theta))
    
    ASE = np.sqrt(np.diag(cov_mle))

    #return optsol, ASE
    # mle = [theta, -(obj1)/(N), itern-1, cov_mle, ASE, variances] #theta, LogL, iterations, col_mle, ASE, variances
    mle = [theta, ASE, variances]
    return mle


def nls_control(theta, variances, bounds, data, control, itnmax=100, tol=1e-4):
    """
    Fitting non-linear regression with control
    Parameters:
    ----------
    theta     : vector of 4 parameters (bottom response, top response, logIC50, hill slope)
    variances : vector (sigma_c**2, sigma_b**2, sigma_t**2) 
                variance of the data on the curve, controls at bottom and controls at top
    bounds    : (lower, upper)
                boundary for scipy.minimize or scipy.curve_fit
    data      : list of four element: x, y, c1, c2.
                x is vector of concentration
                y is vector of response
                c1 is control at bottom. c1 can be vector or None or 'None'
                c2 is control at top. c2 can be vector or None or 'None'
                x and y must be same length 
    itnmax    : int, option
                maximum number of iterations for minimization
    tol       : float, option
                tolerance value for minimization
    ----------
    return [theta, ASE, variance]
    """
    assert len(data['x'])==len(data['y']), "Length of concentration and response must be the same."

    N = len(data['x'])
    L1 = 0
    L2 = 0
    if 'c1' in control.index and control['c1']!="None" and control['c1'] is not None:
        L1 = len(control['c1'])
    if 'c2' in control.index and control['c2']!="None" and control['c2'] is not None:
        L2 = len(control['c2'])
  
    ## Define the R function which is in replacement of the logL for the MLE
    def R_fun(theta):
        f_fit = f_curve_vec(data['x'], *theta)
        part2 = 0
        part3 = 0
        if 'c1' in control.index and control['c1']!="None" and control['c1'] is not None:
            part2 = np.sum((control['c1']-theta[0])**2)/(2*variances[1])
        if 'c2' in control.index and control['c2']!="None" and control['c2'] is not None:
            part3 = np.sum((control['c2']-theta[1])**2)/(2*variances[2])
        part1 = np.sum((data['y']-f_fit)**2)/(2*variances[0])
        return (part1+part2+part3)
  
    ## the gradient of R_fun
    def grad_R(theta):
        """
        x is the given vector
        theta is the vector of theta1 to theta4
        """
        grad_f = grad_f_vec(data['x'], theta)
        resid = data['y'] - f_curve_vec(data['x'], *theta)
        part1 = [0.0, 0.0, 0.0, 0.0]
        for i in range(N):
            part1 = part1+resid[i]*grad_f[:,i]
        part1 = part1/variances[0]
        part2 = [0.0, 0.0, 0.0, 0.0]
        part3 = [0.0, 0.0, 0.0, 0.0]
        if 'c1' in control.index and control['c1']!="None" and control['c1'] is not None:
            part2[0] = np.sum(control['c1']-theta[0])/variances[1]
        if 'c2' in control.index and control['c2']!="None" and control['c2'] is not None:
            part3[1] = np.sum(control['c2']-theta[1])/variances[2]
        return (-(part1+part2+part3))
  
    ##  the hessian of R_fun
    def hessian_R(theta): 
        grad_f = grad_f_vec(data['x'], theta)
        resid = data['y'] - f_curve_vec(data['x'], *theta)
        part1 = np.reshape(np.zeros(16), (4, 4))
        part2 = np.reshape(np.zeros(16), (4, 4))
        for i in range(N):
            temp = grad_f[:,i]
            temp = temp.reshape(4, 1)
            part1 = part1 + np.matmul(temp, temp.T)
            part2 = part2 + resid[i]*hessian_f(data['x'][i], theta)
        HQ = part1-part2
        HR = HQ/variances[0]
        if 'c1' in control.index and control['c1']!="None" and control['c1'] is not None:
            HR[0,0] = HR[0,0]+L1/variances[1]
        if 'c2' in control.index and control['c2']!="None" and control['c2'] is not None:
            HR[1,1] = HR[1,1]+L2/variances[2]
        return HR
  
    ## optimize R_fun with current variances value
    itern = 1

    obj1 = R_fun(theta)+N/2*np.log(variances[0])
    if 'c1' in control.index and control['c1']!="None" and control['c1'] is not None:
        obj1 = obj1 + L1/2*np.log(variances[1])
    if 'c2' in control.index and control['c2']!="None" and control['c2'] is not None:
        obj1 = obj1 + L2/2*np.log(variances[2])

    obj0 = obj1+1
    while (itern <= itnmax and abs(obj0-obj1)>=tol):
        obj0 = obj1
        optsol = minimize(R_fun, x0=theta, bounds=bounds, #method ='trust-constr', 
                          options={'disp': False, 'maxiter': 10000})
        theta = optsol.x
        ## update the variances 
        resid = data['y'] - f_curve_vec(data['x'], *theta)
        variances[0] = np.mean(resid**2)
        if 'c1' in control.index and control['c1']!="None" and control['c1'] is not None:
            variances[1] = np.mean((control['c1']-theta[0])**2)
        if 'c2' in control.index and control['c2']!="None" and control['c2'] is not None:
            variances[2] = np.mean((control['c2']-theta[1])**2)
      
        ## update the loglikelihood function
        obj1 = R_fun(theta)+N/2*np.log(variances[0])
        if 'c1' in control.index and control['c1']!="None" and control['c1'] is not None:
            obj1 = obj1 + L1/2*np.log(variances[1])
        if 'c2' in control.index and control['c2']!="None" and control['c2'] is not None:
            obj1 = obj1 + L2/2*np.log(variances[2])

        itern = itern + 1
    
    try:
        cov_mle = np.linalg.inv(hessian_R(theta))
    except:
        print("Singular matrix while extracting cov_mle!")
        cov_mle = Moore_Penrose_discard_zero(hessian_R(theta))
    
    ASE = np.sqrt(np.diag(cov_mle))

    #mle = [theta, -(obj1)/(N+L1+L2), itern-1, cov_mle, ASE, variances] #theta, LogL, iterations, col_mle, ASE, variances
    mle = [theta, ASE, variances]
    return mle


def parameter_estimation(data, theta=None, variances=None, itnmax=100, tol=1e-4): 
    """
    Fitting non-linear regression without control
    Parameters:
    ----------
    data      : list of two element: x, y, c1, c2.
                x is vector of concentration
                y is vector of responsex and y must be same length
    theta     : vector of 4 parameters (bottom response, top response, logIC50, hill slope)
    ----------
    return [theta, ASE, variance]
    """

    # Initial value and boundary for theta
    min_y = min(data['y'])
    max_y = max(data['y'])
    range_y = max_y - min_y

    if theta is None:
        theta_0 = [min_y, max_y, data['x'][np.argmin(np.square(data['y']-np.mean(data['y'])))], 1.0]
        upper = [min_y + 0.25*range_y, max_y + 0.25*range_y, 0, 50]
        lower = [min_y - 0.25*range_y, max_y - 0.25*range_y, -50, 0]
    else:
        theta_0 = theta
        upper = [theta[0] + 0.25*range_y, theta[1] + 0.25*range_y, 0, 50]
        lower = [theta[0] - 0.25*range_y, theta[1] - 0.25*range_y, -50, 0]
        # upper = [max(theta[0]*4, theta[0]/4), max(theta[1]*4, theta[1]/4), max(theta[2]*4, theta[2]/4), max(theta[3]*4, theta[3]/4)]
        # lower = [min(theta[0]*4, theta[0]/4), min(theta[1]*4, theta[1]/4), min(theta[2]*4, theta[2]/4), min(theta[3]*4, theta[3]/4)]
    
    # bounds = [(l,u) for (l,u) in zip(lower, upper)]
    
    # # Inital value for variances
    # if variances is None:
    #     y_hat = f_curve_vec(data['x'], *theta)
    #     curve_variance = np.sqrt(np.sum((y_hat - data['y'])**2)/(len(y_hat)-4))
    #     variances = np.array([curve_variance])
    
    # mle = nls(theta=theta, variances=variances, bounds=bounds, data=data, 
    #           itnmax=100, tol=1e-4)

    fit_f, var_matrix = curve_fit(f_curve_vec, xdata=np.array(data['x']), ydata=np.array(data['y']),
                                  absolute_sigma=True, p0=theta_0,
                                  bounds=(lower, upper))
    
    # Estimate ASE for theta
    y_hat = f_curve_vec(data['x'], *fit_f)
    sigma = np.sqrt(np.sum((y_hat - data['y'])**2)/(len(y_hat)-4))
    ASE = np.sqrt(np.diag(var_matrix))*sigma #unscale_SE*sigma

    mle = [fit_f, ASE, np.array([sigma**2])]

    return mle


def parameter_estimation_control(data, theta=None, variances=None, itnmax=100, tol=1e-4):
    """
    Fitting non-linear regression with control
    Parameters:
    ----------
    data      : list of four element: x, y, c1, c2.
                x is vector of concentration
                y is vector of response
                c1 is control at bottom. c1 can be vector or None or 'None'
                c2 is control at top. c2 can be vector or None or 'None'
                x and y must be same length 
    theta     : option, vector of 4 parameters (bottom response, top response, logIC50, hill slope)
    variances : option, vector (sigma_c**2, sigma_b**2, sigma_t**2) 
                variance of the data on the curve, controls at bottom and controls at top
    itnmax    : int, option
                maximum number of iterations for minimization
    tol       : float, option
                tolerance value for minimization
    ----------
    return [theta, ASE, variance]
    """
    #Extract controls data from data
    control = data[['c1','c2']]
    min_y = min(data['y'])
    max_y = max(data['y'])

    # Initial value and boundary for theta
    if data['c1'] != 'None' and data['c1'] is not None:
        min_y = min(min_y, min(data['c1']))
    if data['c2'] != 'None' and data['c2'] is not None:
        max_y = max(max_y, max(data['c2']))
    range_y = max_y - min_y

    if theta is None:
        theta = [min_y, max_y, data['x'][np.argmin(np.square(data['y']-np.mean(data['y'])))], 1.0]
        upper = [min_y + 0.25*range_y, max_y + 0.25*range_y, 0, 50]
        lower = [min_y - 0.25*range_y, max_y - 0.25*range_y, -50, 0]
    else:
        upper = [theta[0] + 0.25*range_y, theta[1] + 0.25*range_y, 0, 50]
        lower = [theta[0] - 0.25*range_y, theta[1] - 0.25*range_y, -50, 0]
        # upper = [max(theta[0]*4, theta[0]/4), max(theta[1]*4, theta[1]/4), max(theta[2]*4, theta[2]/4), max(theta[3]*4, theta[3]/4)]
        # lower = [min(theta[0]*4, theta[0]/4), min(theta[1]*4, theta[1]/4), min(theta[2]*4, theta[2]/4), min(theta[3]*4, theta[3]/4)]
    
    bounds = [(l,u) for (l,u) in zip(lower, upper)]
    
    # Inital value for variances
    if variances is None:
        y_hat = f_curve_vec(data['x'], *theta)
        curve_variance = np.sqrt(np.sum((y_hat - data['y'])**2)/(len(y_hat)-4))

        if data['c1'] != 'None' and data['c1'] is not None:
            var1 = np.var(control['c1'], ddof=1)
        else:
            var1 = None
        if data['c2'] != 'None' and data['c2'] is not None:
            var2 = np.var(control['c2'], ddof=1)
        else:
            var2 = None
        variances = np.array([curve_variance, var1, var2])
    
    try:
        mle = nls_control(theta=theta, variances=variances, bounds=bounds, data=data, 
                          control=control, itnmax=100, tol=1e-4) #theta, ASE, variances
        fitting_with_control = True
    except np.linalg.LinAlgError:
        print('Singular matrix, performing fit without controls')
        mle = parameter_estimation(data)
        fitting_with_control = False

    return [*mle, fitting_with_control]


def f_curve_vec_fixed_theta1(x, R_t, x_50, H):
    """
    Dose-response curve function with fixed R_b = 0
    
    Parameters:
    ----------
    x   : array
          concentration of inhibitor
    R_t : float
          top response
    x_50: float
          logIC50
    H   : float
          hill slope
    ----------
    return an array of response
    """
    return R_t/(1+10**(x*H-x_50*H))


def f_curve_vec_fixed_thetas(x, x_50, H, R_b=0, R_t=100):
    """
    Dose-response curve function with fixed R_t=100 or fixed both R_b=0 and R_t=100
    
    Parameters:
    ----------
    x   : array
          concentration of inhibitor
    x_50: float
          logIC50
    H   : float
          hill slope
    R_b : option, float
          bottom response
    R_t : option, float
          top response
    ----------
    return an array of response
    """
    return R_b+(R_t-R_b)/(1+10**(x*H-x_50*H))


def parameter_estimation_fixed_thetas(data, theta=None, fixed_R_b=False, fixed_R_t=False): 
    """
    Dose-response curve function with fixed R_b=0, fixed R_t=100 or fixed both R_b=0 and R_t=100
    
    Parameters:
    ----------
    data      : list of two element: x, y, c1, c2.
                x is vector of concentration
                y is vector of responsex and y must be same length
    theta     : vector of 4 parameters (bottom response, top response, logIC50, hill slope)
    fixed_R_b : bool, optional
                True = fixed values of R_b = 0
    fixed_R_t : bool, optional
                True = fixed values of R_t = 100
    ----------
    return [theta, ASE, variance]
    """

    #Initial values and boundary for theta
    min_y = min(data['y'])
    max_y = max(data['y'])
    range_y = max_y - min_y

    if theta is not None:
        theta_0 = theta
        upper = [theta[0] + 0.25*range_y, theta[1] + 0.25*range_y, 0, 50]
        lower = [theta[0] - 0.25*range_y, theta[1] - 0.25*range_y, -50, 0]
    else: 
        theta_0 = [min_y, max_y, data['x'][np.argmin(np.square(data['y']-np.mean(data['y'])))], 1.0]
        upper = [min_y + 0.25*range_y, max_y + 0.25*range_y, 0, 50]
        lower = [min_y - 0.25*range_y, max_y - 0.25*range_y, -50, 0]

    # Fit the curve
    if fixed_R_b and fixed_R_t: #Rbase = 0 and Rmax = 100
        fit_f, var_matrix = curve_fit(f_curve_vec_fixed_thetas, xdata=np.array(data['x']), ydata=np.array(data['y']),
                                      absolute_sigma=True, p0=theta_0[2:4],
                                      bounds=(lower[2:4], upper[2:4]))
        y_hat = f_curve_vec(data['x'], 0, 100, *fit_f)
        theta = fit_f
    
    elif fixed_R_b and (not fixed_R_t): #Rbase = 0 
        fit_f, var_matrix = curve_fit(f_curve_vec_fixed_theta1, xdata=np.array(data['x']), ydata=np.array(data['y']),
                                      absolute_sigma=True, p0=theta_0[1:4],
                                      bounds=(lower[1:4], upper[1:4]))
        y_hat = f_curve_vec(data['x'], 0, *fit_f)
        theta = fit_f
    
    elif (not fixed_R_b) and fixed_R_t: #Rmax = 100
        fit_f, var_matrix = curve_fit(f_curve_vec_fixed_thetas, xdata=np.array(data['x']), ydata=np.array(data['y']),
                                      absolute_sigma=True, p0=[theta_0[2], theta_0[3], theta_0[0]],
                                      bounds=([lower[2], lower[3], lower[0]], [upper[2], upper[3], upper[0]]))
        y_hat = f_curve_vec(data['x'], fit_f[2], 100, fit_f[0], fit_f[1])
        theta = [fit_f[2], fit_f[0], fit_f[1]]
    
    else:
        fit_f, var_matrix = curve_fit(f_curve_vec, xdata=np.array(data['x']), ydata=np.array(data['y']),
                                      absolute_sigma=True, p0=theta_0,
                                      bounds=(lower, upper))
        y_hat = f_curve_vec(data['x'], *fit_f)
        theta = fit_f

    # Extract ASE for theta
    sigma = np.sqrt(np.sum((y_hat - data['y'])**2)/(len(y_hat)-4))
    ASE = np.sqrt(np.diag(var_matrix))*sigma #unscale_SE*sigma

    mle = [theta, ASE, np.array([sigma**2])]
    return mle
