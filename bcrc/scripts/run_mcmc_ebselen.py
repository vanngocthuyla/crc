import pandas as pd
import warnings
import numpy as np
import sys
import os
import itertools
import argparse

import pickle
import arviz as az

import jax
import jax.numpy as jnp
from jax import random
import jax.random as random

import numpyro
import numpyro.distributions as dist
from numpyro.distributions import LogNormal, Normal, Uniform
from numpyro.infer import MCMC, NUTS

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", RuntimeWarning)

from _bdrc_model_outlier_detection import multi_expt, scaling_data

parser = argparse.ArgumentParser()

parser.add_argument( "--data_dir",              type=str,               default="")
parser.add_argument( "--out_dir",               type=str,               default="")

parser.add_argument( "--normalized_data",       action="store_true",    default=False)
parser.add_argument( "--fitting_with_control",  action="store_true",    default=False)
parser.add_argument( "--outlier_detection",     action="store_true",    default=False)

parser.add_argument( "--niters",                type=int,               default=10000)
parser.add_argument( "--nburn",                 type=int,               default=2000)
parser.add_argument( "--nthin",                 type=int,               default=10)
parser.add_argument( "--nchain",                type=int,               default=4)
parser.add_argument( "--random_key",            type=int,               default=0)

args = parser.parse_args()

from jax.config import config
config.update("jax_enable_x64", True)
numpyro.set_host_device_count(args.nchain)

print("Fitting model with control data:", args.fitting_with_control)
print("ninter:", args.niters)
print("nburn:", args.nburn)
print("nchain:", args.nchain)
print("nthin:", args.nthin)

os.chdir(args.data_dir)

## Controls

### **DMSO**

df_nc = pd.read_csv('20210923_MassSpec_DMSO.csv')
records_nc = df_nc.to_dict('records')

wells = ['A12', 'B12', 'C12', 'D12', 'E12', 'F12', 'G12','H12', 'I12', 'J12', 'K12', 'L12', 'M12', 'N12', 'O12', 'P12']
date_table = []
shipment_table = []
control_table = []
count = 0
for record in records_nc:
    date_table.append(record['Date'])
    shipment_table.append(record['Experiments'])
    controls = []
    for j in wells:
        if not np.isnan(record[j]):
            controls.append(record[j])
    control_table.append(np.array(controls))

massspec_neg = pd.DataFrame([date_table, shipment_table, control_table],
                            index=['Date', 'Shipment', 'DMSO']).T

"""**No Enzyme**"""

df_pc = pd.read_csv('20210923_MassSpec_NoEnzyme.csv')
records_pc = df_pc.to_dict('records')

wells = ['A13', 'B13', 'C13', 'D13', 'E13', 'F13', 'G13','H13', 'I13', 'J13', 'K13', 'L13', 'M13', 'N13', 'O13', 'P13']
date_table = []
shipment_table = []
control_table = []
count = 0
for record in records_pc:
    date_table.append(record['Date'])
    shipment_table.append(record['Experiments'])
    controls = []
    for j in wells:
        if not np.isnan(record[j]):
            controls.append(record[j])
    control_table.append(np.array(controls))


massspec_pos = pd.DataFrame([date_table, shipment_table, control_table],
                            index=['Date', 'Shipment', 'NoEnzyme']).T

"""**Combined Controls**"""

massspec_control_init = pd.merge(massspec_neg, massspec_pos, on=['Date', 'Shipment'], how='inner')

massspec_control_init[0:5]

def standard_z_score(dat, cutoff=2.5): 
    z_score = (dat - np.mean(dat))/np.std(dat, ddof=1)
    return dat[abs(z_score)<cutoff]

def robust_z_score(dat, cutoff=2.5): 
    z_score = 0.6745*(dat - np.median(dat))/(np.median(abs(dat-np.median(dat))))
    return dat[abs(z_score)<cutoff]

n = len(massspec_control_init)
massspec_control = pd.DataFrame([massspec_control_init.Date, massspec_control_init.Shipment, np.zeros(n), np.zeros(n)],
                                 index=massspec_control_init.columns).T

count_1 = 0
count_2 = 0
for i in range(n): 
    dat = massspec_control_init.iloc[i]
    temp = robust_z_score(dat['DMSO'])
    if len(temp)==0:
        temp = standard_z_score(dat['DMSO'])
    temp_2 = robust_z_score(dat['NoEnzyme'])
    if len(temp_2)==0:
        temp_2 = standard_z_score(dat['NoEnzyme'])
    massspec_control['DMSO'][i] = temp
    massspec_control['NoEnzyme'][i] = temp_2
    if len(temp) < len(dat['DMSO']):
        count_1 = count_1+1
    if len(temp_2) < len(dat['NoEnzyme']):
        count_2 = count_2+1
print(count_1, 'outliers in DMSO and', count_2, 'outliers in No-Enzyme controls.')

"""### Ebselen"""

df = pd.read_csv('20211110_Ebselen.csv')
conc = np.log10(np.asarray([100.000, 33.333, 11.111, 3.704, 1.235, 0.412, 0.137, 0.046, 0.015, 0.005, 0.002,
                            100.000, 33.333, 11.111, 3.704, 1.235, 0.412, 0.137, 0.046, 0.015, 0.005, 0.002])*1e-6)

ebselen = pd.DataFrame([np.zeros(len(df)), df.Date, df.Experiments, np.zeros(len(df)), np.zeros(len(df))],
                        index=['ID', 'Date', 'Experiments', 'Conc', 'Response']).T

for i in range(len(df)): 
    dat = df.iloc[i]
    conc_array = []
    response_array = []
    for j in range(len(dat)-2):
        if not np.isnan(dat[j+2]): 
            conc_array.append(conc[j])
            response_array.append(dat[j+2])
    if i < 9:
        name = ''.join(['ID_0', str(i+1)])
    else:  
        name = ''.join(['ID_', str(i+1)])
    count = count + 1
    ebselen.ID[i] = name
    ebselen.Conc[i] = conc_array
    ebselen.Response[i] = response_array

ebselen_2 = pd.DataFrame([ebselen.ID, df.Date, df.Experiments, np.zeros(len(df)), np.zeros(len(df)), np.zeros(len(df))],
                         index=['ID', 'Date', 'Experiments', 'Conc', 'n_obs', 'Response']).T

for i in range(len(ebselen)): 
    data = ebselen.iloc[i]
    conc = data.Conc
    raw = data.Response
    conc_array = []
    response_array = []
    n_obs_array = []
    for (key, group) in itertools.groupby(sorted(list(zip(conc, raw))), lambda x : x[0]):
        conc_array.append(key)
        Rgs = np.array(list(group))
        response_array.append(Rgs[:,1])
        n_obs_array.append(len(Rgs[:,1]))
        # print(round(key,2), np.mean(Rgs[:,1]))
    ebselen_2.n_obs[i] = n_obs_array
    ebselen_2.Conc[i]  = conc_array
    ebselen_2.Response[i] = np.concatenate(response_array)


## Unnormalized Data Extraction with Control

records = ebselen_2.to_dict('records')

assay = 'MassSpec'
experiments = {}

for record in records:
    if not record['ID'] in experiments.keys():
        experiments[record['ID']] = []
  
    rundate = record['Date']
    shipment = record['Experiments']

    #all positive and negative controls of each experiment
    index_list = np.array(massspec_control.Shipment==shipment)
    if sum(index_list)>0:
        index = np.where(index_list==1)[0][0]
        all_neg_control = np.array(massspec_control.DMSO[index])
        all_pos_control = np.array(massspec_control.NoEnzyme[index])
    else:
        all_neg_control = 'None'
        all_pos_control = 'None'

    ys = record['Response']

    if all_neg_control != 'None' and np.isnan(np.mean(all_pos_control)) != True:
        # ys = scaling_data(np.asarray(ys), np.mean(all_neg_control), np.mean(all_pos_control))
        # mean_Rs = scaling_data(np.asarray(mean_Rs), np.mean(all_neg_control), np.mean(all_pos_control))
        if np.std(all_pos_control)==0 or np.std(all_neg_control)==0:
            # ys_update = scaling_data(np.asarray(ys), max(ys), min(ys))
            # ys = ys_update            
            all_neg_control = 'None'
            all_pos_control = 'None'
        # else:
            # ys = scaling_data(np.asarray(ys), np.mean(all_neg_control), np.mean(all_pos_control))
            # negative = scaling_data(all_neg_control, np.mean(all_neg_control), np.mean(all_pos_control))
            # positive = scaling_data(all_pos_control, np.mean(all_neg_control), np.mean(all_pos_control))
            # all_neg_control = negative
            # all_pos_control = positive
    else:
        all_neg_control = 'None'
        all_pos_control = 'None'
        # ys_update = scaling_data(np.asarray(ys), max(ys), min(ys))
        # ys = ys_update

    experiments[record['ID']].append({
      'ID': record['ID'], 'assay': assay, 
      'Molecular_Name': 'Ebselen',
      'n_obs': record['n_obs'], 'raw_R': ys,
      'logLtot': np.array(record['Conc']), 
      'Covalent Warhead': 1,
      'Run Date': rundate, 'Shipment': shipment,
      'Negative Control': all_neg_control,
      'Positive Control': all_pos_control})


## Unnormalized Data Extraction without Control

experiments_None = {}

for record in records:
    if not record['ID'] in experiments_None.keys():
        experiments_None[record['ID']] = []
  
    rundate = record['Date']
    shipment = record['Experiments']

    #all positive and negative controls of each experiment
    index_list = np.array(massspec_control.Shipment==shipment)
    if sum(index_list)>0:
        index = np.where(index_list==1)[0][0]
        all_neg_control = np.array(massspec_control.DMSO[index])
        all_pos_control = np.array(massspec_control.NoEnzyme[index])
    else:
        all_neg_control = 'None'
        all_pos_control = 'None'

    ys = record['Response']

    # if all_neg_control != 'None' and np.isnan(np.mean(all_pos_control)) != True:
    #     # ys = scaling_data(np.asarray(ys), np.mean(all_neg_control), np.mean(all_pos_control))
    #     # mean_Rs = scaling_data(np.asarray(mean_Rs), np.mean(all_neg_control), np.mean(all_pos_control))
    #     if np.std(all_pos_control)==0 or np.std(all_neg_control)==0:
    #         ys_update = scaling_data(np.asarray(ys), max(ys), min(ys))
    #         ys = ys_update            
    #         all_neg_control = 'None'
    #         all_pos_control = 'None'
    #     else:
    #         ys = scaling_data(np.asarray(ys), np.mean(all_neg_control), np.mean(all_pos_control))
    #         negative = scaling_data(all_neg_control, np.mean(all_neg_control), np.mean(all_pos_control))
    #         positive = scaling_data(all_pos_control, np.mean(all_neg_control), np.mean(all_pos_control))
    #         all_neg_control = negative
    #         all_pos_control = positive
    # else:
    #     all_neg_control = 'None'
    #     all_pos_control = 'None'
    #     ys_update = scaling_data(np.asarray(ys), max(ys), min(ys))
    #     ys = ys_update
    
    experiments_None[record['ID']].append({
      'ID': record['ID'], 'Molecular_Name': 'Ebselen',
      'n_obs': record['n_obs'], 'raw_R': ys,
      'logLtot': np.array(record['Conc']), 
      'Run Date': rundate, 'Shipment': shipment, 
      'Covalent Warhead': 1,
      'Negative Control': 'None',
      'Positive Control': 'None'})


## Normalized Data Extraction with Control

experiments_norm = {}

for record in records:
    if not record['ID'] in experiments_norm.keys():
        experiments_norm[record['ID']] = []
  
    rundate = record['Date']
    shipment = record['Experiments']

    #all positive and negative controls of each experiment
    index_list = np.array(massspec_control.Shipment==shipment)
    if sum(index_list)>0:
        index = np.where(index_list==1)[0][0]
        all_neg_control = np.array(massspec_control.DMSO[index])
        all_pos_control = np.array(massspec_control.NoEnzyme[index])
    else:
        all_neg_control = 'None'
        all_pos_control = 'None'

    ys = record['Response']

    if all_neg_control != 'None' and np.isnan(np.mean(all_pos_control)) != True:
        # ys = scaling_data(np.asarray(ys), np.mean(all_neg_control), np.mean(all_pos_control))
        # mean_Rs = scaling_data(np.asarray(mean_Rs), np.mean(all_neg_control), np.mean(all_pos_control))
        if np.std(all_pos_control)==0 or np.std(all_neg_control)==0:
            ys_update = scaling_data(np.asarray(ys), max(ys), min(ys))
            ys = ys_update            
            all_neg_control = 'None'
            all_pos_control = 'None'
        else:
            ys = scaling_data(np.asarray(ys), np.mean(all_neg_control), np.mean(all_pos_control))
            negative = scaling_data(all_neg_control, np.mean(all_neg_control), np.mean(all_pos_control))
            positive = scaling_data(all_pos_control, np.mean(all_neg_control), np.mean(all_pos_control))
            all_neg_control = negative
            all_pos_control = positive
    else:
        all_neg_control = 'None'
        all_pos_control = 'None'
        ys_update = scaling_data(np.asarray(ys), max(ys), min(ys))
        ys = ys_update
    experiments_norm[record['ID']].append({
      'ID': record['ID'], 'assay': assay, 
      'Molecular_Name': 'Ebselen',
      'n_obs': record['n_obs'], 'raw_R': ys,
      'logLtot': np.array(record['Conc']), 
      'Covalent Warhead': 1,
      'Run Date': rundate, 'Shipment': shipment,
      'Negative Control': all_neg_control,
      'Positive Control': all_pos_control})


## Normalized Data Extraction without Control

experiments_norm_None = {}

for record in records:
    if not record['ID'] in experiments_norm_None.keys():
        experiments_norm_None[record['ID']] = []
  
    rundate = record['Date']
    shipment = record['Experiments']

    #all positive and negative controls of each experiment
    index_list = np.array(massspec_control.Shipment==shipment)
    if sum(index_list)>0:
        index = np.where(index_list==1)[0][0]
        all_neg_control = np.array(massspec_control.DMSO[index])
        all_pos_control = np.array(massspec_control.NoEnzyme[index])
    else:
        all_neg_control = 'None'
        all_pos_control = 'None'

    ys = record['Response']

    if all_neg_control != 'None' and np.isnan(np.mean(all_pos_control)) != True:
        # ys = scaling_data(np.asarray(ys), np.mean(all_neg_control), np.mean(all_pos_control))
        # mean_Rs = scaling_data(np.asarray(mean_Rs), np.mean(all_neg_control), np.mean(all_pos_control))
        if np.std(all_pos_control)==0 or np.std(all_neg_control)==0:
            ys_update = scaling_data(np.asarray(ys), max(ys), min(ys))
            ys = ys_update            
            all_neg_control = 'None'
            all_pos_control = 'None'
        else:
            ys = scaling_data(np.asarray(ys), np.mean(all_neg_control), np.mean(all_pos_control))
            negative = scaling_data(all_neg_control, np.mean(all_neg_control), np.mean(all_pos_control))
            positive = scaling_data(all_pos_control, np.mean(all_neg_control), np.mean(all_pos_control))
            all_neg_control = negative
            all_pos_control = positive
    else:
        all_neg_control = 'None'
        all_pos_control = 'None'
        ys_update = scaling_data(np.asarray(ys), max(ys), min(ys))
        ys = ys_update

    experiments_norm_None[record['ID']].append({
      'ID': record['ID'], 'assay': assay, 
      'Molecular_Name': 'Ebselen',
      'n_obs': record['n_obs'], 'raw_R': ys,
      'logLtot': np.array(record['Conc']), 
      'Covalent Warhead': 1,
      'Run Date': rundate, 'Shipment': shipment,
      'Negative Control': 'None',
      'Positive Control': 'None'})


"""## Parameter estimation"""

rng_key = random.PRNGKey(args.random_key)
rng_key, rng_key_ = random.split(rng_key)

file_name = 'CVD_Ebs'
if not args.normalized_data: 
    print("Fitting with unnormalized data.")
    file_name = file_name + '_unnorm'

    if args.fitting_with_control:
        print("Fitting with control data.")
        data_to_fit = experiments
        file_name = file_name + '_ctrl'
    else: 
        print("Fitting without control data.")
        data_to_fit = experiments_None
        file_name = file_name + 'no_ctrl'
else: 
    print("Fitting with normalized data.")
    file_name = file_name + '_norm'

    if args.fitting_with_control:
        print("Fitting with control data.")
        data_to_fit = experiments_norm
        file_name = file_name + '_ctrl'
    else: 
        print("Fitting without control data.")
        data_to_fit = experiments_norm_None
        file_name = file_name + 'no_ctrl'

if args.outlier_detection: 
    print("Fitting with outlier detection.")
    file_name = file_name + '_outlier'
else: 
    print("Fitting without outlier detection.")

multi_expt(experiments=data_to_fit, rng_key=rng_key_, outlier_detection=args.outlier_detection, 
           OUT_DIR=args.out_dir, file_name_to_save=file_name)
