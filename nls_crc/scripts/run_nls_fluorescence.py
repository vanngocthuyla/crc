import pandas as pd
import warnings
import numpy as np
import sys
import os
import itertools
import argparse
import pickle

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", RuntimeWarning)

from _nls_model_outlier_detection import multi_expt, scaling_data

parser = argparse.ArgumentParser()

parser.add_argument( "--data_dir",              type=str,               default="")
parser.add_argument( "--out_dir",               type=str,               default="")

parser.add_argument( "--normalized_data",       action="store_true",    default=False)
parser.add_argument( "--fitting_with_control",  action="store_true",    default=False)
parser.add_argument( "--fixed_R_b",             action="store_true",    default=False)
parser.add_argument( "--fixed_R_t",             action="store_true",    default=False)
parser.add_argument( "--outlier_detection",     action="store_true",    default=False)

args = parser.parse_args()

print("Fitting model with control data:", args.fitting_with_control)

os.chdir(args.data_dir)
df_nc = pd.read_csv('20211118_F_NegativeControl.csv')
plate = df_nc.Plate.unique()
well = []
response = []

well_array = []
date_array = []
response_array = []

count = 0
for i in range(len(df_nc)):
    if i == (len(df_nc)-1):
        well.append(df_nc.Well[i])
        response.append(df_nc['Raw Data (RFU)'][i])
        
        date_array.append(df_nc['Run Date'][i])
        well_array.append(well)
        response_array.append(np.array(response))
    else: 
        if df_nc.Plate[i] == plate[count]: 
            well.append(df_nc.Well[i])
            response.append(df_nc['Raw Data (RFU)'][i])
        else: 
            well_array.append(well)
            date_array.append(df_nc['Run Date'][i-1])
            response_array.append(np.array(response))

            well = []
            response = []
            well.append(df_nc.Well[i])
            response.append(df_nc['Raw Data (RFU)'][i])
            count = count+1

fluorescence_neg = pd.DataFrame([date_array, plate, well_array, response_array],
                                 index=['Date', 'Plate', 'NegativeWell', 'NegativeCon']).T

"""**Positive controls**"""

df_pc = pd.read_csv('20211118_F_PositiveControl.csv')

plate = df_pc.Plate.unique()
well = []
response = []

well_array = []
date_array = []
response_array = []

count = 0
for i in range(len(df_pc)):
    if i == (len(df_pc)-1):
        well.append(df_pc.Well[i])
        response.append(df_pc['Raw Data (RFU)'][i])
        
        date_array.append(df_pc['Run Date'][i])
        well_array.append(well)
        response_array.append(np.array(response))
    else: 
        if df_pc.Plate[i] == plate[count]: 
            well.append(df_pc.Well[i])
            response.append(df_pc['Raw Data (RFU)'][i])
        else: 
            well_array.append(well)
            date_array.append(df_pc['Run Date'][i-1])
            response_array.append(np.array(response))

            well = []
            response = []
            well.append(df_pc.Well[i])
            response.append(df_pc['Raw Data (RFU)'][i])
            count = count+1

fluorescence_pos = pd.DataFrame([date_array, plate, well_array, response_array],
                                 index=['Date', 'Plate', 'PositiveWell', 'PositiveCon']).T

"""**Combined Controls**"""

fluorescence_control_init = pd.merge(fluorescence_neg, fluorescence_pos[['Plate', 'PositiveWell', 'PositiveCon']], 
                                     on=['Plate'], how='inner')

def standard_z_score(dat, cutoff=2.5): 
    z_score = (dat - np.mean(dat))/np.std(dat, ddof=1)
    return dat[abs(z_score)<cutoff]

def robust_z_score(dat, cutoff=2.5): 
    z_score = 0.6745*(dat - np.median(dat))/(np.median(abs(dat-np.median(dat))))
    return dat[abs(z_score)<cutoff]

n = len(fluorescence_control_init)
fluorescence_control = pd.DataFrame([fluorescence_control_init.Date, fluorescence_control_init.Plate, np.zeros(n), fluorescence_control_init.NegativeWell, np.zeros(n), fluorescence_control_init.PositiveWell],
                                    index=fluorescence_control_init.columns).T

count_1 = 0
count_2 = 0
for i in range(n): 
    dat = fluorescence_control_init.iloc[i]
    if np.std(dat['NegativeCon'])==0:
        fluorescence_control['NegativeCon'][i] = dat['NegativeCon']
    else: 
        temp = robust_z_score(dat['NegativeCon'])
        if len(temp)==0:
            temp = standard_z_score(dat['NegativeCon'])
        fluorescence_control['NegativeCon'][i] = temp
        if len(temp) < len(dat['NegativeCon']):
            count_1 = count_1+1
    
    if np.std(dat['PositiveCon'])==0:
        fluorescence_control['PositiveCon'][i] = dat['PositiveCon']
    else: 
        temp_2 = robust_z_score(dat['PositiveCon'])
        if len(temp_2)==0:
            temp_2 = standard_z_score(dat['PositiveCon'])
        fluorescence_control['PositiveCon'][i] = temp_2
        if len(temp_2) < len(dat['PositiveCon']):
            count_2 = count_2+1
print(count_1, 'outliers in negative control and', count_2, 'outliers in positive controls.')

df_nc[df_nc['Raw Data (RFU)']<0].Plate.unique()
df_pc[df_pc['Raw Data (RFU)']<0].Plate.unique()

fluorescence_control = fluorescence_control.drop(index=[58, 59])
fluorescence_control = fluorescence_control.reset_index(drop=True)

"""**Experiments**"""

df = pd.read_csv('20211117_CDD_Export.csv')
records = df.to_dict('records')

def plate_name(list_plate_name):
    list_plate = list_plate_name.split(', ')
    unique_plate = [list_plate[0]]
    for i in range(0,len(list_plate)):
        if not list_plate[i] in unique_plate:
            unique_plate.append(list_plate[i])
    return unique_plate

assay_keys = [('ProteaseAssay_Fluorescence_Dose-Response_Weizmann: Concentration (uM)',
               'ProteaseAssay_Fluorescence_Dose-Response_Weizmann: Raw Data (RFU)'
               )]

experiments = {}
count_err = 0
i = 0

for record in records:
  if not record['Molecule Name'] in experiments.keys():
    experiments[record['Molecule Name']] = []
  for key_conc, key_raw in assay_keys:
    logMtot = {\
      'ProteaseAssay_Fluorescence_Dose-Response_Weizmann: Concentration (uM)': np.log10(5E-9)
      }[key_conc]
    assay = {\
      'ProteaseAssay_Fluorescence_Dose-Response_Weizmann: Concentration (uM)':'Fluorescence'
      }[key_conc]
    
    if record['ProteaseAssay_Fluorescence_Dose-Response_Weizmann: Run Date'] == '09/13/21':
        count_err = count_err + 1
        break
    
    if isinstance(record[key_conc], str):
      concs = [float(c)*1E-6 for c in record[key_conc].split(',')]
      raw = [float(y) for y in record[key_raw].split(',')]
      # Calculate the mean of the response at each concentration
      xs = []
      ys = []
      mean_Rs = []
      n_obs = []
      rundate = record['ProteaseAssay_Fluorescence_Dose-Response_Weizmann: Run Date']
      plate = plate_name(record['Plate Name'])

      #all positive and negative controls of each experiment    
      index_list = np.array(fluorescence_control.Date==rundate)*np.array(fluorescence_control.Plate==plate[0])
      if sum(index_list)>0:
          index = np.where(index_list==1)[0][0]
          all_neg_control = np.array(fluorescence_control.NegativeCon[index])
          all_pos_control = np.array(fluorescence_control.PositiveCon[index])
      else:
          all_neg_control = 'None'
          all_pos_control = 'None'

      for (key, group) in itertools.groupby(sorted(list(zip(concs, raw))), lambda x : x[0]):
          Rgs = np.array(list(group))
          mean_Rgs = np.mean(Rgs[:,1])
          n_obs.append(len(Rgs[:,1]))

          xs.append(np.log10(key))
          ys.append(Rgs[:,1])
          mean_Rs.append(mean_Rgs)
      mean_Rs = np.array(mean_Rs)
      ys = np.concatenate(ys, axis=0)

      experiments[record['Molecule Name']].append({
        'assay': assay, 'ID': record['Molecule Name'], 
        'Molecular_Name': record['Molecule Name 2'],
        'plate_name': plate[0],
        #'logMtot':np.ones(len(xs))*logMtot, 
        'logLtot': np.array(xs),
        'n_obs': n_obs, 
        'raw_R': ys,
        'mean_R': mean_Rs, #mean at each concentration
        'Covalent Warhead': record['Covalent Warhead']*1,
        'Run Date': rundate,
        'Negative Control': all_neg_control,
        'Positive Control': all_pos_control})

# print("There are 17 experiments on 09/13/21 and other 3 experiments, which are ['ID_0739', 'ID_1352', 'ID_1542'], do not have control data.")


assay_keys = [('ProteaseAssay_Fluorescence_Dose-Response_Weizmann: Concentration (uM)',
               'ProteaseAssay_Fluorescence_Dose-Response_Weizmann: Raw Data (RFU)'
               )]

experiments_None = {}
count_err = 0
i = 0
for record in records:
  if not record['Molecule Name'] in experiments_None.keys():
    experiments_None[record['Molecule Name']] = []
  for key_conc, key_raw in assay_keys:
    logMtot = {\
      'ProteaseAssay_Fluorescence_Dose-Response_Weizmann: Concentration (uM)': np.log10(5E-9)
      }[key_conc]
    assay = {\
      'ProteaseAssay_Fluorescence_Dose-Response_Weizmann: Concentration (uM)':'Fluorescence'
      }[key_conc]
    
    if record['ProteaseAssay_Fluorescence_Dose-Response_Weizmann: Run Date'] == '09/13/21':
        count_err = count_err + 1
        break
    
    if isinstance(record[key_conc], str):
      concs = [float(c)*1E-6 for c in record[key_conc].split(',')]
      raw = [float(y) for y in record[key_raw].split(',')]
      # Calculate the mean of the response at each concentration
      xs = []
      ys = []
      mean_Rs = []
      n_obs = []
      rundate = record['ProteaseAssay_Fluorescence_Dose-Response_Weizmann: Run Date']
      plate = plate_name(record['Plate Name'])

      #all positive and negative controls of each experiment    
      index_list = np.array(fluorescence_control.Date==rundate)*np.array(fluorescence_control.Plate==plate[0])
      if sum(index_list)>0:
          index = np.where(index_list==1)[0][0]
          all_neg_control = np.array(fluorescence_control.NegativeCon[index])
          all_pos_control = np.array(fluorescence_control.PositiveCon[index])
      else:
          all_neg_control = 'None'
          all_pos_control = 'None'

      for (key, group) in itertools.groupby(sorted(list(zip(concs, raw))), lambda x : x[0]):
          Rgs = np.array(list(group))
          mean_Rgs = np.mean(Rgs[:,1])
          n_obs.append(len(Rgs[:,1]))

          xs.append(np.log10(key))
          ys.append(Rgs[:,1])
          mean_Rs.append(mean_Rgs)
      mean_Rs = np.array(mean_Rs)
      ys = np.concatenate(ys, axis=0)

      experiments_None[record['Molecule Name']].append({
        'assay': assay, 'ID': record['Molecule Name'], 
        'Molecular_Name': record['Molecule Name 2'],
        'plate_name': plate[0],
        #'logMtot':np.ones(len(xs))*logMtot, 
        'logLtot': np.array(xs),
        'n_obs': n_obs, 
        'raw_R': ys,
        'mean_R': mean_Rs, #mean at each concentration
        'Covalent Warhead': record['Covalent Warhead']*1,
        'Run Date': rundate,
        'Negative Control': None,
        'Positive Control': None})


experiments_norm = {}
count_err = 0
count_neg = []
i = 0

for record in records:
  if not record['Molecule Name'] in experiments_norm.keys():
    experiments_norm[record['Molecule Name']] = []
  for key_conc, key_raw in assay_keys:
    logMtot = {\
      'ProteaseAssay_Fluorescence_Dose-Response_Weizmann: Concentration (uM)': np.log10(5E-9)
      }[key_conc]
    assay = {\
      'ProteaseAssay_Fluorescence_Dose-Response_Weizmann: Concentration (uM)':'Fluorescence'
      }[key_conc]
    
    if record['ProteaseAssay_Fluorescence_Dose-Response_Weizmann: Run Date'] == '09/13/21':
        count_err = count_err + 1
        break
    
    if isinstance(record[key_conc], str):
      concs = [float(c)*1E-6 for c in record[key_conc].split(',')]
      raw = [float(y) for y in record[key_raw].split(',')]
      # Calculate the mean of the response at each concentration
      xs = []
      ys = []
      mean_Rs = []
      n_obs = []
      rundate = record['ProteaseAssay_Fluorescence_Dose-Response_Weizmann: Run Date']
      plate = plate_name(record['Plate Name'])

      #all positive and negative controls of each experiment    
      index_list = np.array(fluorescence_control.Date==rundate)*np.array(fluorescence_control.Plate==plate[0])
      if sum(index_list)>0:
          index = np.where(index_list==1)[0][0]
          all_neg_control = np.array(fluorescence_control.NegativeCon[index])
          all_pos_control = np.array(fluorescence_control.PositiveCon[index])
      else:
          all_neg_control = 'None'
          all_pos_control = 'None'

      for (key, group) in itertools.groupby(sorted(list(zip(concs, raw))), lambda x : x[0]):
          Rgs = np.array(list(group))
          mean_Rgs = np.mean(Rgs[:,1])
          n_obs.append(len(Rgs[:,1]))

          xs.append(np.log10(key))
          ys.append(Rgs[:,1])
          mean_Rs.append(mean_Rgs)
      ys = np.concatenate(ys, axis=0)

      if all_neg_control != 'None':
          # ys = scaling_data(np.asarray(ys), np.mean(all_neg_control), np.mean(all_pos_control))
          # mean_Rs = scaling_data(np.asarray(mean_Rs), np.mean(all_neg_control), np.mean(all_pos_control))
          if np.std(all_pos_control)==0 and np.std(all_neg_control)==0:
              ys_update = scaling_data(np.asarray(ys), max(ys), min(ys))
              mean_Rs = scaling_data(np.asarray(mean_Rs), max(ys), min(ys))
              ys = ys_update
              all_neg_control = 'None'
              all_pos_control = 'None'
          else: 
              ys = scaling_data(np.asarray(ys), np.mean(all_neg_control), np.mean(all_pos_control))
              mean_Rs = scaling_data(np.asarray(mean_Rs), np.mean(all_neg_control), np.mean(all_pos_control))
              negative = scaling_data(all_neg_control, np.mean(all_neg_control), np.mean(all_pos_control))
              positive = scaling_data(all_pos_control, np.mean(all_neg_control), np.mean(all_pos_control))
              all_neg_control = negative
              all_pos_control = positive
      else:
          ys_update = scaling_data(np.asarray(ys), max(ys), min(ys))
          mean_Rs = scaling_data(np.asarray(mean_Rs), max(ys), min(ys))
          ys = ys_update

      if all_neg_control == 'None':
          count_neg.append(record['Molecule Name'])
      
      i = i + 1

      experiments_norm[record['Molecule Name']].append({
        'assay': assay, 'ID': record['Molecule Name'], 
        'Molecular_Name': record['Molecule Name 2'],
        'plate_name': plate[0],
        #'logMtot':np.ones(len(xs))*logMtot, 
        'logLtot': np.array(xs),
        'n_obs': n_obs, 
        'raw_R': ys,
        'mean_R': mean_Rs, #mean at each concentration
        'Covalent Warhead': record['Covalent Warhead']*1,
        'Run Date': rundate,
        'Negative Control': all_neg_control,
        'Positive Control': all_pos_control})


experiments_None_norm = {}
count_err = 0
count_neg = []
i = 0

for record in records:
  if not record['Molecule Name'] in experiments_None_norm.keys():
    experiments_None_norm[record['Molecule Name']] = []
  for key_conc, key_raw in assay_keys:
    logMtot = {\
      'ProteaseAssay_Fluorescence_Dose-Response_Weizmann: Concentration (uM)': np.log10(5E-9)
      }[key_conc]
    assay = {\
      'ProteaseAssay_Fluorescence_Dose-Response_Weizmann: Concentration (uM)':'Fluorescence'
      }[key_conc]
    
    if record['ProteaseAssay_Fluorescence_Dose-Response_Weizmann: Run Date'] == '09/13/21':
        count_err = count_err + 1
        break
    
    if isinstance(record[key_conc], str):
      concs = [float(c)*1E-6 for c in record[key_conc].split(',')]
      raw = [float(y) for y in record[key_raw].split(',')]
      # Calculate the mean of the response at each concentration
      xs = []
      ys = []
      mean_Rs = []
      n_obs = []
      rundate = record['ProteaseAssay_Fluorescence_Dose-Response_Weizmann: Run Date']
      plate = plate_name(record['Plate Name'])

      #all positive and negative controls of each experiment    
      index_list = np.array(fluorescence_control.Date==rundate)*np.array(fluorescence_control.Plate==plate[0])
      if sum(index_list)>0:
          index = np.where(index_list==1)[0][0]
          all_neg_control = np.array(fluorescence_control.NegativeCon[index])
          all_pos_control = np.array(fluorescence_control.PositiveCon[index])
      else:
          all_neg_control = 'None'
          all_pos_control = 'None'

      for (key, group) in itertools.groupby(sorted(list(zip(concs, raw))), lambda x : x[0]):
          Rgs = np.array(list(group))
          mean_Rgs = np.mean(Rgs[:,1])
          n_obs.append(len(Rgs[:,1]))

          xs.append(np.log10(key))
          ys.append(Rgs[:,1])
          mean_Rs.append(mean_Rgs)
      ys = np.concatenate(ys, axis=0)

      if all_neg_control != 'None':
          # ys = scaling_data(np.asarray(ys), np.mean(all_neg_control), np.mean(all_pos_control))
          # mean_Rs = scaling_data(np.asarray(mean_Rs), np.mean(all_neg_control), np.mean(all_pos_control))
          if np.std(all_pos_control)==0 and np.std(all_neg_control)==0:
              ys_update = scaling_data(np.asarray(ys), max(ys), min(ys))
              mean_Rs = scaling_data(np.asarray(mean_Rs), max(ys), min(ys))
              ys = ys_update
              all_neg_control = 'None'
              all_pos_control = 'None'
          else: 
              ys = scaling_data(np.asarray(ys), np.mean(all_neg_control), np.mean(all_pos_control))
              mean_Rs = scaling_data(np.asarray(mean_Rs), np.mean(all_neg_control), np.mean(all_pos_control))
              negative = scaling_data(all_neg_control, np.mean(all_neg_control), np.mean(all_pos_control))
              positive = scaling_data(all_pos_control, np.mean(all_neg_control), np.mean(all_pos_control))
              all_neg_control = negative
              all_pos_control = positive
      else:
          ys_update = scaling_data(np.asarray(ys), max(ys), min(ys))
          mean_Rs = scaling_data(np.asarray(mean_Rs), max(ys), min(ys))
          ys = ys_update

      if all_neg_control == 'None':
          count_neg.append(record['Molecule Name'])
      
      i = i + 1

      experiments_None_norm[record['Molecule Name']].append({
        'assay': assay, 'ID': record['Molecule Name'], 
        'Molecular_Name': record['Molecule Name 2'],
        'plate_name': plate[0],
        #'logMtot':np.ones(len(xs))*logMtot, 
        'logLtot': np.array(xs),
        'n_obs': n_obs, 
        'raw_R': ys,
        'mean_R': mean_Rs, #mean at each concentration
        'Covalent Warhead': record['Covalent Warhead']*1,
        'Run Date': rundate,
        'Negative Control': 'None',
        'Positive Control': 'None'})

"""## Parameter estimation"""

file_name = 'CVD_F'
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
        file_name = file_name + '_no_ctrl'
else: 
    print("Fitting with normalized data.")
    file_name = file_name + '_norm'

    if args.fitting_with_control:
        print("Fitting with control data.")
        data_to_fit = experiments_norm
        file_name = file_name + '_ctrl'
    else: 
        print("Fitting without control data.")
        data_to_fit = experiments_None_norm
        file_name = file_name + '_no_ctrl'

if args.fixed_R_b: 
    print("Fixing R_b = 0.")
    file_name = file_name + '_Rb'

if args.fixed_R_t: 
    print("Fixing R_t = 100.")
    file_name = file_name + '_Rt'

if args.outlier_detection: 
    print("Fitting with outlier detection.")
    file_name = file_name + '_outlier'
else: 
    print("Fitting without outlier detection.")

CVD, problem_set = multi_expt(experiments=data_to_fit, fitting_with_control=args.fitting_with_control, 
                              outlier_detection=args.outlier_detection, fixed_R_b=args.fixed_R_b, fixed_R_t=args.fixed_R_t, 
                              OUT_DIR=args.out_dir, file_name_to_save=file_name)
print("DONE")
