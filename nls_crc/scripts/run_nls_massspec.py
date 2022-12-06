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

"""# Mass Spectrometry

## Data Extraction

**DMSO**
"""
os.chdir(args.data_dir)
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
#control_table

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
    if np.mean(dat['DMSO'])!=0 and np.std(dat['DMSO'])!=0:
        temp = robust_z_score(dat['DMSO'])
        if len(temp)==0:
            temp = standard_z_score(dat['DMSO'])
    else: 
        temp = dat['DMSO']
    if np.mean(dat['NoEnzyme'])!=0 and np.std(dat['NoEnzyme'])!=0:
        temp_2 = robust_z_score(dat['NoEnzyme'])
        if len(temp_2)==0:
            temp_2 = standard_z_score(dat['NoEnzyme'])
    else: 
        temp_2 = dat['NoEnzyme']
    massspec_control['DMSO'][i] = temp
    massspec_control['NoEnzyme'][i] = temp_2
    if len(temp) < len(dat['DMSO']):
        count_1 = count_1+1
    if len(temp_2) < len(dat['NoEnzyme']):
        count_2 = count_2+1
print(count_1, 'outliers in DMSO and', count_2, 'outliers in No-Enzyme controls.')

"""**Experiments**"""

df = pd.read_csv('20211117_CDD_Export.csv')
records = df.to_dict('records')

assay_keys = [('ProteaseAssay_RapidFire_Dose-Response_Oxford_Schofield: Conc (uM)',
               'ProteaseAssay_RapidFire_Dose-Response_Oxford_Schofield: raw data'
               )]

assay = 'MassSpec'
experiments = {}
# count = 0

for record in records:
  if not record['Molecule Name'] in experiments.keys():
    experiments[record['Molecule Name']] = []
  for key_conc, key_raw in assay_keys:
    logMtot = {\
      'ProteaseAssay_RapidFire_Dose-Response_Oxford_Schofield: Conc (uM)':np.log10(150E-9)
      }[key_conc]
    assay = {\
      'ProteaseAssay_RapidFire_Dose-Response_Oxford_Schofield: Conc (uM)':'MassSpec',
      }[key_conc]

    if isinstance(record[key_conc], str):
      concs = [float(c)*1E-6 for c in record[key_conc].split(',')]
      raw = [float(y) for y in record[key_raw].split(',')]
      # Calculate the mean of the response at each concentration
      xs = []
      ys = []
      mean_Rs = []
      n_obs = []

      rundate = record['ProteaseAssay_RapidFire_Dose-Response_Oxford_Schofield: Run Date']
      shipment = record['ProteaseAssay_RapidFire_Dose-Response_Oxford_Schofield: Run Conditions']

      #all positive and negative controls of each experiment
      index_list = np.array(massspec_control.Shipment==str(shipment).replace('\r',''))*np.array(massspec_control.Date==rundate)
      if sum(index_list)>0:
          index = np.where(index_list==1)[0][0]
          all_neg_control = np.array(massspec_control.DMSO[index])
          all_pos_control = np.array(massspec_control.NoEnzyme[index])
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

      #Deal with both controls
      if all_neg_control != 'None' and np.isnan(np.mean(all_pos_control)) != True:
          if np.std(all_pos_control)==0 or np.std(all_neg_control)==0:
              # ys_update = scaling_data(np.asarray(ys), max(ys), min(ys))
              # mean_Rs = scaling_data(np.asarray(mean_Rs), max(ys), min(ys))
              # ys = ys_update
              all_neg_control = 'None'
              all_pos_control = 'None'
          # else:
          #     ys = scaling_data(np.asarray(ys), np.mean(all_neg_control), np.mean(all_pos_control))
          #     mean_Rs = scaling_data(np.asarray(mean_Rs), np.mean(all_neg_control), np.mean(all_pos_control))
          #     negative = scaling_data(all_neg_control, np.mean(all_neg_control), np.mean(all_pos_control))
          #     positive = scaling_data(all_pos_control, np.mean(all_neg_control), np.mean(all_pos_control))
          #     all_neg_control = negative
          #     all_pos_control = positive
          #     count = count + 1
      else:
          all_neg_control = 'None'
          all_pos_control = 'None'
          # ys_update = scaling_data(np.asarray(ys), max(ys), min(ys))
          # mean_Rs = scaling_data(np.asarray(mean_Rs), max(ys), min(ys))
          # ys = ys_update
          
      experiments[record['Molecule Name']].append({
        'assay': assay, 'ID': record['Molecule Name'], 
        'Molecular_Name': record['Molecule Name 2'],
        #'logMtot':np.ones(len(xs))*logMtot, 
        'logLtot': np.array(xs), 'n_obs': n_obs, 'raw_R': ys,
        'mean_R': mean_Rs, #mean at each concentration
        'Covalent Warhead': record['Covalent Warhead']*1,
        'Run Date': rundate, 'Shipment': shipment,
        'Negative Control': all_neg_control,
        'Positive Control': all_pos_control})

# print("There are", count, "experiments having both control data.")

# experiments['ID_1314'] #shipment 24b 10/21/20
# experiments['ID_1353'] #shipment 24b 10/13/20
# experiments['ID_1271'] #Shipment 30a\r\nPEG issues\r\nSubstrate 10 uM instead 2 uM
# experiments['ID_1506']
# experiments['ID_1083'] #Shipment 35_34\r\n\r\nSubstrate peaks...
# experiments['ID_1067'] #'Shipment 35b\n\nSubstrate peaks were integrated for + 2 charge states as opposed to +1 charge state as per the assay protocl. + charge state of substrate was no longer present due to high amount of enzyme used.\nConcentration of protein given was 3.4-fold off, hence increasing the enzyme concentration above assay condition of 150 nM.\nFrom my previous experience, less potent compound will seem have apparent lower IC50 when analyzed with +2 charge state of substrate vs +1, however the IC50 of ebselen was similar with both charge state of substrate extraction in RapidFire Integrator.'
# experiments['ID_1692'] #'Shipment 1-3' 22 experiments


experiments_None = {}
# count = 0

for record in records:
  if not record['Molecule Name'] in experiments_None.keys():
    experiments_None[record['Molecule Name']] = []
  for key_conc, key_raw in assay_keys:
    logMtot = {\
      'ProteaseAssay_RapidFire_Dose-Response_Oxford_Schofield: Conc (uM)':np.log10(150E-9)
      }[key_conc]
    assay = {\
      'ProteaseAssay_RapidFire_Dose-Response_Oxford_Schofield: Conc (uM)':'MassSpec',
      }[key_conc]

    if isinstance(record[key_conc], str):
      concs = [float(c)*1E-6 for c in record[key_conc].split(',')]
      raw = [float(y) for y in record[key_raw].split(',')]
      # Calculate the mean of the response at each concentration
      xs = []
      ys = []
      mean_Rs = []
      n_obs = []

      rundate = record['ProteaseAssay_RapidFire_Dose-Response_Oxford_Schofield: Run Date']
      shipment = record['ProteaseAssay_RapidFire_Dose-Response_Oxford_Schofield: Run Conditions']

      #all positive and negative controls of each experiment
      index_list = np.array(massspec_control.Shipment==str(shipment).replace('\r',''))*np.array(massspec_control.Date==rundate)
      if sum(index_list)>0:
          index = np.where(index_list==1)[0][0]
          all_neg_control = np.array(massspec_control.DMSO[index])
          all_pos_control = np.array(massspec_control.NoEnzyme[index])
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

      #Deal with both controls
      if all_neg_control != 'None' and np.isnan(np.mean(all_pos_control)) != True:
          if np.std(all_pos_control)==0 or np.std(all_neg_control)==0:
              # ys_update = scaling_data(np.asarray(ys), max(ys), min(ys))
              # mean_Rs = scaling_data(np.asarray(mean_Rs), max(ys), min(ys))
              # ys = ys_update
              all_neg_control = 'None'
              all_pos_control = 'None'
          # else:
          #     ys = scaling_data(np.asarray(ys), np.mean(all_neg_control), np.mean(all_pos_control))
          #     mean_Rs = scaling_data(np.asarray(mean_Rs), np.mean(all_neg_control), np.mean(all_pos_control))
          #     negative = scaling_data(all_neg_control, np.mean(all_neg_control), np.mean(all_pos_control))
          #     positive = scaling_data(all_pos_control, np.mean(all_neg_control), np.mean(all_pos_control))
          #     all_neg_control = negative
          #     all_pos_control = positive
          #     count = count + 1
      else:
          all_neg_control = 'None'
          all_pos_control = 'None'
          # ys_update = scaling_data(np.asarray(ys), max(ys), min(ys))
          # mean_Rs = scaling_data(np.asarray(mean_Rs), max(ys), min(ys))
          # ys = ys_update
          
      experiments_None[record['Molecule Name']].append({
        'assay': assay, 'ID': record['Molecule Name'], 
        'Molecular_Name': record['Molecule Name 2'],
        #'logMtot':np.ones(len(xs))*logMtot, 
        'logLtot': np.array(xs), 'n_obs': n_obs, 'raw_R': ys,
        'mean_R': mean_Rs, #mean at each concentration
        'Covalent Warhead': record['Covalent Warhead']*1,
        'Run Date': rundate, 'Shipment': shipment,
        'Negative Control': 'None',
        'Positive Control': 'None'})


experiments_norm = {}
count = 0

for record in records:
  if not record['Molecule Name'] in experiments_norm.keys():
    experiments_norm[record['Molecule Name']] = []
  for key_conc, key_raw in assay_keys:
    logMtot = {\
      'ProteaseAssay_RapidFire_Dose-Response_Oxford_Schofield: Conc (uM)':np.log10(150E-9)
      }[key_conc]
    assay = {\
      'ProteaseAssay_RapidFire_Dose-Response_Oxford_Schofield: Conc (uM)':'MassSpec',
      }[key_conc]

    if isinstance(record[key_conc], str):
      concs = [float(c)*1E-6 for c in record[key_conc].split(',')]
      raw = [float(y) for y in record[key_raw].split(',')]
      # Calculate the mean of the response at each concentration
      xs = []
      ys = []
      mean_Rs = []
      n_obs = []

      rundate = record['ProteaseAssay_RapidFire_Dose-Response_Oxford_Schofield: Run Date']
      shipment = record['ProteaseAssay_RapidFire_Dose-Response_Oxford_Schofield: Run Conditions']

      #all positive and negative controls of each experiment
      index_list = np.array(massspec_control.Shipment==str(shipment).replace('\r',''))*np.array(massspec_control.Date==rundate)
      if sum(index_list)>0:
          index = np.where(index_list==1)[0][0]
          all_neg_control = np.array(massspec_control.DMSO[index])
          all_pos_control = np.array(massspec_control.NoEnzyme[index])
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

      # #Using at least one control
      # if all_pos_control != 'None' and np.isnan(np.mean(all_pos_control)) != True:
      #     if np.std(all_pos_control)==0:
      #         all_pos_control = 'None'
      #         scale_pos = min(ys)
      #     else: 
      #         scale_pos = np.mean(all_pos_control)
      # else: 
      #     all_pos_control = 'None'
      #     scale_pos = min(ys)

      # if all_neg_control != 'None':
      #     if np.std(all_neg_control)==0:
      #         all_neg_control = 'None'
      #         scale_neg = max(ys)
      #     else: 
      #         scale_neg = np.mean(all_neg_control)
      #         if max(ys)/(scale_neg-scale_pos)>3.0: 
      #             scale_neg = max(ys)
      #             all_neg_control = 'None'
      # else: 
      #     scale_neg = max(ys)

      # ys = scaling_data(np.asarray(ys), scale_neg, scale_pos)
      # mean_Rs = scaling_data(np.asarray(mean_Rs), scale_neg, scale_pos)
      # if all_pos_control != 'None':
      #     all_pos_control = scaling_data(all_pos_control, scale_neg, scale_pos)
      # if all_neg_control != 'None':
      #     all_neg_control = scaling_data(all_neg_control, scale_neg, scale_pos)

      #Deal with both controls
      if all_neg_control != 'None' and np.isnan(np.mean(all_pos_control)) != True:
          if np.std(all_pos_control)==0 or np.std(all_neg_control)==0:
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
              count = count + 1
      else:
          all_neg_control = 'None'
          all_pos_control = 'None'
          ys_update = scaling_data(np.asarray(ys), max(ys), min(ys))
          mean_Rs = scaling_data(np.asarray(mean_Rs), max(ys), min(ys))
          ys = ys_update
          
      experiments_norm[record['Molecule Name']].append({
        'assay': assay, 'ID': record['Molecule Name'], 
        'Molecular_Name': record['Molecule Name 2'],
        #'logMtot':np.ones(len(xs))*logMtot, 
        'logLtot': np.array(xs), 'n_obs': n_obs, 'raw_R': ys,
        'mean_R': mean_Rs, #mean at each concentration
        'Covalent Warhead': record['Covalent Warhead']*1,
        'Run Date': rundate, 'Shipment': shipment,
        'Negative Control': all_neg_control,
        'Positive Control': all_pos_control})


experiments_norm_None = {}
count = 0

for record in records:
  if not record['Molecule Name'] in experiments_norm_None.keys():
    experiments_norm_None[record['Molecule Name']] = []
  for key_conc, key_raw in assay_keys:
    logMtot = {\
      'ProteaseAssay_RapidFire_Dose-Response_Oxford_Schofield: Conc (uM)':np.log10(150E-9)
      }[key_conc]
    assay = {\
      'ProteaseAssay_RapidFire_Dose-Response_Oxford_Schofield: Conc (uM)':'MassSpec',
      }[key_conc]

    if isinstance(record[key_conc], str):
      concs = [float(c)*1E-6 for c in record[key_conc].split(',')]
      raw = [float(y) for y in record[key_raw].split(',')]
      # Calculate the mean of the response at each concentration
      xs = []
      ys = []
      mean_Rs = []
      n_obs = []

      rundate = record['ProteaseAssay_RapidFire_Dose-Response_Oxford_Schofield: Run Date']
      shipment = record['ProteaseAssay_RapidFire_Dose-Response_Oxford_Schofield: Run Conditions']

      #all positive and negative controls of each experiment
      index_list = np.array(massspec_control.Shipment==str(shipment).replace('\r',''))*np.array(massspec_control.Date==rundate)
      if sum(index_list)>0:
          index = np.where(index_list==1)[0][0]
          all_neg_control = np.array(massspec_control.DMSO[index])
          all_pos_control = np.array(massspec_control.NoEnzyme[index])
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

      # #Using at least one control
      # if all_pos_control != 'None' and np.isnan(np.mean(all_pos_control)) != True:
      #     if np.std(all_pos_control)==0:
      #         all_pos_control = 'None'
      #         scale_pos = min(ys)
      #     else: 
      #         scale_pos = np.mean(all_pos_control)
      # else: 
      #     all_pos_control = 'None'
      #     scale_pos = min(ys)

      # if all_neg_control != 'None':
      #     if np.std(all_neg_control)==0:
      #         all_neg_control = 'None'
      #         scale_neg = max(ys)
      #     else: 
      #         scale_neg = np.mean(all_neg_control)
      #         if max(ys)/(scale_neg-scale_pos)>3.0: 
      #             scale_neg = max(ys)
      #             all_neg_control = 'None'
      # else: 
      #     scale_neg = max(ys)

      # ys = scaling_data(np.asarray(ys), scale_neg, scale_pos)
      # mean_Rs = scaling_data(np.asarray(mean_Rs), scale_neg, scale_pos)
      # if all_pos_control != 'None':
      #     all_pos_control = scaling_data(all_pos_control, scale_neg, scale_pos)
      # if all_neg_control != 'None':
      #     all_neg_control = scaling_data(all_neg_control, scale_neg, scale_pos)

      #Deal with both controls
      if all_neg_control != 'None' and np.isnan(np.mean(all_pos_control)) != True:
          if np.std(all_pos_control)==0 or np.std(all_neg_control)==0:
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
              count = count + 1
      else:
          all_neg_control = 'None'
          all_pos_control = 'None'
          ys_update = scaling_data(np.asarray(ys), max(ys), min(ys))
          mean_Rs = scaling_data(np.asarray(mean_Rs), max(ys), min(ys))
          ys = ys_update
          
      experiments_norm_None[record['Molecule Name']].append({
        'assay': assay, 'ID': record['Molecule Name'], 
        'Molecular_Name': record['Molecule Name 2'],
        #'logMtot':np.ones(len(xs))*logMtot, 
        'logLtot': np.array(xs), 'n_obs': n_obs, 'raw_R': ys,
        'mean_R': mean_Rs, #mean at each concentration
        'Covalent Warhead': record['Covalent Warhead']*1,
        'Run Date': rundate, 'Shipment': shipment,
        'Negative Control': 'None',
        'Positive Control': 'None'})


"""## Parameter estimation"""

file_name = 'CVD_MS'
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
        data_to_fit = experiments_norm_None
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