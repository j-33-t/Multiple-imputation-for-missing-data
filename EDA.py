#  Author: Wiktor Szczepanowski ws108772

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
!wget --output-document=data.csv 'https://docs.google.com/spreadsheets/d/e/2PACX-1vQPQQWpaPHiXUB68_lGUlvAdscETYTc-PdxA73MLqKTRPcvhDyAEEn45ChU2-4oim7i7KxCsYZP_s6H/pub?output=csv'
 
# Loading dataset
data_0 = pd.read_csv('data.csv')
data = data_0[data_0['country'] == 'Poland'].set_index('d2')
data=data.drop('country',axis=1).reset_index()
data=data.drop('a14min',axis=1).reset_index()
 
# Cleaning
data['d2'] = data['d2'].replace(["-9","-7"], [None, None])
data['l1'] = data['l1'].replace([-9,-7], [None, None])
data['l2'] = data['l2'].replace([-9,-7], [None, None])
data['bmk3a'] = data['bmk3a'].replace([-9,-7], [None, None])
data['a6c'] = data['a6c'].replace([-9,-7], [None, None])
data['n2p'] = data['n2p'].replace(["-9","-7"], [None, None])
data['j2'] = data['j2'].replace([-9,-7], [None, None])
data['b6'] = data['b6'].replace([-9,-7], [None, None])
data = data[data['d2'].notna()]
 
variable_names = [column for column in data.columns]
missing_value_count = [data[column].isnull().sum() for column in data.columns]
df = pd.DataFrame(data=list(zip(variable_names, missing_value_count)),columns=['variable_names', 'missing_value_count'])
 
# Produces tables with descriptive statistics and calculates 3 sigma rule
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('max_columns', None)
statistics = pd.concat(
    [np.min(data, axis=0),
     np.max(data, axis=0),
     np.mean(data, axis=0),
     np.std(data, axis=0)], axis=1).rename(
         columns={0: 'Minimum', 1: 'Maximum', 2: 'Mean', 3: 'Standard_Deviation'})
 
def upper_sigma(mean,st_dev):
  return mean+3*st_dev
 
def lower_sigma(mean,st_dev):
  return mean-3*st_dev
 
statistics['lower_3_sigma_rule'] = pd.Series([lower_sigma(mean, st_dev) for mean, st_dev in zip(statistics['Mean'], statistics['Standard_Deviation'])],index=statistics.index)
statistics['upper_3_sigma_rule'] = pd.Series([upper_sigma(mean, st_dev) for mean, st_dev in zip(statistics['Mean'], statistics['Standard_Deviation'])],index=statistics.index)
 
print(statistics)
 
# Heatmap of correlations between the variables in the dataset
sns.set(style="white")
corr = data.corr()
#Generate a mask for the upper triangle:
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
#Set up the matplotlib figure and a diverging colormap:
f, ax = plt.subplots(figsize=(8, 7))
cmap = sns.diverging_palette(220, 10, as_cmap=True)
#Draw the heatmap with the mask and correct aspect ratio:
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1, center=0,
square=True, linewidths=.5, cbar_kws={"shrink": .5})
print(corr)
 
# Produces histograms for continuous variables
for i, feature in enumerate(data.columns):
  plt.figure(i)
  sns.histplot(data=data, x=feature)
