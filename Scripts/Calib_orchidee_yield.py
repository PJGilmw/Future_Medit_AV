# -*- coding: utf-8 -*-
"""
Created on Wed Mar 19 21:01:40 2025

@author: pjouannais

Computing the list of calibration factors from ORCHIDEE (ERA LAND 5 reanalysis) to actual grain yields obtained in France for 2010-2020.

"""



import numpy as np
from scipy import sparse


import numpy as np
import pandas as pd
import sys
from itertools import *
import itertools

import os

import xarray as xr
import matplotlib.pyplot as plt
import numpy as np





""" Actual Yields France"""


df_rendements = pd.read_excel('../Calib yields France/Rendement20102023_selec_infosurf.xlsx')
df_locations = pd.read_csv('../Calib yields France/latlong_dep.csv',sep=";")

# Group by department number and calculate the mean latitude and longitude



df_locations["mean_lat"]=(df_locations['north lat'] + df_locations['south lat'])/2
df_locations["mean_long"]=(df_locations["east long"] + df_locations["west long"])/2

df_locations['Departement']  = df_locations['Departement'].astype(int)


# Extract department number from the first column as strings
# df_rendements['Department Number'] = df_rendements.iloc[:, 0].str.extract(r'(\d{3})')
df_rendements['Departement'] = df_rendements.iloc[:, 0].str.extract(r'(\d{3})')

# Drop rows where department number couldn't be extracted (NaN values)
df_rendements = df_rendements.dropna(subset=['Departement'])

# Convert the department numbers to integers
df_rendements['Departement'] = df_rendements['Departement'].astype(int)

# Merge the dataframes on department number
merged_df = pd.merge(df_rendements, df_locations, on='Departement')


merged_df = merged_df.drop("LIB_DEP",axis=1)
merged_df = merged_df.drop("north lat",axis=1)
merged_df = merged_df.drop("south lat",axis=1)
merged_df = merged_df.drop("east long",axis=1)
merged_df = merged_df.drop("west long",axis=1)



merged_df_wheat = merged_df[merged_df['LIB_SAA']=="03 - Total blé tendre (01 + 02)"]
merged_df_oat = merged_df[merged_df['LIB_SAA']=="13 - Total avoine (11 + 12)"]
merged_df_soy = merged_df[merged_df['LIB_SAA']=="33 - Soja"]


def filter_based_on_surface_andclean(merged_df,quantile):
    # We remove the departement for which the cultivated surfaces are lower than the first quartile of surfaces , to make sure there is enough surface for representivness.
    q1_values = merged_df.groupby("LIB_SAA")["SURF_2010"].quantile(quantile)

    # Filter the DataFrame based on Q1 threshold per group
    merged_df_filtered = merged_df[
        merged_df.apply(lambda row: row["SURF_2010"] > q1_values[row["LIB_SAA"]], axis=1)
    ]
    
    merged_df_filtered = merged_df_filtered.drop(merged_df_filtered.filter(regex="^SURF|^PROD").columns, axis=1)

    merged_df_filtered = merged_df_filtered.rename(
        columns=lambda col: int(col.replace("REND_", "")) if col.startswith("REND_") else col
    )

    return merged_df_filtered
    
merged_df_wheat_filtered =  filter_based_on_surface_andclean(merged_df_wheat,0.30)   
merged_df_oat_filtered =  filter_based_on_surface_andclean(merged_df_oat,0.30)   
merged_df_soy_filtered =  filter_based_on_surface_andclean(merged_df_soy,0.30)   

    
dfmin=pd.DataFrame(merged_df_wheat_filtered.min()).transpose()
dfmax=pd.DataFrame(merged_df_wheat_filtered.max()).transpose()

maxfrance= max(list(dfmax.iloc[0][1:-3])) # For sensi


# Average
average_yields_FR_wheat = pd.DataFrame(merged_df_wheat_filtered.drop(columns=["LIB_SAA"]).mean()).transpose()

# Reset index if needed
average_yields_FR_wheat.reset_index(drop=True, inplace=True)



# Average NPP France from ORCHIDEE (ERA5)

datasets_orchidee_france_eraland5= xr.open_dataset("../Calib yields France/France_ERA5.nc")
list(datasets_orchidee_france_eraland5.variables)
averages_npp = []
year=2009
for i in range(11):
    year+=1

    averages_npp.append(float(np.mean(datasets_orchidee_france_eraland5["NPP"][i])))






average_yields_FR_wheat.loc[:-3]

average_yields_FR_wheat.iloc[:, :-6] 

actual_average_yields_FR = list(average_yields_FR_wheat.iloc[:, :-6].iloc[0]*100)




# We actually don't need to use the grain yield ratios
ratios_without_conversion = [b/a for a,b in zip(averages_npp,actual_average_yields_FR)]

#[7.77536462737055,
 # 7.001916044458442,
 # 6.770308178524785,
 # 7.526953002133721,
 # 6.013672339795572,
 # 8.234609074789143,
 # 5.508625158970049,
 # 6.95844023423945,
 # 7.680775397470734,
 # 8.060300261610445,
 # 7.139236612970741]

#We will use this as the empirical distribution for the conversion of NPP orhcidee outptus to grain yields







