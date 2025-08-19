# -*- coding: utf-8 -*-
"""
Created on Mon Apr 14 14:07:20 2025

@author: pjouannais

This script follows the same structure as "Compute"
It is only used to collect the electric production for fixed AV panels from ORCHIDEE-AV in the same format as the impact result files.
This will ease the computation of AV fixed-tilt impacts. 
Exports the results.



"""

import os



import bw2data as bd
import bw2calc as bc
import bw2analyzer as ba
import bw2io as bi
import numpy as np
from scipy import sparse

import bw_processing as bwp

import matrix_utils as mu
import numpy as np
import pandas as pd
import sys
from itertools import *
import itertools

import functions_foreground_favs as func_foreg

from util_functions import *


from activities_parameters_and_functions import *
# from activities_parameters_and_functions_global_favs_npp import *

from exploration_functions_bw import *

import functions_orchidee_pipeline as pipe

from Main_functions import *
import Main_functions

import itertools


import sys

import re


import xarray as xr
import pandas as pd
import numpy as np
import cftime


from sklearn.linear_model import LinearRegression

import multiprocessing




""" Initialize"""



num_cpus = multiprocessing.cpu_count()




# The project does not matter, we will just collect outputs from ORCHIDEE
name_project=  "eco-3.10_conseq_prem_image_SSP2_base_image_short0404"

bd.projects.set_current(name_project) # your project



additional_biosphere = bd.Database('additional_biosphere')


ecoinvent_conseq = bd.Database('ecoinvent-3.10-consequential')

biosphere_db_name =  'ecoinvent-3.10-biosphere'


Biosphere =  bd.Database(biosphere_db_name)





# For sensitivity, only 1 year (2070) but under the two types of arguments to make the marginal mix.
dict_db_names = {
    'ei_consequential_3.10_modelimage_pathSSP2-Base_year2025_argshort_slope':[],
    'ei_consequential_3.10_modelimage_pathSSP2-Base_year2050_argshort_slope':[],
    'ei_consequential_3.10_modelimage_pathSSP2-Base_year2070_argshort_slope':[],
    'ei_consequential_3.10_modelimage_pathSSP2-Base_year2090_argshort_slope':[]
                 
}




meth1 = ('ReCiPe 2016 v1.03, midpoint WITH ILUC (H)',
              'climate change',
              'global warming potential (GWP100) ILUC')



meth3= ('ReCiPe 2016 v1.03, midpoint (H)',
  'material resources: metals/minerals',
  'surplus ore potential (SOP)')

meth5=('ReCiPe 2016 v1.03, midpoint (H)',
  'ecotoxicity: freshwater',
  'freshwater ecotoxicity potential (FETP)')



meth6= ('ReCiPe 2016 v1.03, midpoint (H)',
  'eutrophication: freshwater',
  'freshwater eutrophication potential (FEP)')

meth7=  ('ReCiPe 2016 v1.03, midpoint (H)',
  'particulate matter formation',
  'particulate matter formation potential (PMFP)')




list_meth = [meth1,meth3,meth5,meth6,meth7]




uncert = False # Background uncertainty. Still not coded so remains false.

croptype="wheat"

dict_res = {}

size_uncert = 20

name_background_2 = 'ecoinvent-3.10-consequential'  # Original ecoinvent (not premise)





""" Load data """




# Model 1
ds1 = xr.open_dataset('../orchidee_data/NCC-NorESM1-M_DMI-HIRHAM5_C3.nc') # C3 conv and AVS  
ds2 = xr.open_dataset('../orchidee_data/NCC-NorESM1-M_DMI-HIRHAM5_EY_ST.nc') # energy yield AVS 
ds3 = xr.open_dataset('../orchidee_data/NCC-NorESM1-M_DMI-HIRHAM5_EY_20.nc') # energy yield REF
 

ds1b = xr.open_dataset('../orchidee_data/CNRM-CERFACS-CNRM-CM5_SMHI-RCA4_C3.nc') 
ds2b= xr.open_dataset('../orchidee_data/CNRM-CERFACS-CNRM-CM5_SMHI-RCA4_EY_ST.nc') 
ds3b = xr.open_dataset('../orchidee_data/CNRM-CERFACS-CNRM-CM5_SMHI-RCA4_EY_20.nc') 
 
ds1c = xr.open_dataset('../orchidee_data/ICHEC-EC-EARTH_KNMI-RACMO22_C3.nc') 
ds2c= xr.open_dataset('../orchidee_data/ICHEC-EC-EARTH_KNMI-RACMO22_EY_ST.nc') 
ds3c = xr.open_dataset('../orchidee_data/ICHEC-EC-EARTH_KNMI-RACMO22_EY_20.nc') 
  
ds1d = xr.open_dataset('../orchidee_data/IPSL-IPSL-CM5A-MR_SMHI-RCA4_C3.nc')
ds2d= xr.open_dataset('../orchidee_data/IPSL-IPSL-CM5A-MR_SMHI-RCA4_EY_ST.nc') 
ds3d = xr.open_dataset('../orchidee_data/IPSL-IPSL-CM5A-MR_SMHI-RCA4_EY_20.nc') 
  

ds1e = xr.open_dataset('../orchidee_data/NCC-NorESM1-M_DMI-HIRHAM5_C3.nc') 
ds2e = xr.open_dataset('../orchidee_data/NCC-NorESM1-M_DMI-HIRHAM5_EY_ST.nc') 
ds3e = xr.open_dataset('../orchidee_data/NCC-NorESM1-M_DMI-HIRHAM5_EY_20.nc') 
  
ds1f = xr.open_dataset('../orchidee_data/NCC-NorESM1-M_GERICS-REMO2015_C3.nc') 
ds2f = xr.open_dataset('../orchidee_data/NCC-NorESM1-M_GERICS-REMO2015_EY_ST.nc')
ds3f = xr.open_dataset('../orchidee_data/NCC-NorESM1-M_GERICS-REMO2015_EY_20.nc')
  

ds1g = xr.open_dataset('../orchidee_data/MOHC-HadGEM2-ES_GERICS-REMO2015_C3.nc') 
ds2g = xr.open_dataset('../orchidee_data/MOHC-HadGEM2-ES_GERICS-REMO2015_EY_ST.nc')
ds3g = xr.open_dataset('../orchidee_data/MOHC-HadGEM2-ES_GERICS-REMO2015_EY_20.nc') 

ds1g = extend_dataset_one_step(ds1g)



ds1h = xr.open_dataset('../orchidee_data/MOHC-HadGEM2-ES_DMI-HIRHAM5_C3.nc') 

ds1h = extend_dataset_one_step(extend_dataset_one_step(ds1h))

ds2h = xr.open_dataset('../orchidee_data/MOHC-HadGEM2-ES_DMI-HIRHAM5_EY_ST.nc') 

ds2h = extend_dataset_one_step(ds2h)


ds3h = xr.open_dataset('../orchidee_data/MOHC-HadGEM2-ES_DMI-HIRHAM5_EY_20.nc') 

ds3h = extend_dataset_one_step(ds3h)



ds1i = xr.open_dataset('../orchidee_data/MOHC-HadGEM2-ES_KNMI-RACMO22_C3.nc') 

ds1i = extend_dataset_one_step(extend_dataset_one_step(ds1i))


ds2i = xr.open_dataset('../orchidee_data/MOHC-HadGEM2-ES_KNMI-RACMO22_EY_ST.nc')


ds2i = extend_dataset_one_step(ds2i)


ds3i = xr.open_dataset('../orchidee_data/MOHC-HadGEM2-ES_KNMI-RACMO22_EY_20.nc') 
ds3i = extend_dataset_one_step(ds3i)



ds1_list=[ds1,ds1b,ds1c,ds1d,ds1e,ds1f,ds1g,ds1h,ds1i]
ds2_list=[ds2,ds2b,ds2c,ds2d,ds2e,ds2f,ds2g,ds2h,ds2i]
ds3_list=[ds3,ds3b,ds3c,ds3d,ds3e,ds3f,ds3g,ds3h,d32i]







# Variable Name correspondance between orchidee : acv

corres_name_orchidee_acv_npp_water = {
    "crop_yield_ref":"crop yield ref",
    "crop_yield_avs":"crop yield avs",
    "Water_efficiency_ref":"water efficiency ref",
    "Water_efficiency_avs":"water efficiency avs",
    'Y_ST_avs':"electric yield avs",
    'Y_ST_ref':"electric yield ref st",
    'Y_fixed_ref':"electric yield ref fixed",
    'Total_Soil_Carbon': "total_soil_carbon ref",
    'Total_Soil_Carbon_ST': "total_soil_carbon avs"

    }


    
dry = False
keep_npp = True
yearly_dataframes_orchidee_lower_res,new_lat, new_lon = pipe.pipeline_orchidee_acv_multiple_models_npp_water_lower_res(ds1_list,ds2_list,ds3_list,corres_name_orchidee_acv_npp_water,dry,keep_npp)



# Apply cleaning to all DataFrames in the dictionary
cleaned_yearly_dataframes_orchidee = {
    year: pipe.clean_dataframe(df) for year, df in yearly_dataframes_orchidee_lower_res.items()
}








cleaned_yearly_dataframes_orchidee_carbon = pipe.add_C_accumulation_wholegrid(cleaned_yearly_dataframes_orchidee,
                                           2007, # Two years between wich the slope is calcuated
                                           2037)







cleaned_yearly_dataframes_orchidee_carbon={key:cleaned_yearly_dataframes_orchidee_carbon[key] for key in list(cleaned_yearly_dataframes_orchidee_carbon.keys()) if key >=2025}









""" Sliding averages"""


cleaned_yearly_dataframes_orchidee_sliding_averages_opti_std = pipe.calculate_sliding_avg_and_std_diff_two_vars(cleaned_yearly_dataframes_orchidee_carbon, 30)


























n_points = cleaned_yearly_dataframes_orchidee_sliding_averages_opti_std[2030].shape[1]


    
    

n_models = len(cleaned_yearly_dataframes_orchidee_sliding_averages_opti_std[2030].iloc[0,0]) 
    





# Initalize dictionary of res
dict_res = {}


for name_background in dict_db_names.keys():
    
    # For 1 background database
    
    
    # create the corresponding name of the foreground
    name_foreground = pipe.name_foreground_from_premise(name_background,"ei_consequential_3.10_")
    
    

    
    # collect the corresponding year for the db
    year = pipe.find_corresponding_orchidee_year(name_foreground)
    
    



    

    
        
    
    # Finalize foreground
    
    parameters_database = {'original_surface_panel_single': 10.0,  # Just to put something
     'original_surface_panel_multi': 9.999999999999998,
     'original_surface_panel_cis': 10.0,
     'original_surface_panel_cdte': 10.739856801909307,
     'inverter_specificweight': 5.971978,
     'mass_to_power_electric_installation': 4.049167824561404,
     'new_efficiency_single': 0.3,
     'new_efficiency_multi': 0.3,
     'new_efficiency_cis': 0.3,
     'new_efficiency_cdte': 0.2793333333333333,
     'lifetime': 30.0}




    # collect the grid returned by orchidee for this year
    grid_year= cleaned_yearly_dataframes_orchidee_sliding_averages_opti_std[year]
    
    




    #♣ Any value will do. We won't use anything in the dictionaries.
    # iLUC param
    iluc_par_distributions = {
    
        "NPP_weighted_ha_y_eq_cropref":['unif', [1,0.72, 1.16, 0, 0],"."],
        "NPP_weighted_ha_y_eq_ratio_avs_cropref":['unique', [1,0.5, 2, 0, 0],"."],  # Same quality of land
        "NPP_weighted_ha_y_eq_ratio_pv_cropref":['unique', [1,0.5, 2, 0, 0],"."], # Same quality of land

        "iluc_lm_PV":['list', [1840,1850],"kg CO2 eq.ha eq-1"], # Same market of land

        "iluc_lm_AVS":['unique', [1840,300, 2200, 0],"kg CO2 eq.ha eq-1"],  # Same market of land
        "iluc_lm_cropref":['unique', [1840,300, 2200, 0],"kg CO2 eq.ha eq-1"] # Same market of land

        
        }
    
    
    

    # We just set up the dictionaries with any values. It 's not necessary to collect the ORCHIDEE electric outputs.
    
    # Agronomical param
    Ag_par_distributions = {
    
        "prob_switch_crop":['switch',4,[0,1,0,0],["maize","wheat","soy","alfalfa"],"."],
        "crop_yield_upd_ref":['unif', [1,0.03, 1.5, 1],"."],  # Based on min and max yields in France and SPain
        "crop_yield_ratio_avs_ref":['unif', [1,0.75, 1.25, 1],"."],  
        "crop_fert_upd_ref":['unif', [1,0.03, 1.5, 0, 0],"."], 
        "crop_fert_ratio_avs_ref":['unif', [1,0.75, 1.25, 0, 0],"."], 
        "crop_mach_upd_ref":['unif', [1,0.03, 1.5, 0, 0],"."], # 
        "crop_mach_ratio_avs_ref":['unif', [1,0.75, 1.25, 0, 0],"."],
        "water_upd_ref":['unique', [0,0.8, 1.2, 0, 0],"."], # By putting 0 : removes irrigtion
        "water_ratio_avs_ref":['unique', [1,0.8, 2, 0, 0],"."],
        "carbon_accumulation_soil_ref":['unique', [0,0.8, 1.2, 0, 0],"kg C.ha-1.y-1"],
        "carbon_accumulation_soil_AVS":['unique', [0,0.8, 1.2, 0, 0],"kg C.ha-1.y-1"],
        "carbon_accumulation_soil_PV":['unique', [0,0.8, 1.2, 0, 0],"kg C.ha-1.y-1"],
        
        "init_yield_alfalfa":['unique', [5000,0, 0, 0, 0],"."],
        "init_yield_soy":['unique', [5000,0, 0, 0, 0],"."],
        "init_yield_wheat":['unique', [5000,0, 0, 0, 0],"."],
        "init_yield_maize":['unique', [5000,0, 0, 0, 0],"."],
        

        "correc_orchidee":['unique', [1,0,0,0,0],"."] # ratios for france # For the sensitivity analysis, there is no need to use this parameter as we will directly overwrite the activity yield with the the yield * crop_ref_update
        }



        
        
    
    # PV param
    PV_par_distributions = {
        'annual_producible_PV': ['unif', [1200, 900, 2000, 0, 0],"kwh.kwp-1.y-1"],
        'annual_producible_ratio_avs_pv': ['unique', [1,0.6, 1.3, 0, 0],"."],
        
        'mass_to_power_ratio_electric_instal_PV': ['unif', [2.2,2.2-0.25*2.2, 2.2+0.25*2.2, 0, 0],"kg electric installation . kwp -1"],  # Besseau minimum 2.2
        
        'panel_efficiency_PV': ['unif', [0.228,0.179, 0.270, 0, 0],"."],   # Besseau maximum 22.8%
        
        
        'inverter_specific_weight_PV': ['unif', [0.85,0.85-0.25*0.85, 0.85+0.25*0.85, 0, 0],"kg inverter.kwp-1"],   # Besseau maximum 0.85
       
        'inverter_lifetime_PV': ['unif', [15,15-0.25*15, 15+0.25*15, 0, 0],"y"], # Besseau 15
    
        'plant_lifetime_PV': ['unif', [30,30-0.25*30, 30-0.25*30, 0, 0],"y"], # Besseau and ecoinvent 30
        
        'plant_lifetime_ratio_avs_pv': ['unif', [1,1-0.25*1, 1+0.25*1, 0, 0],"."],  
        
        'concrete_mount_upd_PV': ['unif', [1,1-0.25*1, 1+0.25*1,0.5,0],"."], # TO BE DEFINED WITH ZIMMERMAN
        
        'concrete_mount_ratio_avs_pv': ['unif', [0,0, 2, 1,0],"."],   # here facotr 4 because could be way worse
        
        'aluminium_mount_upd_PV': ['unif', [0.38,0.38-0.25*0.38, 0.38+0.25*0.38, 0, 0],"."], # here besseau suggests 1.5 kilo and the original value is 3.98. So update = 1.5/3.98 =  0.38
            
        'aluminium_mount_ratio_avs_pv': ['unif', [0,0,2,1,0],"."],
        
        'steel_mount_upd_PV': ['unif', [1,1-0.25*1, 1+0.25*1,0.5,0],"."],
            
        'steel_mount_ratio_avs_pv': ['unif', [0,0,2,1,0],"."],
    
        'poly_mount_upd_PV': ['unif', [1,1-0.25*1,1+0.25*1,0.5,0],"."],
            
        'poly_mount_ratio_avs_pv': ['unif',[0,0,2,1,0],"."],
    
    
        'corrbox_mount_upd_PV': ['unif', [1,1-0.25*1,1+0.25*1,0.5],"."],
            
        'corrbox_mount_ratio_avs_pv': ['unif', [0,0,2,1,0],"."],
        
        
        'zinc_mount_upd_PV': ['unif', [1,1-0.25*1,1+0.25*1,0.5],"."],
            
        'zinc_mount_ratio_avs_pv': ['unif', [0,0,2,1,0],"."],
    
    
        'surface_cover_fraction_PV': ['unif', [0.4383,0.4383-0.25*0.4383,0.4383+0.25*0.4383, 0, 0],"m2panel.m-2"],
        
        'surface_cover_fraction_AVS': ['unif', [0.7617,0.7617-0.25*0.7617,0.7617+0.25*0.7617, 0, 0],"."],
        
        
        
        
        'aluminium_panel_weight_PV': ['unif', [1.5,1.5-0.25*1.5,1.5+0.25*1.5, 0, 0],"kg aluminium.m-2 panel"],  #Besseau :The ecoinvent inventory model assumes 2.6 kg aluminum/m2 whereas more recent studies indicate 1.5 kg/m2.42 Some PV panels are even frameless
        
        
    
    
        'manufacturing_eff_wafer_upd_PV': ['unif', [1,1-0.25*1,1+0.25*1, 0, 0],"."],
    
        'solargrade_electric_intensity_PV': ['unif', [0,11-0.25*11,11+0.25*11, 0, 0],"."],  # Besseau range  A CHECKER (110 in ecoivnent)
        
    
        "prob_switch_substielec":['switch',5,[1,0,0,0,0],["substit_margi","substit_PV","substit_margi_fr","substit_margi_it","substit_margi_es"],"."],
        
        "impact_update_margi":['unif',[1,1-0.25*1,1+0.25*1, 0, 0],"."]
    
        }
    
    
    



    # iLUC param
    iluc_par_distributions = {
    
        "NPP_weighted_ha_y_eq_cropref":['unif', [1,0.72, 1.16, 0, 0],"."],
        "NPP_weighted_ha_y_eq_ratio_avs_cropref":['unique', [1,0.5, 2, 0, 0],"."],  # Same quality of land
        "NPP_weighted_ha_y_eq_ratio_pv_cropref":['unique', [1,0.5, 2, 0, 0],"."], # Same quality of land
        # "iluc_lm_PV":['unif', [1800,300, 2200, 0, 0],"kg CO2 eq.ha eq-1"],
        # "iluc_lm_AVS":['unif', [300,1400, 2200, 0, 0],"kg CO2 eq.ha eq-1"],
        # "iluc_lm_cropref":['unif', [1800,1400, 2200, 0, 0],"kg CO2 eq.ha eq-1"]
    
        "iluc_lm_PV":['unique', [1840,300, 2200, 0],"kg CO2 eq.ha eq-1"], # Same market of land

        "iluc_lm_AVS":['unique', [1840,300, 2200, 0],"kg CO2 eq.ha eq-1"],  # Same market of land
        "iluc_lm_cropref":['unique', [1840,300, 2200, 0],"kg CO2 eq.ha eq-1"] # Same market of land

        
        }
    
    # We collect the intitial output values for the crop activities and use them as parameters
    
    
    # output_wheat = [exc["amount"] for exc in list(wheat_fr_ref.exchanges()) if exc["type"]=="production"][0]
    # output_soy = [exc["amount"] for exc in list(soy_ch_ref.exchanges()) if exc["type"]=="production"][0]
    # output_alfalfa = [exc["amount"] for exc in list(alfalfa_ch_ref.exchanges()) if exc["type"]=="production"][0]
    # output_maize= [exc["amount"] for exc in list(maize_ch_ref.exchanges()) if exc["type"]=="production"][0]
    
    # Agronomical param
    Ag_par_distributions = {
    
        "prob_switch_crop":['switch',4,[0,1,0,0],["maize","wheat","soy","alfalfa"],"."],
        "crop_yield_upd_ref":['unique', [1,2, 1, 1],"."],
        "crop_yield_ratio_avs_ref":['unique', [1,1, 0.5, 1],"."],
        "crop_fert_upd_ref":['unique', [1,0.8, 1.2, 0, 0],"."],
        "crop_fert_ratio_avs_ref":['unique', [1,0.8, 1.2, 0, 0],"."],
        "crop_mach_upd_ref":['unique', [1,0.8, 1.2, 0, 0],"."],
        "crop_mach_ratio_avs_ref":['unique', [1,0.8, 1.2, 0, 0],"."],
        "water_upd_ref":['unique', [1,0.8, 1.2, 0, 0],"."],
        "water_ratio_avs_ref":['unique', [1,0.8, 2, 0, 0],"."],
        "carbon_accumulation_soil_ref":['unique', [0,0.8, 1.2, 0, 0],"kg C.ha-1.y-1"],
        "carbon_accumulation_soil_AVS":['unique', [0,0.8, 1.2, 0, 0],"kg C.ha-1.y-1"],
        "carbon_accumulation_soil_PV":['unique', [0,0.8, 1.2, 0, 0],"kg C.ha-1.y-1"],
        
        "init_yield_alfalfa":['unique', [6752.650415288,0, 0, 0, 0],"."],
        "init_yield_soy":['unique', [6752.650415288,0, 0, 0, 0],"."],
        "init_yield_wheat":['unique', [6752.650415288,0, 0, 0, 0],"."],
        "init_yield_maize":['unique', [6752.650415288,0, 0, 0, 0],"."],
        
        # conversion npp to yield
       #  "root_shoot_ratio": ['empirical',[0.21,0.14,0.13,0.17,0.25,0.17],"."],
       #  "straw_grain_ratio":['empirical', [1.564102564, 1.325581395, 1.941176471, 1.631578947, 1.564102564, 1.380952381, 0.639344262, 1.380952381, 1.702702703, 0.666666667,
       # 2.125,1.702702703,2.03030303,1.43902439,1.631578947,3.166666667,2.125,1.5],"."],
       #  "carbon_content_grain":['empirical', [0.413,0.418,0.434,0.426,0.455],"."],
       #  "carbon_content_straw":['unique', [0.446,0, 0, 0, 0],"."],
       #  "water_content_grain":['unique', [0.14,0, 0, 0, 0],"."],
       #  "water_content_straw":['unique', [0.14,0, 0, 0, 0],"."],
        "correc_orchidee":['empirical', [7.77536462737055,
         7.001916044458442,
         6.770308178524785,
         7.526953002133721,
         6.013672339795572,
         8.234609074789143,
         5.508625158970049,
         6.95844023423945,
         7.680775397470734,
         8.060300261610445,
         7.139236612970741],"."] # ratios for france
        }


        
        
    
    # PV param
    PV_par_distributions = {
        'annual_producible_PV': ['unique', [1200, 900, 2000, 0, 0],"kwh.kwp-1.y-1"],
        'annual_producible_ratio_avs_pv': ['unique', [1,0.6, 1.3, 0, 0],"."],
        
        'mass_to_power_ratio_electric_instal_PV': ['unique', [2.2,1.5, 8, 0, 0],"kg electric installation . kwp -1"],  # Besseau minimum 2.2
        
        'panel_efficiency_PV': ['unique', [0.228,0.10, 0.40, 0, 0],"."],   # Besseau maximum 22.8%
        
        
        'inverter_specific_weight_PV': ['unique', [0.85,0.4, 7, 0, 0],"kg inverter.kwp-1"],   # Besseau maximum 0.85
       
        'inverter_lifetime_PV': ['unique', [15,10, 20, 0, 0],"y"], # Besseau 15
    
        'plant_lifetime_PV': ['unique', [30,25, 35, 0, 0],"y"], # Besseau and ecoinvent 30
        
        'plant_lifetime_ratio_avs_pv': ['unique', [1,0.3, 2, 0, 0],"."],  
        
        'concrete_mount_upd_PV': ['list', [1],"."], #first one is SIRTA
        
        'concrete_mount_ratio_avs_pv': ['list', [0],"."],   # first one is SIRTA no concrete
        
        'aluminium_mount_upd_PV': ['list', [0.38],"."], # Besseau used 1.5kg in his parasol file so it corresponds to a ratio of 0.38 with ecoinvent

        'aluminium_mount_ratio_avs_pv': ['list', [0],"."], # no alu in sirta
        
        'steel_mount_upd_PV': ['list', [0.46],"."],  # according to ratios computed with Besseau
            
        'steel_mount_ratio_avs_pv': ['list', [3.06],"."],  # 1.41 is the with the original ratio if we take the amount in ecoinvent so if we take the updated one 									#it's 1.41/0.46 =  3.06  
    
        'poly_mount_upd_PV': ['list', [1],"."],
            
        'poly_mount_ratio_avs_pv': ['list',[1],"."],
    
    
        'corrbox_mount_upd_PV': ['list', [1],"."],
            
        'corrbox_mount_ratio_avs_pv': ['list', [1],"."],
        
        
        'zinc_mount_upd_PV': ['list', [0.46],"."],  # zinc for coating steel
            
        'zinc_mount_ratio_avs_pv': ['list', [3.06],"."],
    
    
        'surface_cover_fraction_PV': ['unique', [0.7617,0.20, 0.6, 0, 0],"m2panel.m-2"],
        
        'surface_cover_fraction_AVS': ['unique', [0.4383,0.10, 0.6, 0, 0],"."],
        
        
        
        
        'aluminium_panel_weight_PV': ['unique', [1.5,0, 3, 0, 0],"kg aluminium.m-2 panel"],  #Besseau :The ecoinvent inventory model assumes 2.6 kg aluminum/m2 whereas more recent studies indicate 1.5 kg/m2.42 Some PV panels are even frameless
        
        
    
    
        'manufacturing_eff_wafer_upd_PV': ['unique', [1,0.1, 2, 0, 0],"."],
    
        'solargrade_electric_intensity_PV': ['unique', [40,11, 40, 0, 0],"."],  # Besseau range 
        
    
        "prob_switch_substielec":['switch',5,[1,0,0,0,0],["substit_margi","substit_PV","substit_margi_fr","substit_margi_it","substit_margi_es"],"."],
        
        "impact_update_margi":['unique',[1,0.01, 3, 0, 0],"."]
    
        }

    
    

    
    
    
    
    # Put all the dictionnaries in one
    
    dictionnaries = {"Ag_par_distributions" : Ag_par_distributions,
                     "PV_par_distributions": PV_par_distributions,
                     "iluc_par_distributions":iluc_par_distributions}
    
    





        
    

    #  Create the paramters sample considering the grid from orchidee and the foreground uncertainty/variability
    names_param_total, sample_dataframe,values_for_datapackages,size_model_gridpoints,size_list_config,total_size= pipe.update_param_with_premise_and_orchidee_uncert_withorchideemodel_asparam(dictionnaries,
                                               parameters_database,
                                               grid_year,
                                               croptype,
                                               size_uncert,
                                              "fixed",
                                              n_points,
                                              n_models)
    
  
    
    keep_columns= ["annual_producible_ratio_avs_pv","annual_producible_PV"]

    dict_res[name_background]= [sample_dataframe[keep_columns],grid_year]                                        
    




""" Export"""




x = datetime.datetime.now()

month=str(x.month)
day=str(x.day)
microsec=str(x.strftime("%f"))
             


name_file_res ='To_compute_PVref'+"_"+month+"_"+day+"_"



export_pickle_2(dict_res, name_file_res, "Results")





    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    