# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 14:48:56 2024

@author: pjouannais

Script which calculates the LCAs within a project : for one SSP-IAM, all grid points, all years, all uncertainty. 
Exports the results.
Change the brightway project and the databases on lines 82 and 110.



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

from exploration_functions_bw import *

import functions_orchidee_pipeline as pipe

from Main_functions_favs import *
import Main_functions_favs

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



list(bd.projects)


name_project=  "eco-3.10_conseq_prem_image_SSP1_base_image_2601"

bd.projects.set_current(name_project) # your project



additional_biosphere = bd.Database('additional_biosphere')


ecoinvent_conseq = bd.Database('ecoinvent-3.10-consequential')

biosphere_db_name =  'ecoinvent-3.10-biosphere'


Biosphere =  bd.Database(biosphere_db_name)



list(bd.databases)




# We run separately for each ssp/IAM scenario
# so one dict_db_names per ssp/IAM



dict_db_names = {
    'ei_consequential_3.10_modelimage_pathSSP1-Base_year2050_argshort_slope':[],
    'ei_consequential_3.10_modelimage_pathSSP1-Base_year2070_argshort_slope':[],
    'ei_consequential_3.10_modelimage_pathSSP1-Base_year2090_argshort_slope':[],
    'ei_consequential_3.10_modelimage_pathSSP1-Base_year2050_arglong_area':[],
    'ei_consequential_3.10_modelimage_pathSSP1-Base_year2070_arglong_area':[],
    'ei_consequential_3.10_modelimage_pathSSP1-Base_year2090_arglong_area':[]
                 
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


# Set up the additional biosphere with iLUC emissions
additional_biosphere_multi_categories,list_modif_meth,list_virtual_emissions_same_order_as_meth =  add_virtual_flow_to_additional_biosphere(list_meth)




additional_biosphere = bd.Database('additional_biosphere')

ecoinvent_conseq = bd.Database('ecoinvent-3.10-consequential')





uncert = False # Background uncertainty. Still not coded so remains false.

croptype="wheat"


size_uncert = 20 # 20 randomly samples parameters values for uncertain parameters

name_background_2 = 'ecoinvent-3.10-consequential'  # Original ecoinvent (not premise)





""" Load data """



def extend_dataset_one_step(ds: xr.Dataset) -> xr.Dataset:
    """Extend a dataset by one timestep using the last timestep's data.
    Necessary for 3 climate models"""
    # Detect length and last index of tstep
    tstep_len = ds.dims['tstep']
    last_index = tstep_len - 1
    
    # Get last time value
    last_time = ds['time'].values[last_index]
    
    # Generate new time (assumes yearly step and noleap calendar)
    new_time = cftime.DatetimeNoLeap(
        year=last_time.year + 1,
        month=last_time.month,
        day=last_time.day,
        hour=last_time.hour,
        minute=last_time.minute,
        second=last_time.second
    )

    # Prepare dictionary for new data
    new_data_vars = {}
    
    for var_name, da in ds.data_vars.items():
        # Take the last timestep for each variable and expand tstep
        da_last = da.isel(tstep=last_index).expand_dims(tstep=1)
        new_data_vars[var_name] = da_last

    # Add the new time as a DataArray, matching how it is in the original ds
    new_data_vars['time'] = xr.DataArray([new_time], dims='tstep')

    # Reuse other coordinates (except tstep/time)
    coords = {k: v for k, v in ds.coords.items() if k not in ['tstep', 'time']}

    # Create the new Dataset for the added timestep
    new_ds = xr.Dataset(data_vars=new_data_vars, coords=coords)

    # Concatenate along 'tstep'
    ds_extended = xr.concat([ds, new_ds], dim='tstep')

    return ds_extended




# The value will be overwritten for the sensitivity analysis 

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
keep_npp = True  # The npp will be converted to yield afterwards
yearly_dataframes_orchidee_lower_res,new_lat, new_lon = pipe.pipeline_orchidee_acv_multiple_models_npp_water_lower_res(ds1_list,ds2_list,ds3_list,corres_name_orchidee_acv_npp_water,dry,keep_npp)



# Apply cleaning to all DataFrames in the dictionary
cleaned_yearly_dataframes_orchidee = {
    year: pipe.clean_dataframe(df) for year, df in yearly_dataframes_orchidee_lower_res.items()
}








# Computes the soil CO2 accumulation for AV systems. But ot used eventually for the computation.

cleaned_yearly_dataframes_orchidee_carbon = pipe.add_C_accumulation_wholegrid(cleaned_yearly_dataframes_orchidee,
                                           2007, # Two years between which the slope is calcuated
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
    
    print(name_background)
    
    # create the corresponding name of the foreground
    name_foreground = pipe.name_foreground_from_premise(name_background,"ei_consequential_3.10_")
    
    
    background_db =  bd.Database(name_background)

    
    # collect the corresponding year for the db
    year = pipe.find_corresponding_orchidee_year(name_foreground)
    
    
    exists_already = name_foreground in list(bd.databases)
    

    
        
    # Create foreground. Only collects if already created
    
    foregroundAVS = func_foreg.create_foreground_db(name_foreground,
                                        name_background,
                                        name_background_2,
                                        Biosphere.name,
                                        additional_biosphere.name,
                                        exists_already)
    
    # Finalize foreground
    
    foregroundAVS,parameters_database = func_foreg.finalize_foreground_db(name_foreground,
                                                                                    name_background,
                                                                                    Biosphere.name,
                                                                                    additional_biosphere.name,
                                                                                    exists_already)
    
    

    # Collect values of parameters in the background db (efficiency etc)     
    dict_db_names[name_background].append(parameters_database)
    
    # Load the activities and the functions to the global environment
    
    load_activities_dict_andfunctions(foregroundAVS,
                                      background_db,
                                      Biosphere,
                                      additional_biosphere,
                                      globals())




    # Parameterize the marginal electricity activities to be able to modify its impacts directly
    elec_marginal_fr_copy = init_act_variation_impact(elec_marginal_fr_copy,list_meth,additional_biosphere_multi_categories)
    
    elec_marginal_fr_current_copy = init_act_variation_impact(elec_marginal_fr_current_copy,list_meth,additional_biosphere_multi_categories)
    elec_marginal_es_current_copy = init_act_variation_impact(elec_marginal_es_current_copy,list_meth,additional_biosphere_multi_categories)
    elec_marginal_it_current_copy = init_act_variation_impact(elec_marginal_it_current_copy,list_meth,additional_biosphere_multi_categories)

    # Add the indices of the virtual impact flows that are parameterized.
    list_indices_modif_elec = [[(act_meth.id,elec_marginal_fr_copy.id) for act_meth in list_virtual_emissions_same_order_as_meth]
                               +[(act_meth.id,elec_marginal_fr_current_copy.id) for act_meth in list_virtual_emissions_same_order_as_meth] 
                               +[(act_meth.id,elec_marginal_es_current_copy.id) for act_meth in list_virtual_emissions_same_order_as_meth]
                               +[(act_meth.id,elec_marginal_it_current_copy.id) for act_meth in list_virtual_emissions_same_order_as_meth]][0]
    
    dict_funct["modif_impact_marginal_elec"]={"func":vectorize_function(modif_impact_marginal_elec),
                         "indices":list_indices_modif_elec}


    # Collect the full list of indices, i.e. flows in the matrix that are parameterized
    list_totalindices = []  
                               
    for a in dict_funct:
        
        list_totalindices= list_totalindices+dict_funct[a]["indices"]
        


    # collect the grid returned by orchidee for this year
    grid_year= cleaned_yearly_dataframes_orchidee_sliding_averages_opti_std[year]
    
    

        
    

    #  Create the paramters sample considering the grid from orchidee and the foreground uncertainty/variability
    names_param_total, sample_dataframe,values_for_datapackages,size_model_gridpoints,size_list_config,total_size= pipe.update_param_with_premise_and_orchidee_uncert_withorchideemodel_asparam(dictionnaries,
                                               parameters_database,
                                               grid_year,
                                               croptype,
                                               size_uncert,
                                              "st",
                                              n_points,
                                              n_models)
    
    number_chunks = min(num_cpus,total_size)
       
    # Create the datapackages to compute the lcas. As many datapackages as there as working cpus for parallelizing
   
    list_arrays_for_datapackages,values_for_datapackages,list_chunk_sizes = create_modif_arrays_para(AVS_elec_main,
                              meth1,
                              name_foreground,
                              name_background,
                              "additional_biosphere",
                              "additional_biosphere_multi_categories",
                              list_totalindices,
                              dict_funct,
                              values_for_datapackages,
                              number_chunks)
    
    
    
    
    """Compute the LCAs"""
    
    # List of FUs
    
    list_fu=[
             [AVS_elec_main_single.id,"AVS_elec_main_single"],
             

          
             [PV_ref_single.id,"PV_ref_single"],
  


             [AVS_crop_main_single.id,"AVS_crop_main_single"],
  


             [wheat_fr_ref.id,"wheat_fr_ref"]] 
    
    

    
    
    # COmpute LCAS
    list_tables = compute_stochastic_lcas_para(
                                list_arrays_for_datapackages,
                                list_fu,
                                AVS_elec_main,
                                list_modif_meth,
                                uncert,
                                list_chunk_sizes,
                                name_background,
                                biosphere_db_name,
                                name_foreground,
                                name_project,
                                indices_array_fix,
                                data_array_fix)
    
    
    dict_res[name_background]= [sample_dataframe,list_tables,grid_year]                                        
    





""" Export"""


def extract_year_and_args(input_string):
    # Extract year using regex
    year_match = re.search(r'year(\d{4})', input_string)
    year = int(year_match.group(1)) if year_match else None
    
    # Extract argument using regex
    arg_match = re.search(r'arg(\w+)', input_string)
    arg = arg_match.group(1) if arg_match else None

    return year, arg

def prepare_for_export(dict_res):
    
    for key in dict_res.keys():
        
        [sample_dataframe,list_tables,grid_year] = dict_res[key] 
        
        year, arg = extract_year_and_args(key)
        
        sample_dataframe["Year"] = year
        sample_dataframe["arg_margi"] = arg
        
        dict_res[key]  = [sample_dataframe,list_tables,grid_year]
        
        return dict_res
    
    



x = datetime.datetime.now()

month=str(x.month)
day=str(x.day)
microsec=str(x.strftime("%f"))
             


name_file_res ='Resmedi'+"_"+month+"_"+day+"_"+name_project[16:]

dict_res_finished = prepare_for_export(dict_res)    


export_pickle_2(dict_res_finished, name_file_res, "Results")





    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    