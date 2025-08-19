# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 14:48:56 2024

@author: pjouannais

Script which calculates the LCAs for the GSA.
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

import functions_foreground as func_foreg

from util_functions import *


from activities_parameters_and_functions_sensi import *

from basic_exploration_functions_bw import *

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



list(bd.projects)



name_project=  "eco-3.10_conseq_prem_remind_SSP1_base_remind_2601"

bd.projects.set_current(name_project) # your project



additional_biosphere = bd.Database('additional_biosphere')


ecoinvent_conseq = bd.Database('ecoinvent-3.10-consequential')

biosphere_db_name =  'ecoinvent-3.10-biosphere'


Biosphere =  bd.Database(biosphere_db_name)




dict_db_names = {
    'ei_consequential_3.10_modelremind_pathSSP1-Base_year2070_argshort_slope':[]

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


additional_biosphere_multi_categories,list_modif_meth,list_virtual_emissions_same_order_as_meth =  add_virtual_flow_to_additional_biosphere(list_meth)




additional_biosphere = bd.Database('additional_biosphere')

ecoinvent_conseq = bd.Database('ecoinvent-3.10-consequential')





uncert = False # Background uncertainty. Still not coded so remains false.

croptype="wheat"

dict_res = {}

size_uncert = 50000

name_background_2 = 'ecoinvent-3.10-consequential'  # Original ecoinvent (not premise)





""" Load data """




# Model 1

# The value will be overwritten for the sensitivity analysis so it does not matter 
ds1 = xr.open_dataset('../orchidee_data/NCC-NorESM1-M_DMI-HIRHAM5_C3.nc') # C3 conv and AVS
ds2 = xr.open_dataset('../orchidee_data/NCC-NorESM1-M_DMI-HIRHAM5_EY_ST.nc') # energy yield AVS
ds3 = xr.open_dataset('../orchidee_data/NCC-NorESM1-M_DMI-HIRHAM5_EY_20.nc') # energy yield REF
 

 



ds1_list=[ds1]
ds2_list=[ds2]
ds3_list=[ds3]

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





"""  FOR SMALL SAMPLES"""
keep_only_some_points = random.sample(list(cleaned_yearly_dataframes_orchidee[2007].columns), 1)   # ONLY 1 point as it does not matter



cleaned_yearly_dataframes_orchidee = {key:cleaned_yearly_dataframes_orchidee[key][keep_only_some_points] for key in cleaned_yearly_dataframes_orchidee.keys()}

""" ###############"""





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
    
    print(name_background)
    
    # create the corresponding name of the foreground
    name_foreground = pipe.name_foreground_from_premise(name_background,"ei_consequential_3.10_")
    
    
    background_db =  bd.Database(name_background)

    
    # collect the corresponding year for the db
    year = pipe.find_corresponding_orchidee_year(name_foreground)
    
    
    exists_already = name_foreground in list(bd.databases)



    Biosphere =  bd.Database('ecoinvent-3.10-biosphere')
    

    
        
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
    names_param_total, sample_dataframe,values_for_datapackages,size_model_gridpoints,size_list_config,total_size= pipe.update_param_with_premise_and_orchidee_uncert_withorchideemodel_asparam_sens(dictionnaries,
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
             


name_file_res ='Resmedi'+"_"+month+"_"+day+"_"+microsec+"size="+"sensi"

dict_res_finished = prepare_for_export(dict_res)    


export_pickle_2(dict_res_finished, name_file_res, "Results")





    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    