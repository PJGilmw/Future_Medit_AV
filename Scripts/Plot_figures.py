# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 16:12:20 2024

@author: pjouannais


Script which processes the results and exports figures for the article and the SI.

"""


import numpy as np


 
import pandas as pd
import itertools


from util_functions import *

import copy


import sys

import re

import xarray as xr
   
 
import cftime

from sklearn.linear_model import LinearRegression

import multiprocessing

import matplotlib.pyplot as plt
import numpy.ma as ma

from mpl_toolkits.mplot3d import Axes3D
 
   
import seaborn as sns


from mpl_toolkits.mplot3d.art3d import Poly3DCollection
 
from mpl_toolkits.mplot3d import Axes3D

import os
 

from matplotlib.cm import ScalarMappable

import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize, TwoSlopeNorm

from functions_plot import *

import numpy as np


from functools import reduce




""" Import and process results"""
 







# List of used impact methods for the simulations (in the same order)

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




# Impact global wheat markets
impact_wheat_market = importpickle("../Results/Impacts_wheat_market_4_18.pkl")

impact_wheat_market['Ecoinvent_310_short'] = impact_wheat_market.pop('Ecoinvent_310')

impact_wheat_market["eco-3.10_conseq_prem_remind_SSP5_ndc_remind_short0404"]["ei_consequential_3.10_modelremind_pathSSP5-NDC_year2090_argshort_slope"] =impact_wheat_market["eco-3.10_conseq_prem_remind_SSP5_ndc_remind_long0404"]["ei_consequential_3.10_modelremind_pathSSP5-NDC_year2090_arglong_area"] 


# Names of simulations results files
simus = {"Resmedi_4_5_prem_image_SSP1_base_image_2601",
         "Resmedi_4_7_prem_remind_SSP5_ndc_remind_short0404",
         "Resmedi_4_8_prem_remind_SSP5_ndc_remind_long0404",
         "Resmedi_4_8_prem_image_SSP2_base_image_long0404",
         "Resmedi_4_9_prem_remind_SSP1_base_remind_2601",
         "Resmedi_4_10_prem_remind_SSP2_base_remind_long0404",
         "Resmedi_4_10_prem_remind_SSP2_NPi_remind_short0404",
         "Resmedi_4_8_prem_remind_SSP2_NPi_remind_long0404",
         "Resmedi_4_11_prem_image_SSP2_base_image_short0404",
         "Resmedi_4_15_prem_remind_SSP2_base_remind_short0404",
         "Resmedi_4_16_310"
         
         }

dict_import={key:importpickle("../Results/Server/sim/"+key+".pkl") for key in simus}

# Change name to ease processing
dict_import['Resmedi_4_15_prem_Ecoinvent_310_short0404'] = dict_import.pop('Resmedi_4_16_310')

# WIth ORCHIDEE-AVS inputs to compute fixed PV impacts



# Modify to add an impact column for when global wheat market is substituted instead of local wheat


dict_import_and_add_wheatmarket={key:add_impact_avs_elec_wheatglobalmarket_2_eco310_3(dict_import[key],impact_wheat_market,key) for key in dict_import}




# Change names to ease processing
dict_import_and_add_wheatmarket["Resmedi_4_15_prem_Ecoinvent_310_short0404"]["modelecoinvent-3.10-consequential_year2025_argshort_slope"]=dict_import_and_add_wheatmarket["Resmedi_4_15_prem_Ecoinvent_310_short0404"].pop("ecoinvent-3.10-consequential_year2025")
dict_import_and_add_wheatmarket["Resmedi_4_15_prem_Ecoinvent_310_short0404"]["modelecoinvent-3.10-consequential_year2050_argshort_slope"]=dict_import_and_add_wheatmarket["Resmedi_4_15_prem_Ecoinvent_310_short0404"].pop("ecoinvent-3.10-consequential_year2050")
dict_import_and_add_wheatmarket["Resmedi_4_15_prem_Ecoinvent_310_short0404"]["modelecoinvent-3.10-consequential_year2070_argshort_slope"]=dict_import_and_add_wheatmarket["Resmedi_4_15_prem_Ecoinvent_310_short0404"].pop("ecoinvent-3.10-consequential_year2070")
dict_import_and_add_wheatmarket["Resmedi_4_15_prem_Ecoinvent_310_short0404"]["modelecoinvent-3.10-consequential_year2090_argshort_slope"]=dict_import_and_add_wheatmarket["Resmedi_4_15_prem_Ecoinvent_310_short0404"].pop("ecoinvent-3.10-consequential_year2090")


# Impact elec grid for adding other models
impact_elecgrid = importpickle("../Results/Impacts_elec_market_6_3.pkl")

impact_elecgrid['Ecoinvent_310_short'] = impact_elecgrid.pop('Ecoinvent_310')

dict_import_and_add_wheatmarket_addimpactgrid={key:add_impact_elecgrid(dict_import_and_add_wheatmarket[key],impact_elecgrid,key) for key in dict_import_and_add_wheatmarket}



# A list of the outputs, divided by models (IAM, SSP, year, marginal identification, and type of structures (config))
# 20 =uncertainty sample size
list_models = [divide_by_config(dict_import_and_add_wheatmarket_addimpactgrid[key], 20,2) for key in dict_import_and_add_wheatmarket_addimpactgrid]  


list_years = [2025,2050,2070,2090] 






# Reorganize by years


dict_res = reorganize_by_years(list_models,
                        list_years)
        






dict_import_pvreffixed_ori = importpickle("../Results/To_compute_PVref_6_5_.pkl")
    




# Compute impact PV kWh assuming fixed PV panels
for key in dict_res:
    
    corresponding_string_year = find_string_with_year(dict_import_pvreffixed_ori, key)
    
    res = dict_res[key]
    res_pv_fixed = dict_import_pvreffixed_ori[corresponding_string_year][0]
    
    for model_key in res:
        
        list_res_model = res[model_key]
        
        list_res_config1 = list_res_model[2]
        list_res_config2 = list_res_model[3]
        
        for meth_index in range(len(list_res_config1)):
            
            
            print(len(list_res_config1[meth_index]["PV_ref_single"]))
            print(len(res_pv_fixed["annual_producible_ratio_avs_pv"]))
            
            
            list_res_config1[meth_index]["PV_ref_single_fixed"] =  ref_pv_fixed(list_res_config1[meth_index]["PV_ref_single"],res_pv_fixed["annual_producible_ratio_avs_pv"]) 
            list_res_config2[meth_index]["PV_ref_single_fixed"] =  ref_pv_fixed(list_res_config2[meth_index]["PV_ref_single"],res_pv_fixed["annual_producible_ratio_avs_pv"]) 
            
            

   
    
# Compute absolute and relative differences of impacts    
dict_res_dif = make_dif_over_dict(dict_res,"AVS_elec_main_single","PV_ref_single",2) #nlist=2  # Let's assume we have another funciton to compute dif with fixed ref
dict_res_dif = make_dif_over_dict(dict_res_dif,"AVS_elec_main_single","PV_ref_single_fixed",2) #nlist=2  # Let's assume we have another funciton to compute dif with fixed ref
dict_res_dif = make_dif_over_dict(dict_res_dif,"AVS_elec_main_wheat_market","PV_ref_single_fixed",2) #nlist=2  # Let's assume we have another funciton to compute dif with fixed ref
dict_res_dif = make_dif_over_dict(dict_res_dif,"AVS_elec_main_wheat_market","PV_ref_single",2) #nlist=2  # Let's assume we have another funciton to compute dif with fixed ref

dict_res_dif = make_dif_over_dict(dict_res_dif,"AVS_crop_main_single","wheat_fr_ref",2) #nlist=2  # Let's assume we have another funciton to compute dif with fixed ref




# Organize results to be able to plot # 9 GCM/RCM, 2 config (mounting structures), 20 stochastically sampled values for uncertain parameters
dict_res_orga = apply_organize_results_to_all_multiconfig(dict_res_dif, 9, 2, 20, list_meth,True)





# Organize input parameters to be able to plot
dict_param_orga = apply_organize_results_to_all_param_multiconfig(dict_res_dif, 11, 2, 20,True)

# We focus on Global warming, transpose to ease plotting
dict_res_orga_gw_T = select_categ_and_transpose(dict_res_orga,"global warming potential (GWP100) ILUC")

dict_res_orga_gw_T.keys()




dict_res_orga_gw_T_nodoublon = dict_res_orga_gw_T



        


# Generate dictionary of results with only 1 of the 2 configs 

dict_res_orga_gw_T_config1 = copy.deepcopy(dict_res_orga_gw_T_nodoublon)
dict_res_orga_gw_T_config2 = copy.deepcopy(dict_res_orga_gw_T_nodoublon)


# Remove models containing "config2" from config1
for key, dict_ in dict_res_orga_gw_T_config1.items():
    for model in list(dict_.keys()):  # Use a copy of the keys
        if "config2" in model:
            dict_.pop(model)

# Remove models containing "config1" from config2
for key, dict_ in dict_res_orga_gw_T_config2.items():
    for model in list(dict_.keys()):
        if "config1" in model:
            dict_.pop(model)








""" PLOTS"""




""" Figure 1"""


""" Merging all models into 1 big plot"""

    

list_var_change_ratios = [
  'AVS_elec_main_single-PV_ref_single/PV_ref_single',
  'AVS_elec_main_single-PV_ref_single_fixed/PV_ref_single_fixed',
  'AVS_elec_main_wheat_market-PV_ref_single_fixed/PV_ref_single_fixed',
  'AVS_elec_main_wheat_market-PV_ref_single/PV_ref_single']


list_var_impacts= ['AVS_elec_main_single',
  'PV_ref_single',
  'AVS_crop_main_single',
  'wheat_fr_ref',
  'AVS_elec_main_wheat_market',
  'PV_ref_single_fixed']


# Spearated configs


# config1

for variable in list_var_change_ratios:
    
    global_min, global_max = compute_global_min_max_q3(dict_res_orga_gw_T_config1, variable)

    for year in dict_res_orga_gw_T_config1:
        plot_combined_surface_all_models_per_year(
            year,
            dict_res_orga_gw_T_config1,
            variable,
            global_min,
            global_max,
            "config1"
        )



# different angle
for variable in list_var_change_ratios:
    
    global_min, global_max = compute_global_min_max_q3(dict_res_orga_gw_T_config1, variable)
    print(variable)
    print(global_min,global_max)
    for year in dict_res_orga_gw_T_config1:
        plot_combined_surface_all_models_per_year(
            year,
            dict_res_orga_gw_T_config1,
            variable,
            global_min,
            global_max,
            "config1",
            angle = (10,230)

        )




# config 2


for variable in list_var_change_ratios:
    
    global_min, global_max = compute_global_min_max_q3(dict_res_orga_gw_T_config2, variable)

    for year in dict_res_orga_gw_T_config2:
        plot_combined_surface_all_models_per_year(
            year,
            dict_res_orga_gw_T_config2,
            variable,
            global_min,
            global_max,
            "config2",
            angle = (60,225)
        )


# different angle

for variable in list_var_change_ratios:
    
    global_min, global_max = compute_global_min_max_q3(dict_res_orga_gw_T_config2, variable)

    for year in dict_res_orga_gw_T_config2:
        plot_combined_surface_all_models_per_year(
            year,
            dict_res_orga_gw_T_config2,
            variable,
            global_min,
            global_max,
            "config2",
            angle = (10,230)
        )


combine_stats_csvs_by_substring("AVS_elec_main", output_filename="Combinedstats3dtotal.csv", folder="../Results/Combined_Yearly_3D")



# For SI, crop FU

list_var_impacts= [
  'AVS_crop_main_single',
  'wheat_fr_ref']



for variable in list_var_impacts:
    
    global_min, global_max = compute_global_min_max_q3(dict_res_orga_gw_T_config1, variable)

    for year in dict_res_orga_gw_T_config1:
        plot_combined_surface_all_models_per_year(
            year,
            dict_res_orga_gw_T_config1,
            variable,
            global_min,
            global_max,
            "config1",
            angle = (10,230)
        )


combine_stats_csvs_by_substring("wheat_fr_ref", output_filename="Combinedstats3wheat_augu.csv", folder="../Results/Combined_Yearly_3D")


"""
2D plot % mean below 0"""


# Separated configs


for variable in list_var_change_ratios:
    
    
    plot_percentage_below_zero_map(
        dict_res_orga_gw_T_config1,
        variable,
        "config1"
    )


for variable in list_var_change_ratios:
    
    
    plot_percentage_below_zero_map(
        dict_res_orga_gw_T_config2,
        variable,
        "config2"
    )




"""Panel figures 2 D with mean values"""


"""% dif impact"""

list_models = list(dict_res_orga_gw_T_nodoublon[2050].keys())
list_modelsconfig1 = list(dict_res_orga_gw_T_config1[2050].keys())
list_modelsforfigure=[model for model in list_modelsconfig1 if "short" in model and "ecoinvent" not in model]
chunk_size = 2
list_models_subdiv = [list_models[i:i + chunk_size] for i in range(0, len(list_models), chunk_size)]

list_models_subdiv_1 = [list_modelsconfig1[i:i + chunk_size] for i in range(0, len(list_modelsconfig1), chunk_size)]



""" Panel plots 2D for exploration"""



# Example usage
selected_models = [
    "image_pathSSP1-Base_longconfig1",
    "image_pathSSP2-Base_longconfig2"
]

variable_list = [
    'AVS_elec_main_single-PV_ref_single/PV_ref_single',
    'AVS_elec_main_single-PV_ref_single_fixed/PV_ref_single_fixed',
    'AVS_elec_main_wheat_market-PV_ref_single_fixed/PV_ref_single_fixed',
    'AVS_elec_main_wheat_market-PV_ref_single/PV_ref_single'

]



    
for selected_models in list_models_subdiv:
    
    filename= " ".join(selected_models)+"diff"



    plot_variable_panel_for_selected_models_2(filename,
        dict_res_orga_gw_T_nodoublon,
        selected_models=selected_models,
        variable_list=variable_list,
        unit_label="kg CO2-eq/kWh",
        colormap="coolwarm", #"viridis", "plasma", "coolwarm"
        center_zero=True
    )
    

"""Impacts absolute values"""


# Elec FU


variable_list = [

    'PV_ref_single',
    "PV_ref_single_fixed",

    "AVS_elec_main_single",
    "AVS_elec_main_wheat_market",
    
]


for selected_models in list_models_subdiv:
    
    filename= " ".join(selected_models)+"elecfu"



    plot_variable_panel_for_selected_models_2(filename,
        dict_res_orga_gw_T_nodoublon,
        selected_models=selected_models,
        variable_list=variable_list,
        unit_label="kg CO2-eq/kWh",
        colormap="plasma", #"viridis", "plasma", "coolwarm"
        center_zero=False
    )
    
    
    
# WHeat FU

variable_list = [
  'wheat_fr_ref'
]





for selected_models in list_models_subdiv:
    
    filename= " ".join(selected_models)+"wheatref"



    plot_variable_panel_for_selected_models_2(filename,
        dict_res_orga_gw_T_nodoublon,
        selected_models=selected_models,
        variable_list=variable_list,
        unit_label="kg CO2-eq/kWh",
        colormap="plasma", #"viridis", "plasma", "coolwarm"
        center_zero=False
    )
    
    
    
variable_list = [
    'AVS_crop_main_single'
]





for selected_models in list_models_subdiv:
    
    filename= " ".join(selected_models)+"wheatavs"



    plot_variable_panel_for_selected_models_2(filename,
        dict_res_orga_gw_T_nodoublon,
        selected_models=selected_models,
        variable_list=variable_list,
        unit_label="kg CO2-eq/kWh",
        colormap="coolwarm", #"viridis", "plasma", "coolwarm"
        center_zero=True
    )
    
    

        
    

combine_stats_csvs_by_substring("diff",output_filename="Combined_Stats_differences.csv",folder ="../Results/Panel_Plots")
combine_stats_csvs_by_substring("elecfu",output_filename="Combined_Stats_absolutet.csv",folder ="../Results/Panel_Plots")
combine_stats_csvs_by_substring("wheatref",output_filename="Combined_Stats_wheatref.csv",folder ="../Results/Panel_Plots")
combine_stats_csvs_by_substring("wheatavs",output_filename="Combined_Stats_wheatavs.csv",folder ="../Results/Panel_Plots")






var_map = {
    "differences": "Yield Difference",
    "absolutet": "Absolute Temperature",
    "wheatref": "Reference Wheat Yield",
    "wheatavs": "Average Wheat Yield"
}

combine_all_stats_to_wide_multiheader(
    file_list=[
        "Combined_Stats_differences.csv",
        "Combined_Stats_absolutet.csv",
        "Combined_Stats_wheatref.csv",
        "Combined_Stats_wheatavs.csv"
    ],
    output_filename="All_Stats_Wide3.csv",
    folder="../Results/Panel_Plots",
    excel=False,
    round_digits=4,
    var_name_map=var_map
)




# Figure 


"""Figure 2 """



filename = "Figure2D panel_plot_differentscales"

    
dict_corresp_plot_2D_panel ={'image_pathSSP2-Base_shortconfig1': "SSP2-Base-IMAGE",
                     'remind_pathSSP2-NPi_shortconfig1': "SSP2-Npi-REMIND",
                     'remind_pathSSP5-NDC_shortconfig1':"SSP5-Ndc-REMIND",
                     'remind_pathSSP1-Base_shortconfig1':"SSP1-Base-REMIND",
                     'image_pathSSP1-Base_shortconfig1':"SSP1-Base-IMAGE",
                     'remind_pathSSP2-Base_shortconfig1':"SSP2-Base-REMIND"
                     }

    
plot_variable_panel_grouped_by_column(
filename,
dict_res_orga_gw_T_nodoublon,
list_modelsforfigure,
variable_name="AVS_elec_main_single-PV_ref_single/PV_ref_single",
unit_label="% difference in kg CO2-eq/kWh",
colormap="coolwarm",
center_zero=True,
dict_plot_names=dict_corresp_plot_2D_panel
)



list_models = list(dict_res_orga_gw_T_nodoublon[2050].keys())
list_modelsconfig1 = list(dict_res_orga_gw_T_config1[2050].keys())
list_modelsforfigure=[model for model in list_modelsconfig1 if "short" in model and "ecoinvent" not in model]

# list_modelsforfigure = ecoinvent-3.10-consequential_shortconfig1
list_modelsforfigure_SI = ['image_pathSSP1-Base_shortconfig1',
 'image_pathSSP2-Base_shortconfig1',
 'remind_pathSSP2-Base_shortconfig1',
 'remind_pathSSP2-NPi_shortconfig1',
 'remind_pathSSP1-Base_shortconfig1',
 'ecoinvent-3.10-consequential_shortconfig1']


dict_corresp_plot_2D_panel_SI ={'image_pathSSP2-Base_shortconfig1': "SSP2-Base-IMAGE",
                     'remind_pathSSP2-NPi_shortconfig1': "SSP2-Npi-REMIND",
                     'ecoinvent-3.10-consequential_shortconfig1':"ecoinvent 3.10",
                     'remind_pathSSP1-Base_shortconfig1':"SSP1-Base-REMIND",
                     'image_pathSSP1-Base_shortconfig1':"SSP1-Base-IMAGE",
                     'remind_pathSSP2-Base_shortconfig1':"SSP2-Base-REMIND"
                     }


filename = "Figure 2D panel SI AVS elecmain"


    
plot_variable_panel_grouped_by_column(
filename,
dict_res_orga_gw_T_nodoublon,
list_modelsforfigure_SI,
variable_name="AVS_elec_main_single",
unit_label="kg CO2-eq/kWh",
colormap="plasma",
center_zero=False,
dict_plot_names=dict_corresp_plot_2D_panel_SI

)






filename = "Figure 2D panl SI wheatref"



    
plot_variable_panel_grouped_by_column(
filename,
dict_res_orga_gw_T_nodoublon,
list_modelsforfigure_SI,
variable_name="wheat_fr_ref",
unit_label="kg CO2-eq/kg",
colormap="plasma",
center_zero=False,
dict_plot_names=dict_corresp_plot_2D_panel_SI

)





filename = "Figure 2D panl SI avs_elecmainglobalwheat_ratio"


    
plot_variable_panel_grouped_by_column(
filename,
dict_res_orga_gw_T_nodoublon,
list_modelsforfigure,
variable_name='AVS_elec_main_wheat_market-PV_ref_single/PV_ref_single',
unit_label="kg CO2-eq/kg",
colormap="coolwarm",
center_zero=True,
dict_plot_names=dict_corresp_plot_2D_panel

)




filename = "Figure 2D panl SI avs_elecmainglobalwheat__fixed_ratio"


    
plot_variable_panel_grouped_by_column(
filename,
dict_res_orga_gw_T_nodoublon,
list_modelsforfigure,
variable_name='AVS_elec_main_wheat_market-PV_ref_single_fixed/PV_ref_single_fixed',
unit_label="kg CO2-eq/kg",
colormap="coolwarm",
center_zero=True,
dict_plot_names=dict_corresp_plot_2D_panel

)



filename = "Figure 2D panl SI avs_elecmain__fixed_ratio"



    
plot_variable_panel_grouped_by_column(
filename,
dict_res_orga_gw_T_nodoublon,
list_modelsforfigure,
variable_name='AVS_elec_main_single-PV_ref_single_fixed/PV_ref_single_fixed',
unit_label="kg CO2-eq/kg",
colormap="coolwarm",
center_zero=True,
dict_plot_names=dict_corresp_plot_2D_panel

)







# SAME scale (for paper)

    

# Config 1 


dict_corresp_plot_2D_panel ={'image_pathSSP2-Base_shortconfig1': "SSP2-Base-IMAGE",
                     'remind_pathSSP2-NPi_shortconfig1': "SSP2-Npi-REMIND",
                     'remind_pathSSP5-NDC_shortconfig1':"SSP5-Ndc-REMIND",
                     'remind_pathSSP1-Base_shortconfig1':"SSP1-Base-REMIND",
                     'image_pathSSP1-Base_shortconfig1':"SSP1-Base-IMAGE",
                     'remind_pathSSP2-Base_shortconfig1':"SSP2-Base-REMIND"
                     }

# Step 1: Compute global min and max across selected models and years
mean_values_cells = []


var_name = "AVS_elec_main_single-PV_ref_single/PV_ref_single"

var_name='AVS_elec_main_single-PV_ref_single_fixed/PV_ref_single_fixed'

for year_dict in dict_res_orga_gw_T_nodoublon.values():
    for model_name, res_model in year_dict.items():
        if model_name in selected_models:
            for (lat, lon), cell in res_model.items():
                if isinstance(cell.iloc[0], pd.DataFrame) and var_name in cell.iloc[0].columns:
                    mean_values_cells.append(np.nanmean(cell.iloc[0][var_name].values))


global_min = np.nanpercentile(mean_values_cells, 1) # more contrast
global_max = np.nanpercentile(mean_values_cells, 99)
print("Global min:", global_min)
print("Global max:", global_max)



abs_max = max(abs(global_min), abs(global_max))
norm = TwoSlopeNorm(vmin=-abs_max, vcenter=0, vmax=abs_max)





filename = "Figure2D_panel_plot_samescale"

    

    
plot_variable_panel_grouped_by_column_samescale(norm,
filename,
dict_res_orga_gw_T_nodoublon,
list_modelsforfigure,
variable_name="AVS_elec_main_single-PV_ref_single/PV_ref_single",
unit_label="% difference in kg CO2-eq/kWh",
colormap="coolwarm",
center_zero=True,
dict_plot_names=dict_corresp_plot_2D_panel
)







filename = "Figure 2D panel SI AVS elecmain_samescale"



    
plot_variable_panel_grouped_by_column(
filename,
dict_res_orga_gw_T_nodoublon,
list_modelsforfigure,
variable_name="AVS_elec_main_single",
unit_label="kg CO2-eq/kWh",
colormap="plasma",
center_zero=False,
dict_plot_names=dict_corresp_plot_2D_panel

)






filename = "Figure 2D panel SI wheatref_samescale"


plot_variable_panel_grouped_by_column_samescale(norm,
filename,
dict_res_orga_gw_T_nodoublon,
list_modelsforfigure,
variable_name="wheat_fr_ref",
unit_label="kg CO2-eq/kg",
colormap="plasma",
center_zero=False,
dict_plot_names=dict_corresp_plot_2D_panel

)





filename = "Figure 2D panl SI avs_elecmainglobalwheat_ratio_samescale"

    
plot_variable_panel_grouped_by_column_samescale(norm,
filename,
dict_res_orga_gw_T_nodoublon,
list_modelsforfigure,
variable_name='AVS_elec_main_wheat_market-PV_ref_single/PV_ref_single',
unit_label="kg CO2-eq/kg",
colormap="coolwarm",
center_zero=True,
dict_plot_names=dict_corresp_plot_2D_panel

)




filename = "Figure 2D panl SI avs_elecmainglobalwheat__fixed_ratio_samescale"


    
plot_variable_panel_grouped_by_column_samescale(norm,
filename,
dict_res_orga_gw_T_nodoublon,
list_modelsforfigure,
variable_name='AVS_elec_main_wheat_market-PV_ref_single_fixed/PV_ref_single_fixed',
unit_label="kg CO2-eq/kg",
colormap="coolwarm",
center_zero=True,
dict_plot_names=dict_corresp_plot_2D_panel

)



filename = "Figure 2D panl SI avs_elecmain__fixed_ratio_samescale"

    
plot_variable_panel_grouped_by_column_samescale(norm,
filename,
dict_res_orga_gw_T_nodoublon,
list_modelsforfigure,
variable_name='AVS_elec_main_single-PV_ref_single_fixed/PV_ref_single_fixed',
unit_label="kg CO2-eq/kg",
colormap="coolwarm",
center_zero=True,
dict_plot_names=dict_corresp_plot_2D_panel

)











"""% dif impact"""

list_models = list(dict_res_orga_gw_T_nodoublon[2050].keys())
list_modelsconfig2 = list(dict_res_orga_gw_T_config2[2050].keys())
list_modelsforfigure=[model for model in list_modelsconfig2 if "short" in model and "ecoinvent" not in model]
chunk_size = 2
list_models_subdiv = [list_models[i:i + chunk_size] for i in range(0, len(list_models), chunk_size)]

list_models_subdiv_2 = [list_modelsconfig2[i:i + chunk_size] for i in range(0, len(list_modelsconfig2), chunk_size)]



""" Panel plots 2D for exploration"""




selected_models = [
    "image_pathSSP1-Base_longconfig1",
    "image_pathSSP2-Base_longconfig2"
]

variable_list = [
    'AVS_elec_main_single-PV_ref_single/PV_ref_single',
    'AVS_elec_main_single-PV_ref_single_fixed/PV_ref_single_fixed',
    'AVS_elec_main_wheat_market-PV_ref_single_fixed/PV_ref_single_fixed',
    'AVS_elec_main_wheat_market-PV_ref_single/PV_ref_single'

]




dict_corresp_plot_2D_panel ={'image_pathSSP2-Base_shortconfig2': "SSP2-Base-IMAGE",
                     'remind_pathSSP2-NPi_shortconfig2': "SSP2-Npi-REMIND",
                     'remind_pathSSP5-NDC_shortconfig2':"SSP5-Ndc-REMIND",
                     'remind_pathSSP1-Base_shortconfig2':"SSP1-Base-REMIND",
                     'image_pathSSP1-Base_shortconfig2':"SSP1-Base-IMAGE",
                     'remind_pathSSP2-Base_shortconfig2':"SSP2-Base-REMIND"
                     }

# Step 1: Compute global min and max across selected models and years
mean_values_cells = []


var_name = "AVS_elec_main_single-PV_ref_single/PV_ref_single"

var_name='AVS_elec_main_single-PV_ref_single_fixed/PV_ref_single_fixed'

for year_dict in dict_res_orga_gw_T_nodoublon.values():
    for model_name, res_model in year_dict.items():
        if model_name in selected_models:
            for (lat, lon), cell in res_model.items():
                if isinstance(cell.iloc[0], pd.DataFrame) and var_name in cell.iloc[0].columns:
                    mean_values_cells.append(np.nanmean(cell.iloc[0][var_name].values))



global_min = np.nanpercentile(mean_values_cells, 1) # more contrast
global_max = np.nanpercentile(mean_values_cells, 99)
print("Global min:", global_min)
print("Global max:", global_max)



abs_max = max(abs(global_min), abs(global_max))
norm = TwoSlopeNorm(vmin=-abs_max, vcenter=0, vmax=abs_max)





filename = "Figure2D panel plot_samescale_config2"

    

    
plot_variable_panel_grouped_by_column_samescale(norm,
filename,
dict_res_orga_gw_T_nodoublon,
list_modelsforfigure,
variable_name="AVS_elec_main_single-PV_ref_single/PV_ref_single",
unit_label="% difference in kg CO2-eq/kWh",
colormap="coolwarm",
center_zero=True,
dict_plot_names=dict_corresp_plot_2D_panel
)




filename = "Figure 2D panl SI AVS elecmain_samescale_config2"


    
plot_variable_panel_grouped_by_column(
filename,
dict_res_orga_gw_T_nodoublon,
list_modelsforfigure,
variable_name="AVS_elec_main_single",
unit_label="kg CO2-eq/kWh",
colormap="plasma",
center_zero=False,
dict_plot_names=dict_corresp_plot_2D_panel

)




filename = "Figure 2D panl SI wheatref_samescale_config2"


    
plot_variable_panel_grouped_by_column_samescale(norm,
filename,
dict_res_orga_gw_T_nodoublon,
list_modelsforfigure,
variable_name="wheat_fr_ref",
unit_label="kg CO2-eq/kg",
colormap="plasma",
center_zero=False,
dict_plot_names=dict_corresp_plot_2D_panel

)





filename = "Figure 2D panl SI avs_elecmainglobalwheat_ratio_samescale_config2"



    
plot_variable_panel_grouped_by_column_samescale(norm,
filename,
dict_res_orga_gw_T_nodoublon,
list_modelsforfigure,
variable_name='AVS_elec_main_wheat_market-PV_ref_single/PV_ref_single',
unit_label="kg CO2-eq/kg",
colormap="coolwarm",
center_zero=True,
dict_plot_names=dict_corresp_plot_2D_panel

)




filename = "Figure 2D panl SI avs_elecmainglobalwheat__fixed_ratio_samescale_config2"


    
plot_variable_panel_grouped_by_column_samescale(norm,
filename,
dict_res_orga_gw_T_nodoublon,
list_modelsforfigure,
variable_name='AVS_elec_main_wheat_market-PV_ref_single_fixed/PV_ref_single_fixed',
unit_label="kg CO2-eq/kg",
colormap="coolwarm",
center_zero=True,
dict_plot_names=dict_corresp_plot_2D_panel

)



filename = "Figure 2D panl SI avs_elecmain__fixed_ratio_samescale_config2"


    
plot_variable_panel_grouped_by_column_samescale(norm,
filename,
dict_res_orga_gw_T_nodoublon,
list_modelsforfigure,
variable_name='AVS_elec_main_single-PV_ref_single_fixed/PV_ref_single_fixed',
unit_label="kg CO2-eq/kg",
colormap="coolwarm",
center_zero=True,
dict_plot_names=dict_corresp_plot_2D_panel

)










" SI relative change between years"


list_modelsforfigure=[model for model in list_modelsconfig1 if "short" in model and "ecoinvent" not in model]

list_modelsforfigure_si_relativechanges=[item for item in list_modelsforfigure if item != "remind_pathSSP2-NPi_shortconfig1"] + ["ecoinvent-3.10-consequential_shortconfig1"]


plot_variable_panel_percent_change_grouped_by_column(
    filename="relative_change_panel_withe310tosplit",
    dict_res_orga_gw_T=dict_res_orga_gw_T_nodoublon,
    selected_models=list_modelsforfigure_si_relativechanges,
    variable_name="wheat_fr_ref",
    unit_label="% change",
    colormap="coolwarm",
    center_zero=True
)



# AVS_elec_main_single

plot_variable_panel_percent_change_grouped_by_column(
    filename="relative_change_panel_withe310tosplit",
    dict_res_orga_gw_T=dict_res_orga_gw_T_nodoublon,
    selected_models=list_modelsforfigure_si_relativechanges,
    variable_name="AVS_elec_main_single",
    unit_label="% change",
    colormap="coolwarm",
    center_zero=True
)


plot_variable_panel_percent_change_grouped_by_column(
    filename="relative_change_panel_withe310tosplit",
    dict_res_orga_gw_T=dict_res_orga_gw_T_nodoublon,
    selected_models=list_modelsforfigure_si_relativechanges,
    variable_name="AVS_elec_main_single-PV_ref_single/PV_ref_single",
    unit_label="% change",
    colormap="coolwarm",
    center_zero=True
)





        


""" Panel Figure 3"""





list_var_impacts=['AVS_elec_main_single','PV_ref_single']

for variable in list_var_impacts:
    
        
    # Step 1: Collect Q3 values for color normalization
    global_min, global_max = compute_global_min_max_q3(dict_res_orga_gw_T_nodoublon, variable)

    
    nlist = 1
    uncert = 20
    
    
    for year, year_dict in dict_res_orga_gw_T_nodoublon.items():
        for model, res_model in year_dict.items():
            plot_row_combined_surface_and_percentage(
                res_model=res_model,
                model=model,
                year=year,
                variable_name=variable,
                vmin=global_min,
                vmax=global_max,
                rows_per_model=uncert * nlist,  # Ensure these are defined
                cmap_3d="plasma", #"viridis", "plasma", "coolwarm"
                center_zero=False)
    

    





list_var_change_ratios = ['AVS_elec_main_single-PV_ref_single/PV_ref_single']

for variable in list_var_change_ratios:
    
    
        
    # Step 1: Collect Q3 values for color normalization
    global_min, global_max = compute_global_min_max_q3(dict_res_orga_gw_T_nodoublon, variable)
    nlist = 1
    uncert = 20
    
    
    for year, year_dict in dict_res_orga_gw_T_nodoublon.items():
        for model, res_model in year_dict.items():
            plot_row_combined_surface_and_percentage(
                res_model=res_model,
                model=model,
                year=year,
                variable_name=variable,
                vmin=global_min,
                vmax=global_max,
                rows_per_model=uncert * nlist,  # Ensure these are defined
                angle=(60, 225),
                cmap_3d="coolwarm", #"viridis", "plasma", "coolwarm"
                cmap_perc='RdBu',

                center_zero=True)
            
            
    for year, year_dict in dict_res_orga_gw_T_nodoublon.items():
        for model, res_model in year_dict.items():
            plot_row_combined_surface_and_percentage(
                res_model=res_model,
                model=model,
                year=year,
                variable_name=variable,
                vmin=global_min,
                vmax=global_max,
                rows_per_model=uncert * nlist,  # Ensure these are defined
                angle = (10,230),

                cmap_3d="coolwarm", #"viridis", "plasma", "coolwarm"
                cmap_perc='RdBu',

                center_zero=True)
    
   
    dict_res_orga_gw_T_nodoublon.keys()
    



    



# config1



"""SI COMPLEMENT FIGURE 3"""

# Conf 1 

list_var_change_ratios = ['AVS_elec_main_single-PV_ref_single/PV_ref_single']

dict_corresp_plot_2D_panel ={'image_pathSSP2-Base_shortconfig1': "SSP2-Base-IMAGE",
                     'remind_pathSSP2-NPi_shortconfig1': "SSP2-Npi-REMIND",
                     'remind_pathSSP5-NDC_shortconfig1':"SSP5-Ndc-REMIND",
                     'remind_pathSSP1-Base_shortconfig1':"SSP1-Base-REMIND",
                     'image_pathSSP1-Base_shortconfig1':"SSP1-Base-IMAGE",
                     'remind_pathSSP2-Base_shortconfig1':"SSP2-Base-REMIND"
                     }



list_models = list(dict_res_orga_gw_T_nodoublon[2050].keys())
list_modelsconfig1 = list(dict_res_orga_gw_T_config1[2050].keys())
list_modelsforfigure1=[model for model in list_modelsconfig1 if "short" in model and "ecoinvent" not in model]
chunk_size = 2
list_models_subdiv = [list_models[i:i + chunk_size] for i in range(0, len(list_models), chunk_size)]

list_models_subdiv_1 = [list_modelsconfig1[i:i + chunk_size] for i in range(0, len(list_modelsconfig1), chunk_size)]


list_var_change_ratios = ['AVS_elec_main_single-PV_ref_single/PV_ref_single',
                          "AVS_elec_main_wheat_market-PV_ref_single/PV_ref_single",
                          "AVS_elec_main_wheat_market-PV_ref_single/PV_ref_single",
                          "AVS_elec_main_wheat_market-PV_ref_single/PV_ref_single",
                          "AVS_elec_main_wheat_market-PV_ref_single_fixed/PV_ref_single_fixed",
                          "AVS_elec_main_single-PV_ref_single_fixed/PV_ref_single_fixed",
                          "AVS_elec_main_single-PV_ref_single_fixed/PV_ref_single_fixed"]

# Right order
list_modelsforfigure1 = ['remind_pathSSP2-NPi_shortconfig1', 'image_pathSSP1-Base_shortconfig1', 'remind_pathSSP1-Base_shortconfig1','image_pathSSP2-Base_shortconfig1' , 'remind_pathSSP2-Base_shortconfig1', 'remind_pathSSP5-NDC_shortconfig1' ]




# Common scale

global_min = 1000
global_max = -1000

for variable in list_var_change_ratios:
    
    
        
    # Step 1: Collect Q3 values for color normalization
    global_min_var, global_max_var = compute_global_min_max_q3(dict_res_orga_gw_T_nodoublon, variable)
    
    global_min = min([global_min_var,global_min])
    global_max = max([global_max_var,global_max])

plot_variable_panel_grouped_by_column_q1q3surface(
    filename="fig3_SI_conf1_local_st",
    dict_res_orga_gw_T=dict_res_orga_gw_T_nodoublon,
    selected_models=list_modelsforfigure1,
    variable="AVS_elec_main_single-PV_ref_single/PV_ref_single",
    unit_label="Ratio",
    vmin=global_min,
    vmax=global_max,
    cmap_3d="coolwarm",
    center_zero=True,
    dict_plot_names=dict_corresp_plot_2D_panel,
    angle=(60, 225))
# )



plot_variable_panel_grouped_by_column_q1q3surface(
    filename="fig3_SI_conf1_global_st",
    dict_res_orga_gw_T=dict_res_orga_gw_T_nodoublon,
    selected_models=list_modelsforfigure1,
    variable="AVS_elec_main_wheat_market-PV_ref_single/PV_ref_single",
    unit_label="Ratio",
    vmin=global_min,
    vmax=global_max,
    cmap_3d="coolwarm",
    center_zero=True,
    dict_plot_names=dict_corresp_plot_2D_panel,
    angle=(60, 225))




plot_variable_panel_grouped_by_column_q1q3surface(
    filename="fig3_SI_conf1_global_fixed",
    dict_res_orga_gw_T=dict_res_orga_gw_T_nodoublon,
    selected_models=list_modelsforfigure1,
    variable="AVS_elec_main_wheat_market-PV_ref_single_fixed/PV_ref_single_fixed",
    unit_label="Ratio",
    vmin=global_min,
    vmax=global_max,
    cmap_3d="coolwarm",
    center_zero=True,
    dict_plot_names=dict_corresp_plot_2D_panel,
    angle=(60, 225))



plot_variable_panel_grouped_by_column_q1q3surface(
    filename="forscale_ratio", #"forscale_ratio",fig3_SI_conf1_local_fixed
    dict_res_orga_gw_T=dict_res_orga_gw_T_nodoublon,
    selected_models=list_modelsforfigure1,
    variable="AVS_elec_main_single-PV_ref_single_fixed/PV_ref_single_fixed",
    unit_label="Ratio",
    vmin=global_min,
    vmax=global_max,
    cmap_3d="coolwarm",
    center_zero=True,
    dict_plot_names=dict_corresp_plot_2D_panel,
    angle=(60, 225))








# Conf 2 



dict_corresp_plot_2D_panel ={'image_pathSSP2-Base_shortconfig2': "SSP2-Base-IMAGE",
                     'remind_pathSSP2-NPi_shortconfig2': "SSP2-Npi-REMIND",
                     'remind_pathSSP5-NDC_shortconfig2':"SSP5-Ndc-REMIND",
                     'remind_pathSSP1-Base_shortconfig2':"SSP1-Base-REMIND",
                     'image_pathSSP1-Base_shortconfig2':"SSP1-Base-IMAGE",
                     'remind_pathSSP2-Base_shortconfig2':"SSP2-Base-REMIND"
                     }


list_models = list(dict_res_orga_gw_T_nodoublon[2050].keys())
list_modelsconfig2 = list(dict_res_orga_gw_T_config2[2050].keys())
list_modelsforfigure2=[model for model in list_modelsconfig2 if "short" in model and "ecoinvent" not in model]
chunk_size = 2
list_models_subdiv = [list_models[i:i + chunk_size] for i in range(0, len(list_models), chunk_size)]

list_models_subdiv_2 = [list_modelsconfig2[i:i + chunk_size] for i in range(0, len(list_modelsconfig2), chunk_size)]


list_modelsforfigure2= ['remind_pathSSP2-NPi_shortconfig2', 'image_pathSSP1-Base_shortconfig2', 'remind_pathSSP1-Base_shortconfig2','image_pathSSP2-Base_shortconfig2' , 'remind_pathSSP2-Base_shortconfig2', 'remind_pathSSP5-NDC_shortconfig2' ]





dict_res_orga_gw_T_nodoublon[2070].keys()

plot_variable_panel_grouped_by_column_q1q3surface(
    filename="fig3_SI_conf1_local_st_conf2",
    dict_res_orga_gw_T=dict_res_orga_gw_T_nodoublon,
    selected_models=list_modelsforfigure2,
    variable="AVS_elec_main_single-PV_ref_single/PV_ref_single",
    unit_label="Ratio",
    vmin=global_min,
    vmax=global_max,
    cmap_3d="coolwarm",
    center_zero=True,
    dict_plot_names=dict_corresp_plot_2D_panel,
    angle=(60, 225))
# )



plot_variable_panel_grouped_by_column_q1q3surface(
    filename="fig3_SI_conf1_global_st_conf2",
    dict_res_orga_gw_T=dict_res_orga_gw_T_nodoublon,
    selected_models=list_modelsforfigure2,
    variable="AVS_elec_main_wheat_market-PV_ref_single/PV_ref_single",
    unit_label="Ratio",
    vmin=global_min,
    vmax=global_max,
    cmap_3d="coolwarm",
    center_zero=True,
    dict_plot_names=dict_corresp_plot_2D_panel,
    angle=(60, 225))




plot_variable_panel_grouped_by_column_q1q3surface(
    filename="fig3_SI_conf1_global_fixed_conf2",
    dict_res_orga_gw_T=dict_res_orga_gw_T_nodoublon,
    selected_models=list_modelsforfigure2,
    variable="AVS_elec_main_wheat_market-PV_ref_single_fixed/PV_ref_single_fixed",
    unit_label="Ratio",
    vmin=global_min,
    vmax=global_max,
    cmap_3d="coolwarm",
    center_zero=True,
    dict_plot_names=dict_corresp_plot_2D_panel,
    angle=(60, 225))



plot_variable_panel_grouped_by_column_q1q3surface(
    filename="fig3_SI_conf1_local_fixed_conf2",
    dict_res_orga_gw_T=dict_res_orga_gw_T_nodoublon,
    selected_models=list_modelsforfigure2,
    variable="AVS_elec_main_single-PV_ref_single_fixed/PV_ref_single_fixed",
    unit_label="Ratio",
    vmin=global_min,
    vmax=global_max,
    cmap_3d="coolwarm",
    center_zero=True,
    dict_plot_names=dict_corresp_plot_2D_panel,
    angle=(60, 225))




"""Long vs short lasting"""


dict_corresp_plot_2D_panel_long ={'image_pathSSP2-Base_longconfig1': "SSP2-Base-IMAGE",
                     'remind_pathSSP2-NPi_longconfig1': "SSP2-Npi-REMIND",
                     'remind_pathSSP5-NDC_longconfig1':"SSP5-Ndc-REMIND",
                     'remind_pathSSP1-Base_longconfig1':"SSP1-Base-REMIND",
                     'image_pathSSP1-Base_longconfig1':"SSP1-Base-IMAGE",
                     'remind_pathSSP2-Base_longconfig1':"SSP2-Base-REMIND"
                     }


list_models = list(dict_res_orga_gw_T_nodoublon[2050].keys())
list_modelsconfig1 = list(dict_res_orga_gw_T_config1[2050].keys())
list_modelsforfigure1_long=[model for model in list_modelsconfig1 if "long" in model and "ecoinvent" not in model]
chunk_size = 2
list_models_subdiv = [list_models[i:i + chunk_size] for i in range(0, len(list_models), chunk_size)]

list_models_subdiv_1 = [list_modelsconfig1[i:i + chunk_size] for i in range(0, len(list_modelsconfig1), chunk_size)]


list_modelsforfigure_1_long= ['remind_pathSSP2-NPi_longconfig1', 'image_pathSSP1-Base_longconfig1', 'remind_pathSSP1-Base_longconfig1','image_pathSSP2-Base_longconfig1' , 'remind_pathSSP2-Base_longconfig1', 'remind_pathSSP5-NDC_longconfig1' ]



plot_variable_panel_grouped_by_column_q1q3surface(
    filename="fig3_SI_conf1_local_st_long",
    dict_res_orga_gw_T=dict_res_orga_gw_T_nodoublon,
    selected_models=list_modelsforfigure_1_long,
    variable="AVS_elec_main_single-PV_ref_single/PV_ref_single",
    unit_label="Ratio",
    vmin=global_min,
    vmax=global_max,
    cmap_3d="coolwarm",
    center_zero=True,
    dict_plot_names=dict_corresp_plot_2D_panel_long,
    angle=(60, 225))
# )

    

"""Absolute score PV"""

    
    
        
    # Step 1: Collect Q3 values for color normalization
global_min_var, global_max_var = compute_global_min_max_q3(dict_res_orga_gw_T_nodoublon, "PV_ref_single")

plot_variable_panel_grouped_by_column_q1q3surface(
    filename="fig3_SI_PVref",
    dict_res_orga_gw_T=dict_res_orga_gw_T_nodoublon,
    selected_models=list_modelsforfigure2,
    variable="PV_ref_single",
    unit_label="Ratio",
    vmin=global_min_var,
    vmax=global_max_var,
    cmap_3d="plasma",
    center_zero=False,
    dict_plot_names=dict_corresp_plot_2D_panel,
    angle=(60, 225))
# )

    


    
    
    
"""Figure 4 CROP main """

list_var = ['AVS_crop_main_single']




for variable in list_var:
    
        
    # Step 1: Collect Q3 values for color normalization
    global_min, global_max = compute_global_min_max_q3(dict_res_orga_gw_T_nodoublon, variable)

    
    nlist = 1
    uncert = 20
    
    
    for year, year_dict in dict_res_orga_gw_T_nodoublon.items():
        for model, res_model in year_dict.items():
            plot_row_combined_surface_and_percentage(
                res_model=res_model,
                model=model,
                year=year,
                variable_name=variable,
                vmin=global_min,
                vmax=global_max,
                rows_per_model=uncert * nlist,  # Ensure these are defined
                angle=(60, 225),

                cmap_3d="plasma", #"viridis", "plasma", "coolwarm"
                center_zero=False)
    
    
    for year, year_dict in dict_res_orga_gw_T_nodoublon.items():
        for model, res_model in year_dict.items():
            plot_row_combined_surface_and_percentage(
                res_model=res_model,
                model=model,
                year=year,
                variable_name=variable,
                vmin=global_min,
                vmax=global_max,
                rows_per_model=uncert * nlist,  # Ensure these are defined
                angle= (10,230),

                cmap_3d="plasma", #"viridis", "plasma", "coolwarm"
                center_zero=False)
    
    



for variable in list_var:
    
        
    # Step 1: Collect Q3 values for color normalization

    global_min, global_max = compute_global_min_max_q3(dict_res_orga_gw_T_nodoublon, variable)

    nlist = 1
    uncert = 20
    
    
    for year, year_dict in dict_res_orga_gw_T_nodoublon.items():
        for model, res_model in year_dict.items():
            
            if model == "remind_pathSSP1-Base_shortconfig1":
                

                plot_row_combined_surface_and_percentage(
                    res_model=res_model,
                    model=model,
                    year=year,
                    variable_name=variable,
                    vmin=global_min,
                    vmax=global_max,
                    rows_per_model=uncert * nlist,  # Ensure these are defined
                    angle=(60, 225),
    
                    cmap_3d="plasma", #"viridis", "plasma", "coolwarm"
                    center_zero=False)
    
    
                plot_row_combined_surface_and_percentage(
                    res_model=res_model,
                    model=model,
                    year=year,
                    variable_name=variable,
                    vmin=global_min,
                    vmax=global_max,
                    rows_per_model=uncert * nlist,  # Ensure these are defined
                    angle= (10,230),
    
                    cmap_3d="plasma", #"viridis", "plasma", "coolwarm"
                    center_zero=False)
        
        
    


for variable in list_var:
    
        
    # Step 1: Collect Q3 values for color normalization

    global_min, global_max = compute_global_min_max_q3(dict_res_orga_gw_T_nodoublon, variable)

    nlist = 1
    uncert = 20
    
    
    for year, year_dict in dict_res_orga_gw_T_nodoublon.items():
        for model, res_model in year_dict.items():
            # print(model)
            
            if model == "ecoinvent-3.10-consequential_shortconfig1":
                
                dict_res_orga_gw_T_sub = {year: {model:dict_res_orga_gw_T_nodoublon[year][model]}}
                dict_res_orga_gw_T_sub.keys()
                global_min, global_max = compute_global_min_max_q3(dict_res_orga_gw_T_sub, variable)
    
                plot_row_combined_surface_and_percentage(
                    res_model=res_model,
                    model=model,
                    year=year,
                    variable_name=variable,
                    vmin=global_min,
                    vmax=global_max,
                    rows_per_model=uncert * nlist,  # Ensure these are defined
                    angle=(60, 225),
    
                    cmap_3d="plasma", #"viridis", "plasma", "coolwarm"
                    center_zero=False)
    
    
                plot_row_combined_surface_and_percentage(
                    res_model=res_model,
                    model=model,
                    year=year,
                    variable_name=variable,
                    vmin=global_min,
                    vmax=global_max,
                    rows_per_model=uncert * nlist,  # Ensure these are defined
                    angle= (10,230),
    
                    cmap_3d="plasma", #"viridis", "plasma", "coolwarm"
                    center_zero=False)
        
        
        
        
# SI Crop main panel, figure 4






    
    
        
    # Step 1: Collect Q3 values for color normalization
global_min_var, global_max_var = compute_global_min_max_q3(dict_res_orga_gw_T_nodoublon, "AVS_crop_main_single")

# conf1 
plot_variable_panel_grouped_by_column_q1q3surface(
    filename="fig4_SI_cropmain_conf1",
    dict_res_orga_gw_T=dict_res_orga_gw_T_nodoublon,
    selected_models=list_modelsforfigure1,
    variable="AVS_crop_main_single",
    unit_label="Ratio",
    vmin=global_min_var,
    vmax=global_max_var,
    cmap_3d="plasma",
    center_zero=False,
    dict_plot_names=dict_corresp_plot_2D_panel,
    angle=(60, 225))
# )



# conf2
plot_variable_panel_grouped_by_column_q1q3surface(
    filename="fig4_SI_cropmain_conf2",
    dict_res_orga_gw_T=dict_res_orga_gw_T_nodoublon,
    selected_models=list_modelsforfigure2,
    variable="AVS_crop_main_single",
    unit_label="Ratio",
    vmin=global_min_var,
    vmax=global_max_var,
    cmap_3d="plasma",
    center_zero=False,
    dict_plot_names=dict_corresp_plot_2D_panel,
    angle=(60, 225))
# )







"""Export full stats for  Fig 5 (with R)"""


for variable in list_var_change_ratios+list_var_impacts:
    
    export_distributions_by_ssp_and_year(
        dict_res_orga_gw_T_nodoublon,
        variable_name=variable
    )



# Exports csvs to plot with R


for variable in list_var_change_ratios+list_var_impacts:
    
    export_distributions_by_ssp_and_year(
        dict_res_orga_gw_T_nodoublon,
        variable_name=variable
    )

list_var = ['AVS_crop_main_single']

for variable in list_var:


    
    export_distributions_by_ssp_and_year(
        dict_res_orga_gw_T_nodoublon,
        variable_name=variable
    )



# Param
for param in ["yield_wheat", "yield_wheat_avs"]:


        export_distributions_by_ssp_and_year(
            dict_param_orga,
            variable_name=param
        )




def iam_name(model_name):
    
    iam_match = re.match(r'([^_]+)_path', model_name)
    iam = iam_match.group(1) if iam_match else 'Unknown'
    return iam


def margi_name(model_name):
        
    marginal_match = re.search(r'_([^_]+)config', model_name)
    marginal = marginal_match.group(1) if marginal_match else 'Unknown'
    return marginal

def config_name(model_name):
        
    config_match = re.search(r'config(\w+)', model_name)
    config = config_match.group(1) if config_match else 'Unknown'
    
    return config



def extract_ssp(model_name):
    match = re.search(r'path(.*?)_(?:long|short)', model_name)
    return match.group(1) if match else 'Unknown'

def build_flat_dataframe(dict_res, variable_name):
    data_rows = []
    for year, year_dict in dict_res.items():
        for model, res_model in year_dict.items():
            ssp = extract_ssp(model)
            for (lat, lon), cell in res_model.items():
                if isinstance(cell.iloc[0], pd.DataFrame) and variable_name in cell.iloc[0].columns:
                    values = cell.iloc[0][variable_name].values
                    for val in values:
                        data_rows.append({
                            "value": val,
                            "model": model,
                            "ssp": ssp,
                            "year": int(year)
                        })
    return pd.DataFrame(data_rows)

df_flat = build_flat_dataframe(dict_res_orga_gw_T_nodoublon, variable_name='AVS_elec_main_single-PV_ref_single/PV_ref_single')
df_flat.columns

df_flat["IAM"] = df_flat['model'].apply(iam_name)
df_flat["config"] = df_flat['model'].apply(config_name)
df_flat["margi"] = df_flat['model'].apply(margi_name)


df_flat=df_flat[df_flat["ssp"]!="Unknown"]




ssp = list(np.unique(df_flat["ssp"])) 
years= list(np.unique(df_flat["year"]))
unique_combinations = df_flat[['ssp', 'year']].drop_duplicates()
unique_combinations["ssp"].iloc[2]
list_tupples = [(unique_combinations["ssp"].iloc[a],unique_combinations["year"].iloc[a]) for a in range(unique_combinations.shape[0]) ]



df_flat.to_csv("ALlvaluesfordistrib.csv", index=False)







df_flat = build_flat_dataframe(dict_res_orga_gw_T_nodoublon, variable_name='AVS_elec_main_single-PV_ref_single/PV_ref_single')
df_flat.columns

df_flat["IAM"] = df_flat['model'].apply(iam_name)
df_flat["config"] = df_flat['model'].apply(config_name)
df_flat["margi"] = df_flat['model'].apply(margi_name)


df_flat=df_flat[df_flat["ssp"]!="Unknown"]

df_flat["Wheat market"]="local"
df_flat["PV comparison"]="ST"


df_flat_wheatmarket = build_flat_dataframe(dict_res_orga_gw_T_nodoublon, variable_name='AVS_elec_main_wheat_market-PV_ref_single/PV_ref_single')



df_flat_wheatmarket["IAM"] = df_flat_wheatmarket['model'].apply(iam_name)
df_flat_wheatmarket["config"] = df_flat_wheatmarket['model'].apply(config_name)
df_flat_wheatmarket["margi"] = df_flat_wheatmarket['model'].apply(margi_name)
df_flat_wheatmarket["Wheat market"]="global"
df_flat_wheatmarket["PV comparison"]="ST"

df_flat_wheatmarket=df_flat_wheatmarket[df_flat_wheatmarket["ssp"]!="Unknown"]

df_flat_wheatmarket_singlepv = build_flat_dataframe(dict_res_orga_gw_T_nodoublon, variable_name='AVS_elec_main_single-PV_ref_single_fixed/PV_ref_single_fixed')


df_flat_wheatmarket_singlepv["IAM"] = df_flat_wheatmarket_singlepv['model'].apply(iam_name)
df_flat_wheatmarket_singlepv["config"] = df_flat_wheatmarket_singlepv['model'].apply(config_name)
df_flat_wheatmarket_singlepv["margi"] = df_flat_wheatmarket_singlepv['model'].apply(margi_name)
df_flat_wheatmarket_singlepv["Wheat market"]="local"
df_flat_wheatmarket_singlepv["PV comparison"]="fixed"

df_flat_wheatmarket_singlepv=df_flat_wheatmarket_singlepv[df_flat_wheatmarket_singlepv["ssp"]!="Unknown"]


df_flat_wheatmarket_singlepv_fixed = build_flat_dataframe(dict_res_orga_gw_T_nodoublon, variable_name='AVS_elec_main_wheat_market-PV_ref_single_fixed/PV_ref_single_fixed')


df_flat_wheatmarket_singlepv_fixed["IAM"] = df_flat_wheatmarket_singlepv_fixed['model'].apply(iam_name)
df_flat_wheatmarket_singlepv_fixed["config"] = df_flat_wheatmarket_singlepv_fixed['model'].apply(config_name)
df_flat_wheatmarket_singlepv_fixed["margi"] = df_flat_wheatmarket_singlepv_fixed['model'].apply(margi_name)
df_flat_wheatmarket_singlepv_fixed["Wheat market"]="global"
df_flat_wheatmarket_singlepv_fixed["PV comparison"]="fixed"

df_flat_wheatmarket_singlepv_fixed=df_flat_wheatmarket_singlepv_fixed[df_flat_wheatmarket_singlepv_fixed["ssp"]!="Unknown"]


df_flat.to_csv("ALlvaluesfordistrib.csv", index=False)

df_total = pd.concat([df_flat,df_flat_wheatmarket_singlepv_fixed,df_flat_wheatmarket_singlepv,df_flat_wheatmarket])

df_total.to_csv("ALlvaluesfordistrib_total.csv", index=False)






# add crop main


df_flat_cropmain = build_flat_dataframe(dict_res_orga_gw_T_nodoublon, variable_name='AVS_crop_main_single')


df_flat_cropmain["IAM"] = df_flat_cropmain['model'].apply(iam_name)
df_flat_cropmain["config"] = df_flat_cropmain['model'].apply(config_name)
df_flat_cropmain["margi"] = df_flat_cropmain['model'].apply(margi_name)


df_flat_cropmain=df_flat_cropmain[df_flat_cropmain["ssp"]!="Unknown"]

df_flat_cropmain.to_csv("ALlvaluesfordistrib_cropmain.csv", index=False)

# Crop main differences


df_flat_cropmain_dif = build_flat_dataframe(dict_res_orga_gw_T_nodoublon, variable_name='AVS_crop_main_single-wheat_fr_ref/wheat_fr_ref')


df_flat_cropmain_dif["IAM"] = df_flat_cropmain_dif['model'].apply(iam_name)
df_flat_cropmain_dif["config"] = df_flat_cropmain_dif['model'].apply(config_name)
df_flat_cropmain_dif["margi"] = df_flat_cropmain_dif['model'].apply(margi_name)


df_flat_cropmain_dif=df_flat_cropmain_dif[df_flat_cropmain_dif["ssp"]!="Unknown"]

df_flat_cropmain_dif.to_csv("ALlvaluesfordistrib_cropmain_dif.csv", index=False)







"""Param plots"""





# crop_yield_ratio_avs_ref


mean_values = []

for year, year_dict in dict_param_orga.items():
    for model, res_model in year_dict.items():
        for col_name, cell in res_model.items():
            if isinstance(cell.iloc[0], pd.DataFrame) and 'crop_yield_ratio_avs_ref' in cell.iloc[0].columns:
                values = cell.iloc[0]['crop_yield_ratio_avs_ref'].values
                mean_values.append(np.mean(values))

# Step 2: Compute mean of min values and mean of max values
global_min = np.min(mean_values) 
global_max = np.max(mean_values) 

# Step 3: Generate and save plots with the computed common scale
for year, year_dict in dict_param_orga.items():
    for model, res_model in year_dict.items():
        plot_variable_2d_mean_white_samescale(res_model, model, str(year), 'crop_yield_ratio_avs_ref', ".", global_min, global_max)






for year, year_dict in dict_param_orga.items():
    for model, res_model in year_dict.items():
        for col_name, cell in res_model.items():
            a=cell
            b=col_name
            
            cell.iloc[0]["yield_wheat"]= cell.iloc[0]["correc_orchidee"]*cell.iloc[0]["crop_yield_upd_ref"]*6752.650415288
            cell.iloc[0]["yield_wheat_avs"]= cell.iloc[0]["yield_wheat"] * cell.iloc[0]["crop_yield_ratio_avs_ref"]


param = "yield_wheat"



mean_values = []

for year, year_dict in dict_param_orga.items():
    for model, res_model in year_dict.items():
        for col_name, cell in res_model.items():
            if isinstance(cell.iloc[0], pd.DataFrame) and param in cell.iloc[0].columns:
                values = cell.iloc[0][param].values
                mean_values.append(np.mean(values))

# Step 2: Compute mean of min values and mean of max values
global_min = np.min(mean_values) 
global_max = np.max(mean_values) 

# Step 3: Generate and save plots with the computed common scale
for year, year_dict in dict_param_orga.items():
    for model, res_model in year_dict.items():
        plot_variable_2d_mean_white_samescale(res_model, model, str(year), param, ".", global_min, global_max)





param = "yield_wheat_avs"



mean_values = []

for year, year_dict in dict_param_orga.items():
    for model, res_model in year_dict.items():
        for col_name, cell in res_model.items():
            if isinstance(cell.iloc[0], pd.DataFrame) and param in cell.iloc[0].columns:
                values = cell.iloc[0][param].values
                mean_values.append(np.mean(values))

# Step 3: Generate and save plots with the computed common scale
for year, year_dict in dict_param_orga.items():
    for model, res_model in year_dict.items():
        plot_variable_2d_mean_white_samescale(res_model, model, str(year), param, ".", global_min, global_max)





param = "crop_yield_ratio_avs_ref"



mean_values = []

for year, year_dict in dict_param_orga.items():
    for model, res_model in year_dict.items():
        for col_name, cell in res_model.items():
            print(cell.iloc[0].columns)
            if isinstance(cell.iloc[0], pd.DataFrame) and param in cell.iloc[0].columns:
                values = cell.iloc[0][param].values
                mean_values.append(np.mean(values))


# Step 2: Compute mean of min values and mean of max values
global_min = np.min(mean_values) 
global_max = np.max(mean_values) 

# Step 3: Generate and save plots with the computed common scale
for year, year_dict in dict_param_orga.items():
    for model, res_model in year_dict.items():
        plot_variable_2d_mean_white_samescale(res_model, model, str(year), param, ".", global_min, global_max)



# Mean values evolution
param = "crop_yield_upd_ref"

res_year = {}

for year, year_dict in dict_param_orga.items():
    for model, res_model in year_dict.items():
        mean_values_model =[]
        for col_name, cell in res_model.items():
            print(cell.iloc[0].columns)
            if isinstance(cell.iloc[0], pd.DataFrame) and param in cell.iloc[0].columns:
                values = cell.iloc[0][param].values
                mean_values_model.append(np.mean(values)) # Will be a unique value
                
        res_year[year] = mean_values_model # not the most effcient cause it will overwrite the last values for the last model but it works ok

res_year_crop_yield_upd_ref= {key:[np.mean(res_year[key]),np.median(res_year[key])] for key in res_year.keys()}

res_year.plot()








# Plot param evolution for SI


param = "crop_yield_upd_ref"

# Step 1: Gather data per model and year
model_year_data = {}

for year, year_dict in dict_param_orga.items():
    for model, res_model in year_dict.items():
        if model not in model_year_data:
            model_year_data[model] = {}
        model_year_data[model][year] = res_model

# Step 2: Define year pairs to compare
year_pairs = [ (2025,2050), (2050, 2070), (2070, 2090)]

# Step 3: Precompute all % differences and gather them
all_percent_changes = []

model_percent_diffs = {}

for model, years_data in model_year_data.items():
    model_percent_diffs[model] = {}

    for year_start, year_end in year_pairs:
        if year_start in years_data and year_end in years_data:
            res_model_start = years_data[year_start]
            res_model_end = years_data[year_end]
            
            percent_diff = {}

            # Loop through all locations
            for col_name, cell_start in res_model_start.items():
                if (isinstance(cell_start.iloc[0], pd.DataFrame) and
                    param in cell_start.iloc[0].columns and
                    col_name in res_model_end and
                    isinstance(res_model_end[col_name].iloc[0], pd.DataFrame) and
                    param in res_model_end[col_name].iloc[0].columns):
                    
                    start_values = cell_start.iloc[0][param].values
                    end_values = res_model_end[col_name].iloc[0][param].values

                    start_mean = np.nanmean(start_values)
                    end_mean = np.nanmean(end_values)

                    if np.isnan(start_mean) or start_mean == 0:
                        percent_change = np.nan
                    else:
                        percent_change = 100 * (end_mean - start_mean) / start_mean

                    percent_diff[col_name] = percent_change
                    all_percent_changes.append(percent_change)

            model_percent_diffs[model][(year_start, year_end)] = percent_diff



global_min = np.nanmin(all_percent_changes)
global_max = np.nanmax(all_percent_changes)

print(f"Global color scale range: {global_min:.1f}% to {global_max:.1f}%")




for model, yearpair_diffs in model_percent_diffs.items():
    for (year_start, year_end), percent_diff in yearpair_diffs.items():
        plot_variable_2d_percent_change(
            percent_diff, model, f"{year_start}-{year_end}", param, ".",
            global_min, global_max
        )
        
        


# common scale

param = "crop_yield_ratio_avs_ref"

# Step 1: Gather data per model and year
model_year_data = {}

for year, year_dict in dict_param_orga.items():
    for model, res_model in year_dict.items():
        if model not in model_year_data:
            model_year_data[model] = {}
        model_year_data[model][year] = res_model

# Step 2: Define year pairs to compare
year_pairs = [ (2025,2050), (2050, 2070), (2070, 2090)]

# Step 3: Precompute all % differences and gather them
all_percent_changes = []

model_percent_diffs = {}

for model, years_data in model_year_data.items():
    model_percent_diffs[model] = {}

    for year_start, year_end in year_pairs:
        if year_start in years_data and year_end in years_data:
            res_model_start = years_data[year_start]
            res_model_end = years_data[year_end]
            
            percent_diff = {}

            # Loop through all locations
            for col_name, cell_start in res_model_start.items():
                if (isinstance(cell_start.iloc[0], pd.DataFrame) and
                    param in cell_start.iloc[0].columns and
                    col_name in res_model_end and
                    isinstance(res_model_end[col_name].iloc[0], pd.DataFrame) and
                    param in res_model_end[col_name].iloc[0].columns):
                    
                    start_values = cell_start.iloc[0][param].values
                    end_values = res_model_end[col_name].iloc[0][param].values

                    start_mean = np.nanmean(start_values)
                    end_mean = np.nanmean(end_values)

                    if np.isnan(start_mean) or start_mean == 0:
                        percent_change = np.nan
                    else:
                        percent_change = 100 * (end_mean - start_mean) / start_mean

                    percent_diff[col_name] = percent_change
                    all_percent_changes.append(percent_change)

            model_percent_diffs[model][(year_start, year_end)] = percent_diff



print(f"Global color scale range: {global_min:.1f}% to {global_max:.1f}%")




for model, yearpair_diffs in model_percent_diffs.items():
    for (year_start, year_end), percent_diff in yearpair_diffs.items():
        plot_variable_2d_percent_change(
            percent_diff, model, f"{year_start}-{year_end}", param, ".",
            global_min, global_max
        )
        
        
        
