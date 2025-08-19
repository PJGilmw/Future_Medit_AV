# -*- coding: utf-8 -*-
"""
Created on Wed Apr 30 13:33:52 2025

@author: pjouannais


Plotting functions + Functions to process results before plotting.

"""



import numpy as np

import pandas as pd

from itertools import *

from util_functions import *

import re

import matplotlib.pyplot as plt

import numpy.ma as ma

 
import seaborn as sns

import os
 
from matplotlib.cm import ScalarMappable

import matplotlib.gridspec as gridspec

from matplotlib.colors import TwoSlopeNorm, Normalize

import glob

from collections import defaultdict


def organize_results_to_plot(resfor1year,
                             n_models,
                             size_list_config,
                             size_uncert,
                             gridyear,
                             list_meth):
    
    # Mode 1 Uncertainty = Simulation models + Parameters # DISAGGREGATED
    number_values_1_location  = n_models * size_list_config * size_uncert

    list_coord = list(gridyear.columns)


    grid_year_result = pd.DataFrame(index=[meth[2] for meth in list_meth], columns=list_coord,dtype=object)
    

    # meth_index=0
    for meth_index in range(len(list_meth)):
        
        
        resfor1year_for1meth = resfor1year[meth_index]
        
        name_index = list_meth[meth_index][2]
        
        list_res_per_grid_point = divide_per_gridpoint(resfor1year_for1meth,
                                 list_coord,
                                 number_values_1_location)
        
        for coord_index in range(len(list_coord)):
            
            grid_year_result.loc[name_index][list_coord[coord_index]]  = list_res_per_grid_point[coord_index]
            
            
    return grid_year_result





def organize_param_to_plot(paramfor1year,
                             n_models,
                             size_list_config,
                             size_uncert,
                             gridyear
                             ):
    
    # Mode 1 Uncertainty = Simulation models + Parameters # DISAGGREGATED
    number_values_1_location  = n_models * size_list_config * size_uncert

    list_coord = list(gridyear.columns)


    
    grid_year_param = pd.DataFrame(index=[0], columns=list_coord,dtype=object)
    

        
        
    
    
    list_res_per_grid_point = divide_per_gridpoint(paramfor1year,
                             list_coord,
                             number_values_1_location)
    
    for coord_index in range(len(list_coord)):
        
        grid_year_param.iloc[0][list_coord[coord_index]]  = list_res_per_grid_point[coord_index]
        
            
    return grid_year_param



def divide_per_gridpoint(resfor1year,
                         list_coord,
                         number_values_1_location):
# Calculate the number of rows per smaller DataFrame
    rows_per_df = len(resfor1year) // len(list_coord)
    
    # Split the DataFrame
    dfs = []
    start = 0
    for i in range(len(list_coord)):
        # Adjust the size of each chunk to distribute extra rows
        end = start + rows_per_df 
        dfs.append(resfor1year.iloc[start:end])
        start = end
        
    return dfs    
    

    

def remove_slash_from_title(title):
# Replace any slashes with an empty string
    return title.replace('/', '')





def extract_year(input_string):
    """
    Extracts the year value from the input string.
    The year is expected to appear after the substring "year".

    Parameters:
    input_string (str): The string to search for the year.

    Returns:
    int or None: The extracted year as an integer, or None if not found.
    """
    match = re.search(r"year(\d{4})", input_string)
    return int(match.group(1)) if match else None


def extract_after_arg(s):
    match = re.search(r'arg(.*)', s)
    return match.group(1) if match else None



def extract_between_model_and_year(s):
    match = re.search(r'model(.*?)year', s)
    return match.group(1) if match else None






def extract_prem_segment(s):
    match = re.search(r'prem_(.*?(?:long|short))', s)
    if match:
        return match.group(1)
    else:
        match_alt = re.search(r'prem_(.*)_', s)
        if match_alt:
            return match_alt.group(1)
    return None


def extract_combined(s):
    # print("Input string:", s)

    # Extract part between 'model' and 'year'
    model_idx = s.find("model")
    year_idx = s.find("year")

    if model_idx == -1 or year_idx == -1 or model_idx >= year_idx:
        print("Could not find valid 'model...year' section") 
        return None

    between_model_year = s[model_idx + len("model"):year_idx]

    # Now match after '_arg' (instead of '-arg')
    match_arg = re.search(r'arg([^_]+)', s)
    if not match_arg:
        print("Could not find '_arg' section")
        return None

    after_arg = match_arg.group(1)

    result = f"{between_model_year}{after_arg}"
    return result




def importpickle(path):
    """Imports a pickle object"""

    with open(path, 'rb') as pickle_load:
        obj = pickle.load(pickle_load)
    return obj  


    


    
def make_dif(df,var1,var2):
    
    df[var1+"-"+var2] =   df[var1] - df[var2]
    df[var1+"-"+var2+"/"+var2] =   (df[var1] - df[var2])/df[var2]
    
    return df




def make_dif_over_dict(dict_res,var1,var2,nlist):
    
    if nlist == 1:
        for year in list(dict_res.keys()):
            
            for model in dict_res[year]:
                
                list_res = dict_res[year][model][1]
                df_input_param = dict_res[year][model][0]
                
                for res_meth in list_res:
                    
                    res_meth = make_dif(res_meth,var1,var2)
                    
    elif nlist == 2:
        for year in list(dict_res.keys()):
     
            for model in dict_res[year]:
                
                list_res_config1 = dict_res[year][model][2]
                list_res_config2 = dict_res[year][model][3]
                df_input_param_config1 = dict_res[year][model][0]
                df_input_param_config2 = dict_res[year][model][1]
                
                for meth_index in range(len(list_res_config1)):
                    
                    list_res_config1[meth_index] = make_dif(list_res_config1[meth_index],var1,var2)
                    list_res_config2[meth_index] = make_dif(list_res_config2[meth_index],var1,var2)
                    
    return dict_res             
                
                
    






def add_impact_avs_elec_wheatglobalmarket_2_eco310_3(res,impact_wheat_market,key):
    
    keycollect = extract_prem_segment(key)
    print(key)
    print(keycollect)

    corresponding_dict_impact_wheat_market =    impact_wheat_market[[key for key in impact_wheat_market if keycollect in key][0]]
    # Project
    print(keycollect)
    # print(corresponding_dict_impact_wheat_market.keys())

    for key,listdf in res.items():
        resparam= listdf[0]
        res_result= listdf[1]
        
        if keycollect!="Ecoinvent_310_short":
            print("uu")
            corresponding_impact_wheatmarket = corresponding_dict_impact_wheat_market[key] # db,year
            
            print(corresponding_impact_wheatmarket)
            print(len(corresponding_impact_wheatmarket))
            
            for meth_index in range(len(res_result)):
                print(meth_index)
                impact_wheatglobal = corresponding_impact_wheatmarket[meth_index] # Impact category
                
                
                yield_cropavs = resparam["init_yield_wheat"]*resparam["correc_orchidee"]*resparam["crop_yield_upd_ref"]*resparam["crop_yield_ratio_avs_ref"]
               
                yield_elec= 10000 *resparam["surface_cover_fraction_AVS"]* resparam["panel_efficiency_PV"] * resparam["annual_producible_PV"] *resparam["annual_producible_ratio_avs_pv"]
    
                res_result[meth_index]["AVS_elec_main_wheat_market"] =res_result[meth_index]["AVS_elec_main_single"] + (res_result[meth_index]["wheat_fr_ref"]-impact_wheatglobal)*yield_cropavs/yield_elec
                res_result[meth_index]["wheat_market"] = impact_wheatglobal
        else:
            for meth_index in range(len(res_result)):
                 print('tttt')
                 print(corresponding_dict_impact_wheat_market)
                 impact_wheatglobal = corresponding_dict_impact_wheat_market[meth_index] # Impact category
                 
                 
                 yield_cropavs = resparam["init_yield_wheat"]*resparam["correc_orchidee"]*resparam["crop_yield_upd_ref"]*resparam["crop_yield_ratio_avs_ref"]
                
                 yield_elec= 10000 *resparam["surface_cover_fraction_AVS"]* resparam["panel_efficiency_PV"] * resparam["annual_producible_PV"] *resparam["annual_producible_ratio_avs_pv"]
     
                 res_result[meth_index]["AVS_elec_main_wheat_market"] =res_result[meth_index]["AVS_elec_main_single"] + (res_result[meth_index]["wheat_fr_ref"]-impact_wheatglobal)*yield_cropavs/yield_elec
                 
                 res_result[meth_index]["wheat_market"] = impact_wheatglobal

    return res





def add_impact_elecgrid(res,impact_elecgrid,key):
    
    keycollect = extract_prem_segment(key)
    print(key)

    corresponding_dict_impact_elec_market =    impact_elecgrid[[key for key in impact_elecgrid if keycollect in key][0]]
    # Project

    for key,listdf in res.items():
        resparam= listdf[0]
        res_result= listdf[1]
        
        if keycollect!="Ecoinvent_310_short":
            corresponding_impact_elecmarket = corresponding_dict_impact_elec_market[key] # db,year
            
            print(corresponding_impact_elecmarket)
            print(len(corresponding_impact_elecmarket))
            
            for meth_index in range(len(res_result)):
                print(meth_index)
                
                impact_elecglobal = corresponding_impact_elecmarket[meth_index] # Impact category
                

                res_result[meth_index]["Elec_margi"] =impact_elecglobal
         
        else:
            
            print(corresponding_dict_impact_elec_market)
            


            
            for meth_index in range(len(res_result)):
                print(meth_index)
                
                impact_elecglobal = corresponding_dict_impact_elec_market[meth_index] # Impact category
                

                res_result[meth_index]["Elec_margi"] =impact_elecglobal
         
    return res





def apply_organize_results_to_all_multiconfig(dict_res, nmodels, size_list_config, size_uncert, list_meth,divideconfig):
    """
    Applies `organize_results_to_plot` to every DataFrame in the nested dictionary `dict_res`.

    Parameters:
    - dict_res: Dictionary of dictionaries containing DataFrames.
    - nmodels: Number of models (passed to `organize_results_to_plot`).
    - size_list_config: Size of list configuration (passed to `organize_results_to_plot`).
    - size_uncert: Size of uncertainty (passed to `organize_results_to_plot`).
    - grid_year_data: The grid year DataFrame (passed to `organize_results_to_plot`).
    - list_meth: The list of methods (passed to `organize_results_to_plot`).

    Returns:
    - new_dict_res: A new dictionary with the same structure, containing the processed DataFrames.
    """
    new_dict_res = {}

    for year, dict_res_year in dict_res.items():
        new_dict_res_year = {}

        for model_name, list_df in dict_res_year.items():
            
            
            if not divideconfig:
                print(len(list_df))
                # Apply organize_results_to_plot to each DataFrame
                new_dict_res_year[model_name] = organize_results_to_plot(
                    list_df[1], 
                    nmodels, 
                    size_list_config, 
                    size_uncert, 
                    list_df[2],  # Pass the corresponding grid year data for the year
                    list_meth
                )
            
                new_dict_res[year] = new_dict_res_year
            
            else:
                    
                # Apply organize_results_to_plot to each DataFrame
                new_dict_res_year[model_name+"config1"] = organize_results_to_plot(
                    list_df[2], 
                    nmodels, 
                    1, 
                    size_uncert, 
                    list_df[4],  # Pass the corresponding grid year data for the year
                    list_meth
                )
            
                new_dict_res_year[model_name+"config2"] = organize_results_to_plot(
                    list_df[3], 
                    nmodels, 
                    1, 
                    size_uncert, 
                    list_df[4],  # Pass the corresponding grid year data for the year
                    list_meth
                )
            
            new_dict_res[year] = new_dict_res_year
                
                

    return new_dict_res






def apply_organize_results_to_all_param_multiconfig(dict_res, nmodels, size_list_config, size_uncert,divideconfig):
    """
    Applies `organize_results_to_plot` to every DataFrame in the nested dictionary `dict_res`.

    Parameters:
    - dict_res: Dictionary of dictionaries containing DataFrames.
    - nmodels: Number of models (passed to `organize_results_to_plot`).
    - size_list_config: Size of list configuration (passed to `organize_results_to_plot`).
    - size_uncert: Size of uncertainty (passed to `organize_results_to_plot`).
    - grid_year_data: The grid year DataFrame (passed to `organize_results_to_plot`).
    - list_meth: The list of methods (passed to `organize_results_to_plot`).

    Returns:
    - new_dict_res: A new dictionary with the same structure, containing the processed DataFrames.
    """
    new_dict_res = {}


      
    for year, dict_res_year in dict_res.items():
        new_dict_res_year = {}

        for model_name, list_df in dict_res_year.items():
            
            
            if not divideconfig:
                print(len(list_df))
                # Apply organize_results_to_plot to each DataFrame
                new_dict_res_year[model_name] = organize_param_to_plot(
                      list_df[0], 
                      nmodels, 
                      size_list_config, 
                      size_uncert, 
                      list_df[2]  # Pass the corresponding grid year data for the year
                      
                  )
              
              
         
              
                new_dict_res[year] = new_dict_res_year
              
            
            else:
                    
                # Apply organize_results_to_plot to each DataFrame
                new_dict_res_year[model_name+"config1"] = organize_param_to_plot(
                    list_df[0], 
                    nmodels, 
                    1, 
                    size_uncert, 
                    list_df[4]  # Pass the corresponding grid year data for the year
                )
            
                new_dict_res_year[model_name+"config2"] = organize_param_to_plot(
                    list_df[1], 
                    nmodels, 
                    1, 
                    size_uncert, 
                    list_df[4]  # Pass the corresponding grid year data for the year
                    
                )
            
            new_dict_res[year] = new_dict_res_year    

    return new_dict_res




def select_categ_and_transpose(dict_res, meth):
    """
    Applies `organize_results_to_plot` to every DataFrame in the nested dictionary `dict_res`.

    Parameters:
    - dict_res: Dictionary of dictionaries containing DataFrames.
    - nmodels: Number of models (passed to `organize_results_to_plot`).
    - size_list_config: Size of list configuration (passed to `organize_results_to_plot`).
    - size_uncert: Size of uncertainty (passed to `organize_results_to_plot`).
    - grid_year_data: The grid year DataFrame (passed to `organize_results_to_plot`).
    - list_meth: The list of methods (passed to `organize_results_to_plot`).

    Returns:
    - new_dict_res: A new dictionary with the same structure, containing the processed DataFrames.
    """
    new_dict_res = {}

    for year, dict_res_year in dict_res.items():
        new_dict_res_year = {}

        for model_name, df in dict_res_year.items():
            # Apply organize_results_to_plot to each DataFrame
            new_dict_res_year[model_name] = pd.DataFrame(df.loc[meth]).T
        
        new_dict_res[year] = new_dict_res_year

    return new_dict_res









def plot_variable_2d_mean_white_samescale(grid_year_result_3, model, year, variable_name, legend, vmin, vmax):
    lon_coords = []
    lat_coords = []
    data_means = []

    # Iterate through columns (grid points)
    for col_name, cell in grid_year_result_3.items():
        if isinstance(cell.iloc[0], pd.DataFrame) and variable_name in cell.iloc[0].columns:
            lat, lon = col_name  # Extract coordinates
            values = cell.iloc[0][variable_name].values
            lon_coords.append(lon)
            lat_coords.append(lat)
            data_means.append(np.mean(values))

    # Convert to numpy arrays
    lon_coords = np.array(lon_coords)
    lat_coords = np.array(lat_coords)
    data_means = np.array(data_means)

    # Reshape for plotting
    lon_unique = np.unique(lon_coords)
    lat_unique = np.unique(lat_coords)
    lon_mesh, lat_mesh = np.meshgrid(lon_unique, lat_unique)

    # Map extracted mean data to the grid
    mean_surface = np.full(lon_mesh.shape, np.nan)  
    for lon, lat, mean in zip(lon_coords, lat_coords, data_means):
        i = np.where(lat_unique == lat)[0][0]
        j = np.where(lon_unique == lon)[0][0]
        mean_surface[i, j] = mean

    # Mask NaN values explicitly
    mean_surface_masked = ma.masked_invalid(mean_surface)

    # Create 2D plot
    fig, ax = plt.subplots(figsize=(12, 8))

    # Use the provided vmin and vmax for consistent color scaling
    cmap = plt.get_cmap('viridis')
    cmap.set_bad(color='white')  
    norm = plt.Normalize(vmin=vmin, vmax=vmax)

    # print("Longitude values:", lon_coords)
    # print("Latitude values:", lat_coords)
    # print("Data means:", data_means)
    if len(lon_coords) == 0 or len(lat_coords) == 0 or len(data_means) == 0:
        

        raise ValueError("No valid data points to plot for 'maize_ch_ref'.")

    # Plot with common color scale
    c = ax.pcolormesh(lon_mesh, lat_mesh, mean_surface_masked, cmap=cmap, shading='auto', norm=norm)

    # Add color bar
    cbar = fig.colorbar(c, ax=ax,orientation="horizontal")
    cbar.set_label(f'{legend}')

    # Set title
    ax.set_title(f' {model} {year} {variable_name} (Mean)')
    ax.set_xticks([])
    ax.set_yticks([])
    # Save the figure as JPEG


    name_clean = remove_slash_from_title(variable_name)
    filename = f"../Results/{model}_{year}_{name_clean}.jpg"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved plot: {filename}")

    plt.show()








def divide_by_config(res, nuncert,nlist):
    
    
    for key,list_df in res.items():
        
        dfparam = list_df[0]
        list_df_res = list_df[1]
        
        indexes_list1 = [[u for u in  range(i,i+nuncert)] for i in range(0,dfparam.shape[0],nuncert*nlist)] 
        indexes_list1 = [item for sublist in indexes_list1 for item in sublist]

        indexes_set1 = set(indexes_list1)
        indexes_list2 = [u for u in range(dfparam.shape[0]) if u not in indexes_set1]
        
        res_param_list1 = dfparam.loc[indexes_list1].reset_index(drop=True)
        res_param_list2 = dfparam.loc[indexes_list2].reset_index(drop=True)
        
        list_res_meth_config1 = []
        list_res_meth_config2 = []

        for meth_index in range(len(list_df_res)):
            
            res_list1 = list_df_res[meth_index].loc[indexes_list1].reset_index(drop=True)
            res_list2 = list_df_res[meth_index].loc[indexes_list2].reset_index(drop=True)
            
            list_res_meth_config1.append(res_list1)
            list_res_meth_config2.append(res_list2)
            
        res[key]=[res_param_list1,res_param_list2,list_res_meth_config1,list_res_meth_config2,list_df[2]]

        
    return res








def reorganize_by_years(list_models,
                        list_years):
    
            
    dict_res= {}
    for year in list_years:
        
        dict_res_year ={}
        for model in list_models:
            # print(model)
            
            aa= [a for a in model.keys() if str(year) in a]
            if len(aa)!=0:
                
                for name in aa:
                    
                        
                    print(name)
                # To adapt with name long area name_long_area
                    extracted_name = extract_combined(name)
                
                
                    dict_res_year[extracted_name] = model[name]
                
        dict_res[year]=dict_res_year
        
    return dict_res




def find_string_with_year(strings, target_year):
    for s in strings:
        match = re.search(r'year(.*?)_arg', s)
        if match and match.group(1) == str(target_year):
            return s
    return None    
    
    
def ref_pv_fixed(st_impact,ratio_st_fixed):
    
    return st_impact*ratio_st_fixed
    
    
    
    
def plot_variable_panel_for_selected_models_2(
    filename,
    dict_res_orga_gw_T,
    selected_models,
    variable_list,
    unit_label,
    colormap="coolwarm",
    center_zero=True
):

    os.makedirs("../Results/Panel_Plots", exist_ok=True)

    mean_values_cells = []
    stats_records = []

    # Step 1: Compute global min and max across selected models and years
    for year_dict in dict_res_orga_gw_T.values():
        for model_name, res_model in year_dict.items():
            if model_name in selected_models:
                for col_name, cell in res_model.items():
                    if isinstance(cell.iloc[0], pd.DataFrame):
                        for var in variable_list:
                            if var in cell.iloc[0].columns:
                                values = cell.iloc[0][var].values
                                mean_values_cells.append(np.mean(values))

    global_min = np.min(mean_values_cells) 
    global_max = np.max(mean_values_cells) 
    print(global_min)
    print(global_max)

    # Choose norm: centered or regular
    if center_zero:
        abs_max = max(abs(global_min), abs(global_max))
        norm = TwoSlopeNorm(vmin=-abs_max, vcenter=0, vmax=abs_max)
    else:
        norm = Normalize(vmin=global_min, vmax=global_max)

    # Step 2: Count rows and row labels
    row_labels = []
    for model in selected_models:
        for year in dict_res_orga_gw_T:
            if model in dict_res_orga_gw_T[year]:
                row_labels.append((model, year))

    rows = len(row_labels)
    cols = len(variable_list)

    # Create figure with space for colorbar
    fig = plt.figure(figsize=(5 * cols + 1, 4 * rows))
    gs = gridspec.GridSpec(rows, cols + 1, width_ratios=[1] * cols + [0.05], wspace=0.1)

    cmap = plt.get_cmap(colormap)
    cmap.set_bad(color='white')

    last_im = None  # to capture last image for colorbar

    for row_idx, (model, year) in enumerate(row_labels):
        res_model = dict_res_orga_gw_T[year][model]

        for col_idx, variable_name in enumerate(variable_list):
            ax = fig.add_subplot(gs[row_idx, col_idx])
            lon_coords, lat_coords, data_means = [], [], []

            for col_name, cell in res_model.items():
                if isinstance(cell.iloc[0], pd.DataFrame) and variable_name in cell.iloc[0].columns:
                    lat, lon = col_name
                    values = cell.iloc[0][variable_name].values
                    lon_coords.append(lon)
                    lat_coords.append(lat)
                    data_means.append(np.mean(values))

            if len(data_means) == 0:
                ax.set_title(f"No data for {variable_name}")
                ax.axis('off')
                continue

            # Save stats for this model-year-variable
            clean_vals = np.array(data_means)[~np.isnan(data_means)]
            stats_records.append({
                "Model": model,
                "Year": year,
                "Variable": variable_name,
                "Min": np.min(clean_vals),
                "Max": np.max(clean_vals),
                "Mean": np.mean(clean_vals),
                "Median": np.median(clean_vals)
            })

            lon_coords = np.array(lon_coords)
            lat_coords = np.array(lat_coords)
            data_means = np.array(data_means)

            lon_unique = np.unique(lon_coords)
            lat_unique = np.unique(lat_coords)
            lon_mesh, lat_mesh = np.meshgrid(lon_unique, lat_unique)

            mean_surface = np.full(lon_mesh.shape, np.nan)
            for lon, lat, mean in zip(lon_coords, lat_coords, data_means):
                i = np.where(lat_unique == lat)[0][0]
                j = np.where(lon_unique == lon)[0][0]
                mean_surface[i, j] = mean

            mean_surface_masked = ma.masked_invalid(mean_surface)

            last_im = ax.pcolormesh(
                lon_mesh, lat_mesh, mean_surface_masked,
                cmap=cmap, shading='auto', norm=norm
            )
            ax.set_xticks([])
            ax.set_yticks([])

            if row_idx == 0:
                ax.set_title(variable_name.replace('/', '\n'), fontsize=10)
            if col_idx == 0:
                ax.set_ylabel(f"{model}\n{year}", fontsize=10)

    # Add colorbar to last column
    cbar_ax = fig.add_subplot(gs[:, -1])
    cbar = fig.colorbar(last_im, cax=cbar_ax)
    cbar.set_label(unit_label)

    fig.suptitle("Panel of Selected Models and Years", fontsize=16)
    plt.tight_layout(rect=[0, 0, 0.95, 0.96])

    # Save figure
    fig_path = f"../Results/Panel_Plots/{filename}.jpg"
    plt.savefig(fig_path, dpi=300)
    print(f"Saved panel: {fig_path}")
    plt.close()

    # Save statistics to CSV
    stats_df = pd.DataFrame(stats_records)
    stats_path = f"../Results/Panel_Plots/Stats_{filename}.csv"
    stats_df.to_csv(stats_path, index=False)
    print(f"Saved stats: {stats_path}")





def combine_stats_csvs_by_substring(substring, output_filename=None, folder="../Results/Panel_Plots"):
    """
    Combine all Stats_*.csv files in the specified folder that contain the given substring in their filename.

    Parameters:
    - substring (str): Substring to match in filenames.
    - output_filename (str or None): Name of the combined output CSV file. If None, it will be auto-generated.
    - folder (str): Path to the folder containing Stats_*.csv files.

    Returns:
    - None: Saves the combined DataFrame as a CSV in the same folder.
    """
    pattern = os.path.join(folder, f"Stats_*{substring}*.csv")
    matching_files = glob.glob(pattern)

    if not matching_files:
        print(f"No CSV files found containing '{substring}' in the filename.")
        return

    # Load and concatenate all matching files
    combined_df = pd.concat(
        [pd.read_csv(file).assign(SourceFile=os.path.basename(file)) for file in matching_files],
        ignore_index=True
    )
    combined_df = combined_df.round(4)


    # Determine output filename
    if output_filename is None:
        output_filename = f"Combined_Stats_{substring}.csv"

    output_path = os.path.join(folder, output_filename)
    combined_df.to_csv(output_path, index=False)
    print(f"Combined {len(matching_files)} files into: {output_path}")
    
    
def plot_variable_panel_grouped_by_column(
    filename,
    dict_res_orga_gw_T,
    selected_models,
    variable_name,
    unit_label,
    colormap="coolwarm",
    center_zero=True,
    dict_plot_names=None


):
    assert len(selected_models) == 6, "This layout assumes exactly 6 selected models."
    
    # Step 1: Compute global min and max across selected models and years
    mean_values_cells = []

    for year_dict in dict_res_orga_gw_T.values():
        for model_name, res_model in year_dict.items():
            if model_name in selected_models:
                for (lat, lon), cell in res_model.items():
                    if isinstance(cell.iloc[0], pd.DataFrame) and variable_name in cell.iloc[0].columns:
                        mean_values_cells.append(np.nanmean(cell.iloc[0][variable_name].values))

    # global_min = np.nanmin(mean_values_cells)
    # global_max = np.nanmax(mean_values_cells)
    
    global_min = np.nanpercentile(mean_values_cells, 2) # more contrast
    global_max = np.nanpercentile(mean_values_cells, 98)
    print("Global min:", global_min)
    print("Global max:", global_max)

    # Choose norm: centered or regular
    if center_zero:
        abs_max = max(abs(global_min), abs(global_max))
        norm = TwoSlopeNorm(vmin=-abs_max, vcenter=0, vmax=abs_max)
    else:
        norm = Normalize(vmin=global_min, vmax=global_max)

    # Organize models into 3 columns, 2 models each (each with 3 years)
    grouped_models = [selected_models[i:i+2] for i in range(0, 6, 2)]
    years = [2050,2070,2090] 
    assert len(years) == 3, "This layout assumes exactly 3 years."

    rows = 6  # 2 models * 3 years
    cols = 3  # 3 columns of 2-models
    fig = plt.figure(figsize=(5 * cols, 3 * rows))
    gs = gridspec.GridSpec(rows, cols, hspace=0.4, wspace=0.15)

    cmap = plt.get_cmap(colormap)
    cmap.set_bad(color='white')

    last_im = None

    for col_idx, model_pair in enumerate(grouped_models):
        for model_pos, model in enumerate(model_pair):
            for year_idx, year in enumerate(years):
                row_idx = model_pos * 3 + year_idx
                ax = fig.add_subplot(gs[row_idx, col_idx])

                try:
                    res_model = dict_res_orga_gw_T[year][model]
                except KeyError:
                    ax.set_title("No data")
                    ax.axis('off')
                    continue

                lon_coords, lat_coords, data_means = [], [], []

                for (lat, lon), cell in res_model.items():
                    if isinstance(cell.iloc[0], pd.DataFrame) and variable_name in cell.iloc[0].columns:
                        values = cell.iloc[0][variable_name].values
                        lon_coords.append(lon)
                        lat_coords.append(lat)
                        data_means.append(np.mean(values))

                if len(data_means) == 0:
                    ax.set_title(f"No data for {variable_name}")
                    ax.axis('off')
                    continue

                lon_coords = np.array(lon_coords)
                lat_coords = np.array(lat_coords)
                data_means = np.array(data_means)

                lon_unique = np.unique(lon_coords)
                lat_unique = np.unique(lat_coords)
                lon_mesh, lat_mesh = np.meshgrid(lon_unique, lat_unique)

                mean_surface = np.full(lon_mesh.shape, np.nan)
                for lon, lat, mean in zip(lon_coords, lat_coords, data_means):
                    i = np.where(lat_unique == lat)[0][0]
                    j = np.where(lon_unique == lon)[0][0]
                    mean_surface[i, j] = mean

                mean_surface_masked = ma.masked_invalid(mean_surface)

                last_im = ax.pcolormesh(
                    lon_mesh, lat_mesh, mean_surface_masked,
                    cmap=cmap, shading='auto', norm=norm
                )

                ax.set_xticks([])
                ax.set_yticks([])
                # ax.set_title(f"{model} — {year}", fontsize=8)
                mean_val = np.nanmean(data_means)
                median_val = np.nanmedian(data_means)
                if dict_plot_names:
                    subtitle_name = dict_plot_names[model]
                ax.set_title(
                    f"{subtitle_name} — {year}\nMean: {mean_val:.2f}, Median: {median_val:.2f}",
                    fontsize=8
                )

    # Horizontal colorbar below all subplots
    cbar_ax = fig.add_axes([0.2, 0.05, 0.6, 0.02])  # [left, bottom, width, height]
    cbar = fig.colorbar(last_im, cax=cbar_ax, orientation='horizontal')
    cbar.set_label(unit_label)

    fig.suptitle(f"{variable_name}", fontsize=16, y=0.98)

    os.makedirs("../Results/Panel_Plots", exist_ok=True)
    filename = f"../Results/Panel_Plots/{filename}.jpg"
    plt.savefig(filename, dpi=800)
    print(f"Saved panel: {filename}")
    plt.close()



def plot_variable_panel_grouped_by_column_samescale(norm,

    filename,
    dict_res_orga_gw_T,
    selected_models,
    variable_name,
    unit_label,
    colormap="coolwarm",
    center_zero=True,
    dict_plot_names=None


):
    assert len(selected_models) == 6, "This layout assumes exactly 6 selected models."
    

    # Organize models into 3 columns, 2 models each (each with 3 years)
    grouped_models = [selected_models[i:i+2] for i in range(0, 6, 2)]
    years = [2050,2070,2090] 
    assert len(years) == 3, "This layout assumes exactly 3 years."

    rows = 6  # 2 models * 3 years
    cols = 3  # 3 columns of 2-models
    fig = plt.figure(figsize=(5 * cols, 3 * rows))
    gs = gridspec.GridSpec(rows, cols, hspace=0.4, wspace=0.15)

    cmap = plt.get_cmap(colormap)
    cmap.set_bad(color='white')

    last_im = None

    for col_idx, model_pair in enumerate(grouped_models):
        for model_pos, model in enumerate(model_pair):
            for year_idx, year in enumerate(years):
                row_idx = model_pos * 3 + year_idx
                ax = fig.add_subplot(gs[row_idx, col_idx])

                try:
                    res_model = dict_res_orga_gw_T[year][model]
                except KeyError:
                    ax.set_title("No data")
                    ax.axis('off')
                    continue

                lon_coords, lat_coords, data_means = [], [], []

                for (lat, lon), cell in res_model.items():
                    if isinstance(cell.iloc[0], pd.DataFrame) and variable_name in cell.iloc[0].columns:
                        values = cell.iloc[0][variable_name].values
                        lon_coords.append(lon)
                        lat_coords.append(lat)
                        data_means.append(np.mean(values))

                if len(data_means) == 0:
                    ax.set_title(f"No data for {variable_name}")
                    ax.axis('off')
                    continue

                lon_coords = np.array(lon_coords)
                lat_coords = np.array(lat_coords)
                data_means = np.array(data_means)

                lon_unique = np.unique(lon_coords)
                lat_unique = np.unique(lat_coords)
                lon_mesh, lat_mesh = np.meshgrid(lon_unique, lat_unique)

                mean_surface = np.full(lon_mesh.shape, np.nan)
                for lon, lat, mean in zip(lon_coords, lat_coords, data_means):
                    i = np.where(lat_unique == lat)[0][0]
                    j = np.where(lon_unique == lon)[0][0]
                    mean_surface[i, j] = mean

                mean_surface_masked = ma.masked_invalid(mean_surface)

                last_im = ax.pcolormesh(
                    lon_mesh, lat_mesh, mean_surface_masked,
                    cmap=cmap, shading='auto', norm=norm
                )

                ax.set_xticks([])
                ax.set_yticks([])
                # ax.set_title(f"{model} — {year}", fontsize=8)
                mean_val = np.nanmean(data_means)
                median_val = np.nanmedian(data_means)
                if dict_plot_names:
                    subtitle_name = dict_plot_names[model]
                ax.set_title(
                    f"{subtitle_name} — {year}\nMean: {mean_val:.2f}, Median: {median_val:.2f}",
                    fontsize=8
                )

    # Horizontal colorbar below all subplots
    cbar_ax = fig.add_axes([0.2, 0.05, 0.6, 0.02])  # [left, bottom, width, height]
    cbar = fig.colorbar(last_im, cax=cbar_ax, orientation='horizontal')
    cbar.set_label(unit_label)

    fig.suptitle(f"{variable_name}", fontsize=16, y=0.98)

    os.makedirs("../Results/Panel_Plots", exist_ok=True)
    filename = f"../Results/Panel_Plots/{filename}.jpg"
    plt.savefig(filename, dpi=800)
    print(f"Saved panel: {filename}")
    plt.close()





def plot_variable_2d_percent_change(percent_diff, model, year_range, param, save_folder, global_min, global_max):
    # Prepare the data
    lon_coords, lat_coords, values = [], [], []
    for (lat, lon), change in percent_diff.items():
        lon_coords.append(lon)
        lat_coords.append(lat)
        values.append(change)
    # Choose norm: centered or regular
    abs_max = max(abs(global_min), abs(global_max))
    
    print(abs_max)
    norm = TwoSlopeNorm(vmin=-abs_max, vcenter=0, vmax=abs_max)

        
    lon_coords = np.array(lon_coords)
    lat_coords = np.array(lat_coords)
    values = np.array(values)

    lon_unique = np.unique(lon_coords)
    lat_unique = np.unique(lat_coords)
    lon_mesh, lat_mesh = np.meshgrid(lon_unique, lat_unique)

    diff_surface = np.full(lon_mesh.shape, np.nan)
    for lon, lat, change in zip(lon_coords, lat_coords, values):
        i = np.where(lat_unique == lat)[0][0]
        j = np.where(lon_unique == lon)[0][0]
        diff_surface[i, j] = change

    fig, ax = plt.subplots(figsize=(8,6))
    c = ax.pcolormesh(
        lon_mesh, lat_mesh, diff_surface,
        cmap="bwr", shading='auto', norm=norm
    )
    

    plt.colorbar(c, label="Percentage Change [%]",orientation="horizontal")
    ax.set_title(f"{model} — {year_range}\n{param}")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_xticks([])
    ax.set_yticks([])
    
    name_clean = remove_slash_from_title(param)

    os.makedirs(os.path.join(save_folder, "../Results/Percent_Change_Plots"), exist_ok=True)
    save_path = os.path.join(save_folder, "../Results/Percent_Change_Plots", f"{model}_{name_clean}_{year_range}.jpg")
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"Saved {save_path}")





def plot_variable_panel_percent_change_grouped_by_column(
    filename,
    dict_res_orga_gw_T,
    selected_models,
    variable_name,
    unit_label,
    colormap="coolwarm",
    center_zero=True
):
    assert len(selected_models) == 6, "This layout assumes exactly 6 selected models."

    # Define year pairs for relative change
    year_pairs = [(2025, 2050), (2050, 2070), (2070, 2090)]

    # Step 1: Compute global min and max for percentage changes
    percent_changes_all = []
    for model in selected_models:
        for year1, year2 in year_pairs:
            try:
                res_model1 = dict_res_orga_gw_T[year1][model]
                res_model2 = dict_res_orga_gw_T[year2][model]
            except KeyError:
                continue

            for coord in res_model1:
                cell1 = res_model1[coord]
                cell2 = res_model2.get(coord, None)

                if not (isinstance(cell1.iloc[0], pd.DataFrame) and isinstance(cell2.iloc[0], pd.DataFrame)):
                    continue

                if variable_name in cell1.iloc[0].columns and variable_name in cell2.iloc[0].columns:
                    val1 = np.nanmean(cell1.iloc[0][variable_name].values)
                    val2 = np.nanmean(cell2.iloc[0][variable_name].values)
                    if val1 != 0 and not np.isnan(val1) and not np.isnan(val2):
                        percent_changes_all.append(100 * (val2 - val1) / val1)

    global_min = np.nanmin(percent_changes_all)
    global_max = np.nanmax(percent_changes_all)
    global_min = -75
    global_max = 75
    print("Global % change min:", global_min)
    print("Global % change max:", global_max)

    if center_zero:
        abs_max = max(abs(global_min), abs(global_max))
        norm = TwoSlopeNorm(vmin=-abs_max, vcenter=0, vmax=abs_max)
    else:
        norm = Normalize(vmin=global_min, vmax=global_max)

    grouped_models = [selected_models[i:i+2] for i in range(0, 6, 2)]
    rows = 6  # 2 models * 3 pairs
    cols = 3  # 3 columns of 2-models
    fig = plt.figure(figsize=(5 * cols, 3 * rows))
    gs = gridspec.GridSpec(rows, cols, hspace=0.4, wspace=0.15)

    cmap = plt.get_cmap(colormap)
    cmap.set_bad(color='white')
    last_im = None

    for col_idx, model_pair in enumerate(grouped_models):
        for model_pos, model in enumerate(model_pair):
            for pair_idx, (year1, year2) in enumerate(year_pairs):
                row_idx = model_pos * 3 + pair_idx
                ax = fig.add_subplot(gs[row_idx, col_idx])

                try:
                    res_model1 = dict_res_orga_gw_T[year1][model]
                    res_model2 = dict_res_orga_gw_T[year2][model]
                except KeyError:
                    ax.set_title("No data")
                    ax.axis('off')
                    continue

                lon_coords, lat_coords, pct_changes = [], [], []

                for coord in res_model1:
                    cell1 = res_model1[coord]
                    cell2 = res_model2.get(coord, None)

                    if not (isinstance(cell1.iloc[0], pd.DataFrame) and isinstance(cell2.iloc[0], pd.DataFrame)):
                        continue

                    if variable_name in cell1.iloc[0].columns and variable_name in cell2.iloc[0].columns:
                        val1 = np.nanmean(cell1.iloc[0][variable_name].values)
                        val2 = np.nanmean(cell2.iloc[0][variable_name].values)

                        if val1 != 0 and not np.isnan(val1) and not np.isnan(val2):
                            lat, lon = coord
                            lon_coords.append(lon)
                            lat_coords.append(lat)
                            pct_changes.append(100 * (val2 - val1) / val1)

                if len(pct_changes) == 0:
                    ax.set_title(f"No data for {variable_name}")
                    ax.axis('off')
                    continue

                lon_coords = np.array(lon_coords)
                lat_coords = np.array(lat_coords)
                pct_changes = np.array(pct_changes)

                lon_unique = np.unique(lon_coords)
                lat_unique = np.unique(lat_coords)
                lon_mesh, lat_mesh = np.meshgrid(lon_unique, lat_unique)

                surface = np.full(lon_mesh.shape, np.nan)
                for lon, lat, change in zip(lon_coords, lat_coords, pct_changes):
                    i = np.where(lat_unique == lat)[0][0]
                    j = np.where(lon_unique == lon)[0][0]
                    surface[i, j] = change

                surface_masked = ma.masked_invalid(surface)

                last_im = ax.pcolormesh(
                    lon_mesh, lat_mesh, surface_masked,
                    cmap=cmap, shading='auto', norm=norm
                )

                ax.set_xticks([])
                ax.set_yticks([])
                # ax.set_title(f"{model}: {year1} to {year2}", fontsize=8)
                mean_value = np.nanmean(surface_masked)
                ax.set_title(f"{model}\n{year1}–{year2}\nMean: {mean_value:.1f}%", fontsize=8)


    cbar_ax = fig.add_axes([0.2, 0.05, 0.6, 0.02])
    cbar = fig.colorbar(last_im, cax=cbar_ax, orientation='horizontal')
    cbar.set_label("% Change")
    name_clean = variable_name.replace("/", "_")  # Ensure no slashes in filenames

    fig.suptitle(f"Relative Change of {variable_name}", fontsize=16, y=0.98)
    os.makedirs("../Results/Panel_Plots", exist_ok=True)
    filename = f"../Results/Panel_Plots/{filename}_{name_clean}.jpg"
    plt.savefig(filename, dpi=600)
    print(f"Saved panel: {filename}")
    plt.close()









def remove_slash_from_title(s):
    return s.replace('/', '_').replace('\\', '_')


def plot_row_combined_surface_and_percentage(
    res_model,
    model,
    year,
    variable_name,
    vmin,
    vmax,
    rows_per_model,
    angle=(60, 225),
    cmap_3d='viridis',
    cmap_perc='YlGnBu',
    center_zero=True):

    fig = plt.figure(figsize=(14, 5))  # 1 row, 2 plots

    # --- Subplot 1: 3D Surface with Q1–Q3 volume ---
    ax3d = fig.add_subplot(1, 2, 1, projection='3d')
    ax3d.set_facecolor("white")
    ax3d.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))  # X pane
    ax3d.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))  # Y pane
    ax3d.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))  # Z pane
    lon_coords, lat_coords, data_q1, data_q3 = [], [], [], []

    for (lat, lon), cell in res_model.items():
        if isinstance(cell.iloc[0], pd.DataFrame) and variable_name in cell.iloc[0].columns:
            values = cell.iloc[0][variable_name].values
            lon_coords.append(lon)
            lat_coords.append(lat)
            data_q1.append(np.percentile(values, 25))
            data_q3.append(np.percentile(values, 75))

    if not data_q1 or not data_q3:
        print(f"No data for {variable_name} - {model} - {year}")
        return

    lon_coords = np.array(lon_coords)
    lat_coords = np.array(lat_coords)
    data_q1 = np.array(data_q1)
    data_q3 = np.array(data_q3)

    lon_unique = np.unique(lon_coords)
    lat_unique = np.unique(lat_coords)
    lon_mesh, lat_mesh = np.meshgrid(lon_unique, lat_unique)

    q1_surface = np.full_like(lon_mesh, np.nan, dtype=np.float64)
    q3_surface = np.full_like(lon_mesh, np.nan, dtype=np.float64)

    for lon, lat, q1, q3 in zip(lon_coords, lat_coords, data_q1, data_q3):
        i = np.where(lat_unique == lat)[0][0]
        j = np.where(lon_unique == lon)[0][0]
        q1_surface[i, j] = q1
        q3_surface[i, j] = q3

    # Set color normalization (you already have this):
    if center_zero:
        abs_max = max(abs(vmin), abs(vmax))
        norm = TwoSlopeNorm(vmin=-abs_max, vcenter=0, vmax=abs_max)
    else:
        norm = Normalize(vmin=vmin, vmax=vmax)
    # Plot the surfaces
    ax3d.plot_surface(lon_mesh, lat_mesh, q1_surface, cmap=cmap_3d, edgecolor='none', alpha=0.7, norm=norm)
    ax3d.plot_surface(lon_mesh, lat_mesh, q3_surface, cmap=cmap_3d, edgecolor='none', alpha=0.7, norm=norm)
    
    # Add connecting lines
    for i in range(lon_mesh.shape[0]):
        for j in range(lon_mesh.shape[1]):
            if not np.isnan(q1_surface[i, j]) and not np.isnan(q3_surface[i, j]):
                ax3d.plot([lon_mesh[i, j], lon_mesh[i, j]],
                          [lat_mesh[i, j], lat_mesh[i, j]],
                          [q1_surface[i, j], q3_surface[i, j]],
                          color='grey', alpha=0.3)
    
    sm = ScalarMappable(cmap=cmap_3d, norm=norm)
    sm.set_array([])  # Required for ScalarMappable
    
    # Add colorbar to the left of the 3D plot
    cbar3d = fig.colorbar(sm, ax=ax3d, shrink=0.6, pad=0.1, orientation='horizontal') 
    # cbar3d = fig.colorbar(sm, ax=ax3d, shrink=0.6, pad=0.1, location='left') 

    cbar3d.set_label(f'{variable_name} (Q1–Q3)', rotation=90)
    # ✅ Set z-axis limits consistently across plots
    # ax3d.set_zlim(vmin, vmax)
    ax3d.set_zlim(vmin*1.4, vmax)
    # ax3d.set_zlim(vmin*1.8, vmax) for crop main
    


    ax3d.view_init(elev=angle[0], azim=angle[1])
    # ax3d.set_xlabel('Longitude')
    # ax3d.set_ylabel('Latitude')
    
    # ax3d.set_xticks([])
    # ax3d.set_yticks([])
    ax3d.set_xticklabels([])
    ax3d.set_yticklabels([])
    
    ax3d.set_title(f'{model} {year}', fontsize=5, pad=20)

    # --- Subplot 2: % Positive Map ---
    ax2d = fig.add_subplot(1, 2, 2)

    lon_coords_perc, lat_coords_perc, perc_list = [], [], []

    for (lat, lon), cell in res_model.items():
        if isinstance(cell.iloc[0], pd.DataFrame) and variable_name in cell.iloc[0].columns:
            df = cell.iloc[0]
            total_rows = len(df)
            num_models = total_rows // rows_per_model
            count_above_zero = 0

            for i in range(num_models):
                chunk = df[variable_name].iloc[i * rows_per_model:(i + 1) * rows_per_model]
                if chunk.mean() < 0:
                    count_above_zero += 1

            percentage = (count_above_zero / num_models) * 100
            lon_coords_perc.append(lon)
            lat_coords_perc.append(lat)
            perc_list.append(percentage)

    lon_coords_perc = np.array(lon_coords_perc)
    lat_coords_perc = np.array(lat_coords_perc)
    perc_list = np.array(perc_list)

    lon_unique = np.unique(lon_coords_perc)
    lat_unique = np.unique(lat_coords_perc)
    lon_mesh, lat_mesh = np.meshgrid(lon_unique, lat_unique)
    perc_grid = np.full_like(lon_mesh, np.nan, dtype=np.float64)

    for lon, lat, perc in zip(lon_coords_perc, lat_coords_perc, perc_list):
        i = np.where(lat_unique == lat)[0][0]
        j = np.where(lon_unique == lon)[0][0]
        perc_grid[i, j] = perc

    im = ax2d.pcolormesh(lon_mesh, lat_mesh, perc_grid, shading='auto', cmap=cmap_perc, vmin=0, vmax=100)
    cbar = fig.colorbar(im, ax=ax2d, shrink=0.8,orientation='horizontal')
    # cbar = fig.colorbar(im, ax=ax2d, shrink=0.8)
    

    cbar.set_label(f'%')
    # ax2d.set_xlabel('Longitude')
    # ax2d.set_ylabel('Latitude')
    
    ax2d.set_xticks([])
    ax2d.set_yticks([])
    ax2d.set_xticklabels([])
    ax2d.set_yticklabels([])
    
    ax2d.set_title('% RCM with mean values < 0 ')

    # --- Save and show ---
    
    vartitle= remove_slash_from_title(variable_name)
    os.makedirs("../Results/Row_Plots", exist_ok=True)

    base_path = f"../Results/Row_Plots/Row_{model}_{year}_{vartitle}_{angle}"
    
    plt.tight_layout()
    plt.savefig(f"{base_path}.jpg", dpi=900)
    # plt.savefig(f"{base_path}.pdf", dpi=600)
    print(f"Saved: {base_path}.jpg and {base_path}.pdf")
    plt.close()




    
def plot_variable_panel_grouped_by_column_q1q3surface(
    filename,
    dict_res_orga_gw_T,
    selected_models,
    variable,
    unit_label,
    vmin,
    vmax,
    cmap_3d='viridis',
    center_zero=True,
    angle=(60, 225),
    dict_plot_names=None
):


    assert len(selected_models) == 6, "This layout assumes exactly 6 selected models."
    print("Selected models:", selected_models)

    # Organize models into 3 columns, 2 models each (each with 3 years)
    # grouped_models = [selected_models[i:i+2] for i in range(0, 6, 2)]
    grouped_models = [selected_models[i:i+2] for i in range(0, len(selected_models), 2)]

    
    years = [2050, 2070, 2090]
    assert len(years) == 3, "This layout assumes exactly 3 years."

    rows = 6  # 2 models * 3 years
    cols = 3  # 3 columns of 2-models
    # fig = plt.figure(figsize=(5 * cols, 3 * rows))
    # fig = plt.figure(figsize=(8.27, 11.69))  # A4 portrait

    fig = plt.figure(figsize=(5 * cols, 3 * rows))  # Bigger subplots

    # gs = gridspec.GridSpec(rows, cols, hspace=0.2, wspace=0.10)
    gs = gridspec.GridSpec(rows, cols, hspace=0.1, wspace=0.01)


    for col_idx, model_pair in enumerate(grouped_models):
        for model_pos, model in enumerate(model_pair):
            for year_idx, year in enumerate(years):
                row_idx = model_pos * len(years) + year_idx
                ax3d = fig.add_subplot(gs[row_idx, col_idx], projection='3d')
                
                
            

                try:
                    res_model = dict_res_orga_gw_T[year][model]
                except KeyError:
                    ax3d.set_title("No data")
                    ax3d.axis('off')
                    continue

                ax3d.set_facecolor("white")
                ax3d.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
                ax3d.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
                ax3d.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))

                lon_coords, lat_coords, data_q1, data_q3 = [], [], [], []

                for (lat, lon), cell in res_model.items():
                    if isinstance(cell.iloc[0], pd.DataFrame) and variable in cell.iloc[0].columns:
                        values = cell.iloc[0][variable].values
                        lon_coords.append(lon)
                        lat_coords.append(lat)
                        data_q1.append(np.percentile(values, 25))
                        data_q3.append(np.percentile(values, 75))

                if not data_q1 or not data_q3:
                    ax3d.set_title(f"No data for {variable}")
                    ax3d.axis('off')
                    continue

                lon_coords = np.array(lon_coords)
                lat_coords = np.array(lat_coords)
                data_q1 = np.array(data_q1)
                data_q3 = np.array(data_q3)

                lon_unique = np.unique(lon_coords)
                lat_unique = np.unique(lat_coords)
                lon_mesh, lat_mesh = np.meshgrid(lon_unique, lat_unique)

                q1_surface = np.full_like(lon_mesh, np.nan, dtype=np.float64)
                q3_surface = np.full_like(lon_mesh, np.nan, dtype=np.float64)

                for lon, lat, q1, q3 in zip(lon_coords, lat_coords, data_q1, data_q3):
                    i = np.where(lat_unique == lat)[0][0]
                    j = np.where(lon_unique == lon)[0][0]
                    q1_surface[i, j] = q1
                    q3_surface[i, j] = q3

                if center_zero:
                    abs_max = max(abs(vmin), abs(vmax))
                    norm = TwoSlopeNorm(vmin=-abs_max, vcenter=0, vmax=abs_max)
                else:
                    norm = Normalize(vmin=vmin, vmax=vmax)

                surf1 = ax3d.plot_surface(lon_mesh, lat_mesh, q1_surface, cmap=cmap_3d, edgecolor='none', alpha=0.8, norm=norm)
                surf2 = ax3d.plot_surface(lon_mesh, lat_mesh, q3_surface, cmap=cmap_3d, edgecolor='none', alpha=0.8, norm=norm)

                for i in range(lon_mesh.shape[0]):
                    for j in range(lon_mesh.shape[1]):
                        if not np.isnan(q1_surface[i, j]) and not np.isnan(q3_surface[i, j]):
                            ax3d.plot([lon_mesh[i, j], lon_mesh[i, j]],
                                      [lat_mesh[i, j], lat_mesh[i, j]],
                                      [q1_surface[i, j], q3_surface[i, j]],
                                      color='grey', alpha=0.3)

                sm = ScalarMappable(cmap=cmap_3d, norm=norm)
                sm.set_array([])
                # cbar3d = fig.colorbar(sm, ax=ax3d, shrink=0.6, pad=0.1, orientation='horizontal')
                # cbar3d.set_label(f'{variable} (Q1–Q3)', rotation=0)

                ax3d.set_zlim(vmin * 1.4, vmax)
                ax3d.view_init(elev=angle[0], azim=angle[1])
                ax3d.set_xticklabels([])
                ax3d.set_yticklabels([])

                if dict_plot_names:
                    subtitle_name = dict_plot_names[model]
                else:
                    subtitle_name = model

                ax3d.set_title(f"{subtitle_name} — {year}", fontsize=8)

    fig.suptitle(f"{variable} (Q1–Q3 Surface)", fontsize=16, y=0.98)
    os.makedirs("../Results/Row_plots", exist_ok=True)
    filepath = f"../Results/Row_plots/{filename}.jpg"
    plt.savefig(filepath, dpi=1600, bbox_inches='tight')
    print(f"Saved panel: {filepath}")
    plt.close()
    






 

def extract_ssp(model_name):
    match = re.search(r'path(.*?)_(?:long|short)', model_name)
    return match.group(1) if match else 'Unknown'






def plot_variable_distributions_by_ssp_and_year(dict_res, variable_name):



    # Organize values: ssp > year > model -> list of values
    ssp_data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    all_values = []
    stats_records = []

    for year, year_dict in dict_res.items():
        for model, res_model in year_dict.items():
            ssp = extract_ssp(model)
            for (lat, lon), cell in res_model.items():
                if isinstance(cell.iloc[0], pd.DataFrame) and variable_name in cell.iloc[0].columns:
                    values = cell.iloc[0][variable_name].values
                    ssp_data[ssp][year][model].extend(values)
                    all_values.extend(values)

    # Global axis limits
    x_min, x_max = np.nanmin(all_values), np.nanmax(all_values)
    max_density = 0

    # Estimate max density
    for ssp in ssp_data:
        for year in ssp_data[ssp]:
            for model, values in ssp_data[ssp][year].items():
                if values:
                    kde = sns.kdeplot(values, bw_adjust=1)
                    y_vals = kde.get_lines()[0].get_data()[1]
                    max_density = max(max_density, np.nanmax(y_vals))
                    plt.clf()

    # Actual plots and statistics collection
    for ssp, year_dict in ssp_data.items():
        for year, model_dict in year_dict.items():
            plt.figure(figsize=(10, 5))

            for model, values in model_dict.items():
                if values:
                    kde = sns.kdeplot(values, bw_adjust=1, label=model)
                    line = kde.get_lines()[-1]
                    x_vals, y_vals = line.get_data()
                    color = line.get_color()
                    plt.fill_between(x_vals, y_vals, alpha=0.1, color=color)

                    median_val = np.nanmedian(values)
                    plt.axvline(median_val, color=color, linestyle=':', linewidth=1.0, alpha=0.8)

                    # Record statistics
                    values_array = np.array(values)
                    mean = np.nanmean(values_array)
                    std = np.nanstd(values_array)
                    min_val = np.nanmin(values_array)
                    max_val = np.nanmax(values_array)
                    median = np.nanmedian(values_array)
                    cv = std / mean if mean != 0 else np.nan

                    stats_records.append({
                        "SSP": ssp,
                        "Year": year,
                        "Model": model,
                        "Variable": variable_name,
                        "Mean": mean,
                        "Std": std,
                        "Min": min_val,
                        "Max": max_val,
                        "Median": median,
                        "Q1":np.percentile(values_array,25),
                        "Q3":np.percentile(values_array,75),
                        "CV": cv
                    })

            plt.title(f"Distribution of {variable_name}\nSSP: {ssp} — Year: {year}")
            plt.xlabel(variable_name)
            plt.ylabel("Density")
            plt.xlim(x_min, x_max + 0.1)
            plt.ylim(0, max_density * 1.1)
            plt.legend(fontsize=6, title="Model", loc='upper left')
            plt.tight_layout()

            safe_varname = variable_name.replace('/', '_')
            os.makedirs("../Results/Distrib", exist_ok=True)
            os.makedirs(f"../Results/Distrib/{ssp}", exist_ok=True)

            base_path = f"../Results/Distrib/{ssp}/Distribution_SSP_{ssp}_{year}_{safe_varname}"
            plt.savefig(f"{base_path}.jpg", dpi=500)
            print(f"Saved: {base_path}.jpg")

            plt.close()

    # Save all statistics to CSV
    stats_df = pd.DataFrame(stats_records)
    os.makedirs("../Results/Distrib/Statistics", exist_ok=True)
    stats_csv_path = f"../Results/Distrib/Statistics/Stats_{safe_varname}.csv"
    stats_df.to_csv(stats_csv_path, index=False)
    print(f"Saved statistics to: {stats_csv_path}")



def export_distributions_by_ssp_and_year(dict_res, variable_name):
    from collections import defaultdict




    # Organize values: ssp > year > model -> list of values
    ssp_data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    all_values = []
    stats_records = []

    for year, year_dict in dict_res.items():
        for model, res_model in year_dict.items():
            ssp = extract_ssp(model)
            for (lat, lon), cell in res_model.items():
                if isinstance(cell.iloc[0], pd.DataFrame) and variable_name in cell.iloc[0].columns:
                    values = cell.iloc[0][variable_name].values
                    ssp_data[ssp][year][model].extend(values)
                    all_values.extend(values)

    # Global axis limits
    x_min, x_max = np.nanmin(all_values), np.nanmax(all_values)
    max_density = 0

    # Estimate max density
    for ssp in ssp_data:
        for year in ssp_data[ssp]:
            for model, values in ssp_data[ssp][year].items():
                if values:
                    kde = sns.kdeplot(values, bw_adjust=1)
                    y_vals = kde.get_lines()[0].get_data()[1]
                    max_density = max(max_density, np.nanmax(y_vals))
                    plt.clf()

    # Actual plots and statistics collection
    for ssp, year_dict in ssp_data.items():
        for year, model_dict in year_dict.items():
            # plt.figure(figsize=(10, 5))

            for model, values in model_dict.items():
                if values:


                    # Record statistics
                    values_array = np.array(values)
                    mean = np.nanmean(values_array)
                    std = np.nanstd(values_array)
                    min_val = np.nanmin(values_array)
                    max_val = np.nanmax(values_array)
                    median = np.nanmedian(values_array)
                    cv = std / mean if mean != 0 else np.nan

                    stats_records.append({
                        "SSP": ssp,
                        "Year": year,
                        "Model": model,
                        "Variable": variable_name,
                        "Mean": mean,
                        "Std": std,
                        "Min": min_val,
                        "Max": max_val,
                        "Median": median,
                        "Q1":np.percentile(values_array,25),
                        "Q3":np.percentile(values_array,75),
                        "CV": cv
                    })


    safe_varname = variable_name.replace('/', '_')

    # Save all statistics to CSV
    stats_df = pd.DataFrame(stats_records)
    os.makedirs("../Results/Distrib/Statistics", exist_ok=True)
    stats_csv_path = f"../Results/Distrib/Statistics/Stats_{safe_varname}.csv"
    stats_df.to_csv(stats_csv_path, index=False)
    print(f"Saved statistics to: {stats_csv_path}")







def remove_slash_from_title(title):
# Replace any slashes with an empty string
    return title.replace('/', '')



# ok
def plot_combined_surface_all_models_per_year(
    year,
    dict_res_orga_gw_T,
    variable_name,
    vmin,
    vmax,
    config,
    angle=(60, 225),
    cmap_3d="coolwarm",
    center_zero=True
):


    def remove_slash_from_title(title):
        return title.replace("/", "_").replace("\\", "_")

    # Step 1: Combine data across models for this year
    combined_data = {}

    for model, res_model in dict_res_orga_gw_T[year].items():
        for (lat_, lon_), cell in res_model.items():
            if isinstance(cell.iloc[0], pd.DataFrame) and variable_name in cell.iloc[0].columns:
                values = cell.iloc[0][variable_name].values
                key = (lat_, lon_)
                if key not in combined_data:
                    combined_data[key] = []
                combined_data[key].extend(values)

    if not combined_data:
        print(f"No data for year {year}")
        return

    # Step 2: Prepare surface data
    lon_coords, lat_coords, data_q1, data_q3 = [], [], [], []

    for (lat_, lon_), values in combined_data.items():
        if len(values) > 0:
            lon_coords.append(lon_)
            lat_coords.append(lat_)
            data_q1.append(np.percentile(values, 25))
            data_q3.append(np.percentile(values, 75))

    lon_coords = np.array(lon_coords)
    lat_coords = np.array(lat_coords)
    data_q1 = np.array(data_q1)
    data_q3 = np.array(data_q3)

    # Export stats
    stats = {
        "Q1_min": np.min(data_q1),
        "Q1_max": np.max(data_q1),
        "Q1_mean": np.mean(data_q1),
        "Q1_median": np.median(data_q1),
        "Q3_min": np.min(data_q3),
        "Q3_max": np.max(data_q3),
        "Q3_mean": np.mean(data_q3),
        "Q3_median": np.median(data_q3),
    }

    stats_df = pd.DataFrame([stats])
    vartitle = remove_slash_from_title(variable_name)
    os.makedirs("../Results/Combined_Yearly_3D", exist_ok=True)
    stats_path = f"../Results/Combined_Yearly_3D/Stats_{year}_{vartitle}_{config}.csv"
    stats_df.to_csv(stats_path, index=False)
    print(f"Saved statistics to {stats_path}")

    # Step 3: Surface creation
    lon_unique = np.unique(lon_coords)
    lat_unique = np.unique(lat_coords)
    lon_mesh, lat_mesh = np.meshgrid(lon_unique, lat_unique)

    q1_surface = np.full_like(lon_mesh, np.nan, dtype=np.float64)
    q3_surface = np.full_like(lon_mesh, np.nan, dtype=np.float64)

    for lon_, lat_, q1, q3 in zip(lon_coords, lat_coords, data_q1, data_q3):
        i = np.where(lat_unique == lat_)[0][0]
        j = np.where(lon_unique == lon_)[0][0]
        q1_surface[i, j] = q1
        q3_surface[i, j] = q3

    # Step 4: Plotting
    fig = plt.figure(figsize=(8, 6))
    ax3d = fig.add_subplot(111, projection='3d')
    ax3d.set_facecolor("white")

    ax3d.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))  # X pane
    ax3d.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))  # Y pane
    ax3d.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))  # Z pane

    ax3d.set_zlim(vmin * 1.4, vmax)

    if center_zero:
        abs_max = max(abs(vmin), abs(vmax))
        norm = TwoSlopeNorm(vmin=-abs_max, vcenter=0, vmax=abs_max)
    else:
        norm = Normalize(vmin=vmin, vmax=vmax)

    surf1 = ax3d.plot_surface(lon_mesh, lat_mesh, q1_surface, cmap=cmap_3d, norm=norm, alpha=0.7)
    surf2 = ax3d.plot_surface(lon_mesh, lat_mesh, q3_surface, cmap=cmap_3d, norm=norm, alpha=0.7)

    for i in range(lon_mesh.shape[0]):
        for j in range(lon_mesh.shape[1]):
            if not np.isnan(q1_surface[i, j]) and not np.isnan(q3_surface[i, j]):
                ax3d.plot([lon_mesh[i, j]] * 2,
                          [lat_mesh[i, j]] * 2,
                          [q1_surface[i, j], q3_surface[i, j]],
                          color='grey', alpha=0.3)

    ax3d.view_init(elev=angle[0], azim=angle[1])
    ax3d.set_xticklabels([])
    ax3d.set_yticklabels([])
    ax3d.set_facecolor('white')

    sm = ScalarMappable(cmap=cmap_3d, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax3d, shrink=0.7, pad=0.1, label=f'{variable_name}_{config} (Q1–Q3)')
    ax3d.set_title(f"{year}", pad=20)
    ax3d.set_facecolor("white")

    plt.tight_layout()
    plt.savefig(f"../Results/Combined_Yearly_3D/AllModels_{year}_{vartitle}_{config}_{angle}.jpg", dpi=1000)
    plt.close()
    print(f"Saved 3D plot for {year}")





def compute_global_min_max_q3(dict_res_orga_gw_T, variable_name):


    q3_values = []
    q1_values = []

    for year in dict_res_orga_gw_T:
        combined_data = {}

        for model, res_model in dict_res_orga_gw_T[year].items():
            for (lat_, lon_), cell in res_model.items():
                if isinstance(cell.iloc[0], pd.DataFrame) and variable_name in cell.iloc[0].columns:
                    values = cell.iloc[0][variable_name].values
                    key = (lat_, lon_)
                    if key not in combined_data:
                        combined_data[key] = []
                    combined_data[key].extend(values)

        for values in combined_data.values():
            if len(values) > 0:
                q3 = np.percentile(values, 75)
                q1 = np.percentile(values, 25)

                q3_values.append(q3)
                q1_values.append(q1)

    if not q3_values:
        raise ValueError("No valid data found for computing global min/max.")

    # global_min = np.min(q1_values)  # for contrast
    global_min = np.percentile(q1_values,2)

    global_max = np.percentile(q3_values,98)
    
    
    # global_min = np.min(q1_values) # for crop main
    # global_max = np.max(q3_values)

    return global_min, global_max







def plot_percentage_below_zero_map(dict_res_orga_gw_T, variable_name, config, save_dir="../Results/Aggregated_Perc_Below_Zero"):
    os.makedirs(save_dir, exist_ok=True)

    for year, year_dict in dict_res_orga_gw_T.items():
        # Step 1: Aggregate values per location
        combined_data = {}

        for model, res_model in year_dict.items():
            for (lat, lon), cell in res_model.items():
                if isinstance(cell.iloc[0], pd.DataFrame) and variable_name in cell.iloc[0].columns:
                    values = cell.iloc[0][variable_name].values
                    key = (lat, lon)
                    if key not in combined_data:
                        combined_data[key] = []
                    combined_data[key].extend(values)

        # Step 2: Compute percentage below zero
        lon_coords, lat_coords, perc_below_zero = [], [], []

        for (lat, lon), values in combined_data.items():
            if len(values) > 0:
                print(len(values))
                values = np.array(values)
                perc = np.sum(values < 0) / len(values) * 100
                lat_coords.append(lat)
                lon_coords.append(lon)
                perc_below_zero.append(perc)

        # Step 3: Convert to grid
        lon_unique = np.unique(lon_coords)
        lat_unique = np.unique(lat_coords)
        lon_mesh, lat_mesh = np.meshgrid(lon_unique, lat_unique)
        perc_grid = np.full_like(lon_mesh, np.nan, dtype=np.float64)

        for lon, lat, perc in zip(lon_coords, lat_coords, perc_below_zero):
            i = np.where(lat_unique == lat)[0][0]
            j = np.where(lon_unique == lon)[0][0]
            perc_grid[i, j] = perc

        # Step 4: Plot
        plt.figure(figsize=(6, 5))
        im = plt.pcolormesh(lon_mesh, lat_mesh, perc_grid, shading='auto', cmap='RdBu', vmin=0, vmax=100)
        plt.colorbar(im, label=f'% of values < 0\n({variable_name})')
        plt.title(f"{year} - % Below Zero {config}", fontsize=10)
        plt.xticks([])
        plt.yticks([])
        plt.tight_layout()

        save_path = os.path.join(save_dir, f"PercBelowZero_{year}_{config}_{variable_name.replace('/', '_')}.jpg")
        plt.savefig(save_path, dpi=600)
        plt.close()
        print(f"Saved: {save_path}")





def combine_all_stats_to_wide_multiheader(
    file_list,
    output_filename,
    folder="../Results/Panel_Plots",
    excel=False,
    round_digits=None,
    var_name_map=None
):
    """
    Combine multiple Combined_Stats_*.csv files into one wide-format file
    with two header rows: first = variable (renamed if var_name_map provided),
    second = statistic name.

    Parameters:
    - file_list (list of str): List of CSV filenames (inside `folder`) to combine.
    - output_filename (str): Name of the output file (.csv or .xlsx).
    - folder (str): Folder where the input files are stored.
    - excel (bool): If True, saves as Excel with multi-row header.
    - round_digits (int or None): If given, rounds numeric values to this many decimals.
    - var_name_map (dict or None): Mapping {original_variable_name: display_name}.
    """
    dfs = []
    for file in file_list:
        path = os.path.join(folder, file)
        if not os.path.exists(path):
            print(f"Warning: {file} not found in {folder}")
            continue

        df = pd.read_csv(path)
        stats_cols = ["Min", "Max", "Mean", "Median"]

        # Pivot to (stat, variable)
        df_wide = df.pivot_table(
            index=["Model", "Year"],
            columns="Variable",
            values=stats_cols
        )

        # Swap so that (variable, stat)
        df_wide.columns = df_wide.columns.swaplevel(0, 1)

        # Optionally rename variable names
        if var_name_map:
            df_wide.columns = pd.MultiIndex.from_tuples(
                [(var_name_map.get(var, var), stat) for var, stat in df_wide.columns],
                names=["Variable", "Statistic"]
            )

        df_wide = df_wide.sort_index(axis=1, level=0)
        dfs.append(df_wide.reset_index())

    if not dfs:
        print("No data combined.")
        return pd.DataFrame()

    # Merge on Model + Year
    combined_df = reduce(
        lambda left, right: pd.merge(left, right, on=["Model", "Year"], how="outer"),
        dfs
    )

    # Optional rounding
    if round_digits is not None:
        num_cols = combined_df.select_dtypes(include="number").columns
        combined_df[num_cols] = combined_df[num_cols].round(round_digits)

    # Adjust extension automatically
    if excel and not output_filename.lower().endswith(".xlsx"):
        output_filename += ".xlsx"
    if not excel and not output_filename.lower().endswith(".csv"):
        output_filename += ".csv"

    output_path = os.path.join(folder, output_filename)
    if excel:
        combined_df.to_excel(output_path, index=False, merge_cells=False)
    else:
        combined_df.to_csv(output_path, index=False)

    print(f"Saved combined file: {output_path}")
    return combined_df