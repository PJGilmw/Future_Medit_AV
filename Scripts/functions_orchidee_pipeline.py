# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 11:57:27 2024

@author: pjouannais


Functions which constitute the pipeline from ORCHIDEE to the LCA model + random sampling functions

"""



import numpy as np
from scipy import sparse


import matrix_utils as mu
import numpy as np
import pandas as pd
import sys
from itertools import *
import itertools


from util_functions import *


from Main_functions import *
import Main_functions



import itertools


import sys

import re
from scipy.stats.qmc import LatinHypercube


import xarray as xr
import cftime


from sklearn.linear_model import LinearRegression




from scipy.stats import norm


def merge_grid_points_auto_4_b(dataset, method='trim'):
    """
    Merge 16 grid points into 1 by averaging their values, keeping the central coordinate
    as the new grid point. Handles NaN values by averaging over remaining points:
      - 1-12 NaNs: Average computed with available points
      - 12+ NaNs: New point is NaN.
    Automatically processes all variables in the dataset except 'time'.

    Args:
        dataset (xarray.Dataset): The original dataset.
        method (str): How to handle uneven dimensions. Options are 'trim' or 'pad'.
                      - 'trim': Trims rows/columns from the end of uneven dimensions.
                      - 'pad': Pads the dataset with NaNs to make dimensions divisible by 4.

    Returns:
        xarray.Dataset: A new dataset with merged grid points.
    """
    lat_dim = dataset.sizes['lat']
    lon_dim = dataset.sizes['lon']

    # Handle uneven dimensions
    if lat_dim % 4 != 0 or lon_dim % 4 != 0:
        if method == 'trim':
            dataset = dataset.isel(
                lat=slice(0, lat_dim - lat_dim % 4),
                lon=slice(0, lon_dim - lon_dim % 4)
            )
        elif method == 'pad':
            pad_lat = (4 - lat_dim % 4) if lat_dim % 4 != 0 else 0
            pad_lon = (4 - lon_dim % 4) if lon_dim % 4 != 0 else 0
            dataset = dataset.pad(
                lat=(0, pad_lat),
                lon=(0, pad_lon),
                constant_values=np.nan
            )
        else:
            raise ValueError("Invalid method. Use 'trim' or 'pad'.")

    # Adjusted latitude and longitude dimensions
    lat_dim = dataset.sizes['lat']
    lon_dim = dataset.sizes['lon']
    lat_new = dataset['lat'].values.reshape(-1, 4).mean(axis=1)
    lon_new = dataset['lon'].values.reshape(-1, 4).mean(axis=1)

    # Create new dimensions for lat and lon
    new_lat = xr.DataArray(lat_new, dims='lat', name='lat')
    new_lon = xr.DataArray(lon_new, dims='lon', name='lon')

    # Create a new dataset to hold merged data
    merged_dataset = xr.Dataset(coords={'lat': new_lat, 'lon': new_lon})

    # Process each variable
    for var in dataset.data_vars:
        dims = dataset[var].dims
        data = dataset[var].values

        if 'lat' in dims and 'lon' in dims:
            # Reshape and group lat/lon dimensions
            reshaped_data = data.reshape(
                (-1, lat_dim // 4, 4, lon_dim // 4, 4, *data.shape[3:])
            )
            # Merge grid points while handling NaN values
            avg_data = np.where(
                np.isnan(reshaped_data).sum(axis=(2, 4)) >= 8,  # If 12+ NaNs
                np.nan,
                np.nanmean(reshaped_data, axis=(2, 4))  # Compute mean ignoring NaNs
            )

            # Reassign dimensions
            merged_dims = [
                dim if dim not in {'lat', 'lon'} else 'lat' if dim == 'lat' else 'lon'
                for dim in dims
            ]
            merged_dataset[var] = (merged_dims, avg_data)

        else:
            # Variables without lat/lon remain unchanged
            merged_dataset[var] = dataset[var]

    return merged_dataset, new_lat, new_lon


def merge_grid_points_auto_2_b(dataset, method='trim'):  # Actually 2 
    """
    Merge 4 grid points into 1 by averaging their values, keeping the central coordinate
    as the new grid point. Handles NaN values by averaging over remaining points:
      - 1 NaN: Average computed with 3 points
      - 2 NaNs: Average computed with 2 points
      - 3+ NaNs: New point is NaN.
    Automatically processes all variables in the dataset except 'time'.

    Args:
        dataset (xarray.Dataset): The original dataset.
        method (str): How to handle uneven dimensions. Options are 'trim' or 'pad'.
                      - 'trim': Trims one row/column from the end of uneven dimensions.
                      - 'pad': Pads the dataset with NaNs to make dimensions even.

    Returns:
        xarray.Dataset: A new dataset with merged grid points.
    """
    lat_dim = dataset.sizes['lat']
    lon_dim = dataset.sizes['lon']

    # Handle uneven dimensions
    if lat_dim % 2 != 0 or lon_dim % 2 != 0:
        if method == 'trim':
            dataset = dataset.isel(
                lat=slice(0, lat_dim - lat_dim % 2),
                lon=slice(0, lon_dim - lon_dim % 2)
            )
        elif method == 'pad':
            pad_lat = 1 if lat_dim % 2 != 0 else 0
            pad_lon = 1 if lon_dim % 2 != 0 else 0
            dataset = dataset.pad(
                lat=(0, pad_lat),
                lon=(0, pad_lon),
                constant_values=np.nan
            )
        else:
            raise ValueError("Invalid method. Use 'trim' or 'pad'.")

    # Adjusted latitude and longitude dimensions
    lat_dim = dataset.sizes['lat']
    lon_dim = dataset.sizes['lon']
    lat_new = dataset['lat'].values.reshape(-1, 2).mean(axis=1)
    lon_new = dataset['lon'].values.reshape(-1, 2).mean(axis=1)

    # Create new dimensions for lat and lon
    new_lat = xr.DataArray(lat_new, dims='lat', name='lat')
    new_lon = xr.DataArray(lon_new, dims='lon', name='lon')

    # Create a new dataset to hold merged data
    merged_dataset = xr.Dataset(coords={'lat': new_lat, 'lon': new_lon})

    # Process each variable
    for var in dataset.data_vars:
        dims = dataset[var].dims
        data = dataset[var].values

        if 'lat' in dims and 'lon' in dims:
            # Reshape and group lat/lon dimensions
            reshaped_data = data.reshape(
                (-1, lat_dim // 2, 2, lon_dim // 2, 2, *data.shape[3:])
            )
            # Merge grid points while handling NaN values
            avg_data = np.where(
                np.isnan(reshaped_data).sum(axis=(2, 4)) >= 3,  # If 3+ NaNs
                np.nan,
                np.nanmean(reshaped_data, axis=(2, 4))  # Compute mean ignoring NaNs
            )

            # Reassign dimensions
            merged_dims = [
                dim if dim not in {'lat', 'lon'} else 'lat' if dim == 'lat' else 'lon'
                for dim in dims
            ]
            merged_dataset[var] = (merged_dims, avg_data)

        else:
            # Variables without lat/lon remain unchanged
            merged_dataset[var] = dataset[var]

    return merged_dataset, new_lat, new_lon







def convert_npp_yield(npp, dry,keep_npp):
    
    """ dry: True if the returned yield is dry weight,
        keep_npp: True for computing the yield later to consider the parameters uncertainties. False for intermediate plotting of yield with average parameter values.
    """
    
    # from npp in gC.m-2.y-1 to kg . ha-1 . y-1
    
    if not keep_npp:
        
        root_shoot_ratio = 0.21
        straw_grain_ratio = 1.35
        carbon_content_grain = 0.4292
        carbon_content_straw = 0.446
        water_content_grain = 0.14
        water_content_straw = 0.14
            
        
        A = ((1-water_content_grain) * carbon_content_grain + (1-water_content_straw) * straw_grain_ratio * carbon_content_straw )
        
        grain_yield  = npp/((1+root_shoot_ratio) * A*100) # t dry. ha-1 . y-1
        
        grain_yield = grain_yield *1000 # t to kg
        if dry:
            return grain_yield    # kg dry. ha-1 . y-1
        else:
            return grain_yield/(1-water_content_grain) # # kg wet. ha-1 . y-1
            
    else:
        return npp
        





def pipeline_orchidee_acv_multiple_models_npp_water_lower_res(
    ds_plant_list, ds_elec_avs_list, ds_elec_refpv_list, corres_name_orchidee_acv,
    dry,keep_npp ):
    """
    Process multiple datasets of the same structure and return yearly DataFrames
    where each cell contains a list of values across the datasets.

    Args:
        ds_plant_list: List of xarray.Dataset objects for plant data.
        ds_elec_avs_list: List of xarray.Dataset objects for AVS electricity data.
        ds_elec_refpv_list: List of xarray.Dataset objects for reference PV electricity data.
        corres_name_orchidee_acv: Dictionary mapping dataset variable names to ACV names.

    Returns:
        Dictionary where keys are years and values are DataFrames containing lists of values.
    """
    # Combine datasets
    combined_datasets = []

    for ds_plant, ds_elec_avs, ds_elec_refpv in zip(ds_plant_list, ds_elec_avs_list, ds_elec_refpv_list):
        #print(list(ds_plant.variables))
        #print(list(ds_elec_refpv.variables))
        #print(list(ds_elec_avs.variables))
        ds_combined = ds_plant.copy()
        ds_combined = ds_combined.assign(Y_ST_avs=ds_elec_avs['Y_ST'])
        ds_combined = ds_combined.assign(Y_ST_ref=ds_elec_avs['Y_ST'])
        ds_combined = ds_combined.assign(Y_fixed_ref=ds_elec_refpv['Y_20'])
        ds_combined = ds_combined.assign(crop_yield_ref=convert_npp_yield(ds_combined['NPP'],dry,keep_npp))
        ds_combined = ds_combined.assign(crop_yield_avs=convert_npp_yield(ds_combined['NPP_ST'],dry,keep_npp))
        ds_combined = ds_combined.assign(Water_efficiency_ref=ds_combined['crop_yield_ref']/(ds_combined["Evap"]+ds_combined["Transpir"]))
        ds_combined = ds_combined.assign(Water_efficiency_avs=ds_combined['crop_yield_avs']/(ds_combined["Evap_ST"]+ds_combined["Transpir_ST"]))


        #ds_combined, new_lat, new_lon = merge_grid_points_auto_2_b(ds_combined, method='trim')
        ds_combined, new_lat, new_lon = merge_grid_points_auto_4_b(ds_combined, method='trim')


        combined_datasets.append(ds_combined)

    # Convert cftime or datetime to years
    def extract_year_from_time(ds):
        """Extract years from the time variable and replace it in the dataset."""
        years = [int(time.year) for time in ds['time'].values]
        ds = ds.assign_coords(time=("time", years))  # Replace the time coordinate with years
        return ds

    # Apply year extraction
    combined_datasets = [extract_year_from_time(ds) for ds in combined_datasets]

    # Shared dimensions (assumes all datasets have the same lat/lon)
    lat = combined_datasets[0]['lat'].values
    lon = combined_datasets[0]['lon'].values
    coords = [(float(lat[i]), float(lon[j])) for i in range(len(lat)) for j in range(len(lon))]

    # Create a mapping from (lat, lon) to its index in the flattened array
    coord_to_index = {
        (lat[i], lon[j]): (i, j)
        for i in range(len(lat)) for j in range(len(lon))
    }

    # Extract shared years
    years = combined_datasets[0].time.data.tolist()

    # List to hold DataFrames for each year
    yearly_dataframes = {}

    # Loop over each year
    for year_idx, year in enumerate(years):
        data_dict = {}

        # Process each variable across all datasets
        for var_name in corres_name_orchidee_acv.keys():
            name_param_acv = corres_name_orchidee_acv[var_name]

            # Extract data for the current year across all datasets
            all_data_for_year = [
                ds.isel(tstep=year_idx)[var_name].values for ds in combined_datasets
            ]

            # Collect data in the order of the coords list
            combined_data = []
            for coord in coords:
                lat_idx, lon_idx = coord_to_index.get(coord, (None, None))
                if lat_idx is not None and lon_idx is not None:
                    combined_data.append(
                        [data_for_year[lat_idx, lon_idx] for data_for_year in all_data_for_year]
                    )
                else:
                    combined_data.append([None] * len(all_data_for_year))  # Handle missing coordinates

            # Add to the data dictionary
            data_dict[name_param_acv] = combined_data

        # Create DataFrame: rows are variables, columns are coordinates
        df = pd.DataFrame(data_dict, index=coords, dtype=object).transpose()
        yearly_dataframes[year] = df

    return yearly_dataframes, new_lat, new_lon





def add_C_accumulation_wholegrid(cleaned_yearly_dataframes_orchidee,
                                           yearstart,
                                           yearend):
    
    """Compute and add the carbon accumulation for the AVS on the whole grid """
    
    # print(cleaned_yearly_dataframes_orchidee[2007].index)
    
    first_indice = list(cleaned_yearly_dataframes_orchidee[2007].columns)[0]
    
    n_models = len(cleaned_yearly_dataframes_orchidee[2007][first_indice]["total_soil_carbon ref"])
   
    for year in cleaned_yearly_dataframes_orchidee: # every year
    
        #print(df)
        
        cleaned_yearly_dataframes_orchidee[year].loc["soil carbon accumulation avs"] = [[5,5] for i in range(len(cleaned_yearly_dataframes_orchidee[year].columns))]
    
    
    # return cleaned_yearly_dataframes_orchidee
    # Iterate over all coordinates
    for coord in cleaned_yearly_dataframes_orchidee[yearstart].columns:
        
        
        all_years_same_coord = {
            key: cleaned_yearly_dataframes_orchidee[key][coord]
            for key in cleaned_yearly_dataframes_orchidee
        }
        
        list_res_for_all_models = compute_slope_C_accumulation_one_coord(all_years_same_coord,
                                         coord,
                                         n_models,
                                         yearstart,
                                         yearend)
        
        #print(list_res_for_all_models)
        for year in cleaned_yearly_dataframes_orchidee: # every year
        
            #print(cleaned_yearly_dataframes_orchidee[year].index)

            # uuu.loc["soil carbon accumulation avs"][(0.0, 0.0)] = [9,9]

            cleaned_yearly_dataframes_orchidee[year].loc["soil carbon accumulation avs"][coord] =list_res_for_all_models

    return cleaned_yearly_dataframes_orchidee




def compute_slope_C_accumulation_one_coord(all_years_same_coord,
                                 coord,
                                 n_models,
                                 yearstart,
                                 yearend):
    
    """Compute and add the carbon accumulation for the AVS on one point """

    #print(all_years_same_coord[2050].index)
    # Predefine years for regression
    years = np.array(range(yearstart, yearend)).reshape(-1, 1)
    # Number of models
    
    list_res_for_all_models = []
    for model_i in range(n_models):
        #print(model_i)
        
        # Extract model-specific data for all years
        model_C_ref = [
            all_years_same_coord[year]["total_soil_carbon ref"][model_i]
            for year in range(yearstart, yearend)
        ]
        model_C_avs = [
            all_years_same_coord[year]["total_soil_carbon avs"][model_i]
            for year in range(yearstart, yearend)
        ]
        
        # Calculate delta C
        deltaC = np.array([avs - ref for avs, ref in zip(model_C_avs, model_C_ref)][:yearend-yearstart])
        
        # Fit the linear regression model
        model = LinearRegression().fit(years, deltaC)
        
        # Extract slope
        slope = model.coef_[0]

        list_res_for_all_models.append(slope)
        
    return list_res_for_all_models






def calculate_sliding_avg_and_std_diff_two_vars(yearly_dataframes, lifetime):
    """
    Calculate sliding averages and standard deviation of year-to-year differences for two specified variables.

    Args:
        yearly_dataframes: Dictionary where keys are years and values are DataFrames containing lists of data values.
        lifetime: The number of years to include in the sliding window.

    Returns:
        A dictionary where keys are years and values are DataFrames with sliding averages (lists) 
        and standard deviation of year-to-year differences (lists) for the two specified variables.
    """
    # Get sorted list of years and DataFrames
    years = sorted(yearly_dataframes.keys())
    num_years = len(years)
    
    # Combine all DataFrames into a 3D NumPy array for processing
    sample_df = yearly_dataframes[years[0]]  # Get structure from the first DataFrame
    num_coords = sample_df.shape[1]
    num_vars = sample_df.shape[0]

    # Initialize 3D array (years, variables, coordinates)
    data_array = np.empty((num_years, num_vars, num_coords), dtype=object)
    for i, year in enumerate(years):
        data_array[i] = yearly_dataframes[year].values

    # Identify the indices of the two variables
    cropyieldref_idx = list(sample_df.index).index("crop yield ref")
    cropyieldavs_idx = list(sample_df.index).index("crop yield avs")

    # Initialize a dictionary for sliding averages and standard deviations
    sliding_avg_dataframes = {}

    # Process each year
    for start_idx, year in enumerate(years):
        # Define the sliding window
        end_idx = min(start_idx + lifetime, num_years)
        
        # Collect the sliding window data
        window_data = data_array[start_idx:end_idx]  # Shape: (window, variables, coordinates)

        # Initialize averaged data array
        averaged_data = np.empty((num_vars + 2, num_coords), dtype=object)  # Two extra rows for "std_dif_var1" and "std_dif_var2"

        for var_idx in range(num_vars):
            for coord_idx in range(num_coords):
                # Collect lists of values for the sliding window
                lists_to_average = [window_data[year_idx, var_idx, coord_idx] for year_idx in range(end_idx - start_idx)]
                
                # Transpose lists to group values by dataset
                transposed_values = list(zip(*lists_to_average))
                
                # Compute averages for each dataset
                averaged_data[var_idx, coord_idx] = [
                    np.nanmean(dataset_values) if dataset_values else np.nan for dataset_values in transposed_values
                ]

                # Compute standard deviation of year-to-year differences for the first variable
                if var_idx == cropyieldref_idx:
                    year_to_year_differences_var1 = [
                        np.diff(dataset_values) for dataset_values in transposed_values if len(dataset_values) > 1
                    ]
                    averaged_data[-2, coord_idx] = [
                        np.nanstd(differences) if len(differences) > 0 else np.nan
                        for differences in year_to_year_differences_var1
                    ]

                # Compute standard deviation of year-to-year differences for the second variable
                elif var_idx == cropyieldavs_idx:
                    year_to_year_differences_var2 = [
                        np.diff(dataset_values) for dataset_values in transposed_values if len(dataset_values) > 1
                    ]
                    averaged_data[-1, coord_idx] = [
                        np.nanstd(differences) if len(differences) > 0 else np.nan
                        for differences in year_to_year_differences_var2
                    ]

        # Create a new DataFrame for the sliding average and standard deviations
        new_index = list(sample_df.index) + ["std_dif_cropyieldref", "std_dif_cropyieldavs"]
        sliding_avg_df = pd.DataFrame(
            averaged_data, index=new_index, columns=sample_df.columns
        )
        sliding_avg_dataframes[year] = sliding_avg_df

    return sliding_avg_dataframes





def clean_dataframe(df):
    """Remove columns where any cell contains a list with NaN."""
    # Keep only columns where no list contains NaN
    return df.loc[:, df.applymap(lambda cell: not any(np.isnan(x) for x in cell)).all()]





def dataframe_to_dataset(df, lat_vals, lon_vals, var_name, time,number_model):
    """
    Convert a single DataFrame for a specific year into an xarray.Dataset.

    Args:
        df: The DataFrame to convert (rows: variables, columns: coordinates).
        lat_vals: The array of latitude values.
        lon_vals: The array of longitude values.
        var_name: The row name of the variable to convert.
        time: The year to set as the time coordinate.

    Returns:
        xarray.Dataset with the variable reorganized into the original grid format.
    """
    # Initialize a 2D array for the selected variable
    lat_dim = len(lat_vals)
    lon_dim = len(lon_vals)
    data_2d = np.full((lat_dim, lon_dim), np.nan)  # Use NaN as default

    # Iterate through the coordinates and populate the 2D array
    for coord, values in df.loc[var_name].items():
        
        #print(coord)
        
        #print(float(coord[0]))
        try:
            lat_idx = np.where(np.isclose(lat_vals, float(coord[0])))[0]
            lon_idx = np.where(np.isclose(lon_vals, float(coord[1])))[0]

            if len(lat_idx) == 0 or len(lon_idx) == 0:
                print(f"Skipping coordinate {coord} as it is not found.")
                continue

            # Use the first match from np.where
            lat_idx, lon_idx = lat_idx[0], lon_idx[0]
            data_2d[lat_idx, lon_idx] = values[number_model]  # Take the first value in the list
        except Exception as e:
            print(f"Error processing coordinate {coord}: {e}")
            continue

    # Convert the 2D data to a Dataset
    dataset = xr.Dataset(
        {
            var_name: (["lat", "lon"], data_2d)
        },
        coords={
            "lat": lat_vals,
            "lon": lon_vals,
            "time": [time]
        }
    )
    return dataset





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






def find_corresponding_orchidee_year(premisefground):
    
    year_background = extract_year(premisefground)
    
    
    
    return year_background


    
    
def number_config(dictionnaries):
    
    
    list_param_list=[]
    
    for key in dictionnaries.keys():
        
        
        for param in dictionnaries[key]:
            #print(dictionnaries[key][param])
            if dictionnaries[key][param][0]=="list":
                list_param_list.append(len(dictionnaries[key][param][1]))
        
    unique_len_list = np.unique(list_param_list)
    

    if len(unique_len_list)==1:
        return unique_len_list[0]
    
    elif len(unique_len_list)==0:
        return 1

        
    
    else:
        sys.exit("List of parameters values for configrations are of different lengths")








def update_param_with_premise_and_orchidee_uncert_withorchideemodel_asparam(dictionnaries,
                                           parameters_database,
                                           gridpoints,
                                           croptype,
                                           size_uncert,
                                          fixed_or_st_ref,
                                          n_points,
                                          n_models):
    
    """  Update the parameters sample with the values from the orchidee grid and the ones collected in the background """
    
    # Efficiencies align with the background db
    dictionnaries["PV_par_distributions"]["panel_efficiency_PV"] = ['unique', [parameters_database["new_efficiency_single"],0.10, 0.40, 0, 0],"."]

    dictionnaries["PV_par_distributions"]["panel_efficiency_PV_single"] = ['unique', [parameters_database["new_efficiency_single"],0.10, 0.40, 0, 0],"."]
    dictionnaries["PV_par_distributions"]["panel_efficiency_PV_multi"] = ['unique', [parameters_database["new_efficiency_multi"],0.10, 0.40, 0, 0],"."]
    dictionnaries["PV_par_distributions"]["panel_efficiency_PV_cis"] = ['unique', [parameters_database["new_efficiency_cis"],0.10, 0.40, 0, 0],"."]
    dictionnaries["PV_par_distributions"]["panel_efficiency_PV_cdte"] = ['unique', [parameters_database["new_efficiency_cdte"],0.10, 0.40, 0, 0],"."]
    
    
   
    init_yield_alfalfa = dictionnaries["Ag_par_distributions"]["init_yield_alfalfa"][1][0]
    init_yield_soy= dictionnaries["Ag_par_distributions"]["init_yield_soy"][1][0]
    init_yield_wheat= dictionnaries["Ag_par_distributions"]["init_yield_wheat"][1][0]
    init_yield_maize= dictionnaries["Ag_par_distributions"]["init_yield_maize"][1][0]
    
    # number of different combinations of gridpoints and simulation models
    

    size_model_gridpoints = len(list(itertools.chain(*gridpoints.loc["water efficiency ref"])))  
    
    # If 1000 points and 3 models : 3000

    
    
    size_list_config = number_config(dictionnaries)

    # total  number of LCAs
    
    
    total_size = size_model_gridpoints * size_list_config * size_uncert
    
    
    # Create sample
    names_param_total, sample_dataframe= sampling_func_lhs_modif(dictionnaries,  
                                 size_uncert,
                                 size_list_config
                                 
                                 )
    
    # Adjusts sample dataframe with  orchidee's outputs
    sample_dataframe = adjustsample_dataframe_withorchideemodel_asparam(sample_dataframe,
                               gridpoints,
                               croptype,
                                init_yield_alfalfa,
                                init_yield_soy,
                                init_yield_wheat,
                                init_yield_maize,
                                fixed_or_st_ref,
                                n_models,
                                n_points)
    
    
    # Put the input sample in a dictionnary with all the sampled values for the input parameters
    
    values_for_datapackages = {}
    
    for dict in dictionnaries:
        
       values_for_datapackages = Merge(values_for_datapackages , dictionnaries[dict])
        
       
    values_for_datapackages = fix_switch_valuedatapackages(values_for_datapackages)
    
    for param in sample_dataframe:
        
        values_for_datapackages[param]={"values":sample_dataframe[param].tolist()}
    

   


    return names_param_total, sample_dataframe,values_for_datapackages,size_model_gridpoints,size_list_config,total_size
    









def update_param_with_premise_and_orchidee_uncert_withorchideemodel_asparam_sens(dictionnaries,
                                           parameters_database,
                                           gridpoints,
                                           croptype,
                                           size_uncert,
                                          fixed_or_st_ref,
                                          n_points,
                                          n_models):
    
    """  Update the parameters sample with the values from the orchidee grid and the ones collected in the background 
    MODIFIED FOR GSA"""
    

    
   
    init_yield_alfalfa = dictionnaries["Ag_par_distributions"]["init_yield_alfalfa"][1][0]
    init_yield_soy= dictionnaries["Ag_par_distributions"]["init_yield_soy"][1][0]
    init_yield_wheat= dictionnaries["Ag_par_distributions"]["init_yield_wheat"][1][0]
    init_yield_maize= dictionnaries["Ag_par_distributions"]["init_yield_maize"][1][0]
    
    # number of different combinations of gridpoints and simulation models
    

    size_model_gridpoints = len(list(itertools.chain(*gridpoints.loc["water efficiency ref"])))  
    
    # If 1000 points and 3 models : 3000
            
    
    #print("n_points",n_points)
    #print("n_models",n_models)
        
    
    # Number of deterministic config '(list)
    
    
    size_list_config = number_config(dictionnaries)

    # total  number of LCAs
    
    
    total_size = size_model_gridpoints * size_list_config * size_uncert
    
    
    # Create sample
    names_param_total, sample_dataframe = sampling_func_lhs_modif(dictionnaries,  
                                 size_uncert,
                                 size_list_config
                                 
                                 )
    
    

    
    # Put the input sample in a dictionnary with all the sampled values for the input parameters
    
    values_for_datapackages = {}
    
    for dict in dictionnaries:
        
       values_for_datapackages = Merge(values_for_datapackages , dictionnaries[dict])
        
       
    values_for_datapackages = fix_switch_valuedatapackages(values_for_datapackages)
    
    for param in sample_dataframe:
        print(param)
        
        values_for_datapackages[param]={"values":sample_dataframe[param].tolist()}
    

   


    return names_param_total, sample_dataframe,values_for_datapackages,size_model_gridpoints,size_list_config,total_size
    


def adjustsample_dataframe_withorchideemodel_asparam(sample_dataframe,
                           gridpoints,
                           croptype,
                            init_yield_alfalfa,
                            init_yield_soy,
                            init_yield_wheat,
                            init_yield_maize,
                            fixed_or_st_ref,
                            n_models,
                            n_points):
    
    """Adjusts sample dataframe with  orchidee's outputs. 
    At the end the different gird points, orchidee models, and LCA parameters combinations follow a determined structure represented in the csv file "Structure sample dataframe et results"""
   
        
    crop_yield_ref_list = list(itertools.chain(*gridpoints.loc["crop yield ref"])) # M = model , P = grid point [M1P1,M2P1...M1P2,M2P2...] 
    
    crop_yield_avs_list = list(itertools.chain(*gridpoints.loc["crop yield avs"]))
    
    if fixed_or_st_ref == "fixed":
        
        electric_yield_ref_list = list(itertools.chain(*gridpoints.loc["electric yield ref fixed"]))
    
    elif fixed_or_st_ref == "st":
        
        electric_yield_ref_list = list(itertools.chain(*gridpoints.loc["electric yield ref st"]))
        
    else :
        sys.exit("Specify the type of PV ref : fixed or st")   
        
    electric_yield_avs_list = list(itertools.chain(*gridpoints.loc["electric yield avs"]))
     
    
    list_crop_yield_ratio_avs_ref =[a/b for a,b in zip(crop_yield_avs_list,crop_yield_ref_list)]
    list_electric_yield_ratio_avs_ref =[a/b for a,b in zip(electric_yield_avs_list,electric_yield_ref_list)]
    
    
    if croptype=="wheat":
        initial_yield = init_yield_wheat
        
    elif croptype=="alfalfa":
        initial_yield = init_yield_alfalfa   
        
    elif croptype=="soy":
        initial_yield = init_yield_soy 
        
    elif croptype=="maize":
        initial_yield = init_yield_maize     
    
        
    list_crop_yield_ref_update = [a/initial_yield for a in crop_yield_ref_list]
    
    nuncerttimesnconfig = sample_dataframe.shape[0]

    # flat_corresponding_sequence_models = list(itertools.chain.from_iterable(corresponding_sequence_models))
    
    # print("flat_corresponding_sequence_models",flat_corresponding_sequence_models)

    #Reshape the dataframe to be able to host orchidees outputs
    
    sample_dataframe = pd.concat([sample_dataframe for i in range(len(crop_yield_ref_list))]).reset_index(drop=True)


    # Reshape the list of orchidee's outputs to match the uncertainty and the config parameters

    electric_yield_ref_list = list(itertools.chain(*[[a]*nuncerttimesnconfig for a in electric_yield_ref_list] ))
    
    list_electric_yield_ratio_avs_ref =list(itertools.chain(*[[a]*nuncerttimesnconfig for a in list_electric_yield_ratio_avs_ref] ))
   
    list_crop_yield_ratio_avs_ref =list(itertools.chain(*[[a]*nuncerttimesnconfig for a in list_crop_yield_ratio_avs_ref] ))
    
    list_crop_yield_ref_update =list(itertools.chain(*[[a]*nuncerttimesnconfig for a in list_crop_yield_ref_update] ))


    

    sample_dataframe["annual_producible_PV"] = electric_yield_ref_list
    
    sample_dataframe["annual_producible_ratio_avs_pv"] = list_electric_yield_ratio_avs_ref
    
    sample_dataframe["crop_yield_ratio_avs_ref"] = list_crop_yield_ratio_avs_ref
   
    sample_dataframe["crop_yield_upd_ref"] = list_crop_yield_ref_update
    
    #here add orchidee's simulation model as parameter
    sequence_models = list(range(int(n_models)))
    
    # print("nmodels",len(sequence_models))
    # print("n_points",n_points)
    
    corresponding_sequence_models = [[a] * nuncerttimesnconfig for a in sequence_models]
    
    corresponding_sequence_models = list(itertools.chain.from_iterable(corresponding_sequence_models))
   
    corresponding_sequence_models = [corresponding_sequence_models for a in range (n_points)]
    
    corresponding_sequence_models = list(itertools.chain.from_iterable(corresponding_sequence_models))
    
    
    
    sample_dataframe["RCM/GCM_model"] = corresponding_sequence_models

    
    return sample_dataframe
   

def name_foreground_from_premise(main_string, sub_string):
    
    return "fground_"+main_string.replace(sub_string, "")











""" RANDOM SAMPLING FOR PARAMETERS"""










def convert_cat_parameters_switch_act(list_input_cat):
    
    """  """
    
    
    uniq = np.unique(list_input_cat).tolist()

    dict_= {}
    for input_ in uniq:
        
        values=[(input_==i)*1 for i in list_input_cat]
        
        dict_[input_]={"values":values}
        
    return dict_ 






def generate_list_from_switch(values_p,lensample):
    
    """ For a categorical parameter associated with switch parameters, draws stochastic values for these switch parameters """
    
    if sum(values_p)!=1:
        print("ERROR sum of probabilities for switch not 1",sum(values_p))
        #sys.exit()
    
    num_rows = len(values_p) # number of categories for the switch parameter, ex wheat, soy, alfalfa
    num_cols = lensample # number of iterations
    
    array = np.zeros((num_rows, num_cols), dtype=int)
    
    # Determine the position of '1' in each column based on probabilities

    for col in range(num_cols):
        
        rand = np.random.random()
        cumulative_prob = 0
        
        for row in range(num_rows):
            
            cumulative_prob += values_p[row]
            if rand < cumulative_prob:
                array[row, col] = 1
                break
 
    # restructure      
    list_of_listvalues_switch=[]
    for row in range(num_rows):     
        
        list_of_listvalues_switch.append(array[row,].tolist())


    return list_of_listvalues_switch




def sampling_func_lhs_modif(dictionaries, size_uncertainty, size_list_config):
    '''Function that returns a Latin Hypercube sample for the input space of 
    parameters in the dictionaries, handling uniform, normal, empirical, and unique distributions.
    '''
    
    names_param_unif = []
    bounds = []
    
    all_unique_param_and_values = []
    all_switch_param = []
    all_withlist_param = []
    all_normal_param = []  # Store normal distribution parameters
    all_empirical_param = []  # Store empirical distribution parameters
    
    list_values_output_switch = []
    names_param_switch = []
    names_param_unique = []
    names_param_withlist = []
    names_param_norm = []
    names_param_empirical = []

    for key in dictionaries:
        dict_param = dictionaries[key]
        dict_param_input = dict_param.copy()
        
        unique_params = []
        switch_params = []
        withlist_params = []
        empirical_params = []
        
        for param in dict_param_input:
            distrib_type = dict_param_input[param][0]

            if distrib_type == 'unique':
                names_param_unique.append(param)
                unique_params.append(param)
                all_unique_param_and_values.append((param, dict_param_input[param][1][0]))

            elif distrib_type == 'switch':
                switch_params.append(param)
                all_switch_param.append(dict_param_input[param])

            elif distrib_type == 'list':
                withlist_params.append(param)
                all_withlist_param.append((param, dict_param_input[param]))
                names_param_withlist.append(param)                    

            elif distrib_type == 'norm':
                # Handle normal distribution parameters (mean, sd)
                mean = dict_param_input[param][1][3]
                sd = dict_param_input[param][1][4]
                all_normal_param.append((param, mean, sd))
                names_param_norm.append(param)

            elif distrib_type == 'empirical':
                # Store empirical distribution data
                empirical_values = np.array(dict_param_input[param][1])
                all_empirical_param.append((param, empirical_values))
                names_param_empirical.append(param)
                empirical_params.append(param)

        # Remove handled parameters from dictionary
        for a in unique_params + switch_params + withlist_params + empirical_params:
            dict_param_input.pop(a)

        # Handle uniform distributions
        for param in dict_param_input:
            distrib = dict_param_input[param][0]
            if distrib == 'unif':
                bounds.append([dict_param_input[param][1][1], dict_param_input[param][1][2]])
                names_param_unif.append(param)

    # --- Latin Hypercube Sampling for Uniform Distributions ---
    if bounds:
        lhs = LatinHypercube(d=len(bounds))
        lhs_samples = lhs.random(n=size_uncertainty)
        sample_array = np.zeros((size_uncertainty, len(names_param_unif)), dtype="float32")
        
        for i, bound in enumerate(bounds):
            lower, upper = bound
            sample_array[:, i] = lower + lhs_samples[:, i] * (upper - lower)

        sample_dataframe = pd.DataFrame(sample_array, columns=names_param_unif)
    else:
        sample_dataframe = pd.DataFrame()

    # --- Add Parameters with Normal Distributions ---
    if all_normal_param:
        lhs_normal = LatinHypercube(d=len(all_normal_param))
        lhs_samples_normal = lhs_normal.random(n=size_uncertainty)
        
        for idx, (param, mean, sd) in enumerate(all_normal_param):
            normal_samples = norm(loc=mean, scale=sd).ppf(lhs_samples_normal[:, idx])
            sample_dataframe[param] = normal_samples

    # --- Add Parameters with Empirical Distributions ---
    for param, empirical_values in all_empirical_param:
        # Option 1: Simple Resampling (Bootstrap)
        sampled_values = np.random.choice(empirical_values, size=size_uncertainty, replace=True)
        
        # Option 2: Kernel Density Estimation (KDE) if needed
        if len(empirical_values) > 30:  # Use KDE only if enough points
            kde = gaussian_kde(empirical_values)
            sampled_values = kde.resample(size_uncertainty)[0]
        
        sample_dataframe[param] = sampled_values

    # --- Add Parameters with Fixed/Unique Values ---
    for param in all_unique_param_and_values:
        sample_dataframe[param[0]] = param[1]

    # --- Add Switch Parameters ---
    for param in all_switch_param:
        values_p = param[2]
        names_options = param[3]
        
        list_values_output_switch = generate_list_from_switch(values_p, size_uncertainty)
        names_param_switch = [name + "_switch" for name in names_options]
        
        for option in range(len(names_param_switch)):
            sample_dataframe[names_param_switch[option]] = list_values_output_switch[option]

    # --- Expand for Multiple Configurations ---
    sample_dataframe = pd.concat([sample_dataframe for _ in range(size_list_config)]).reset_index(drop=True)
    
    # --- Add Parameters with Lists ---
    for param in all_withlist_param:
        list_values = param[1][1]
        flattened_list = extend_parameters_withlists(list_values, size_uncertainty)
        sample_dataframe[param[0]] = flattened_list
        
    # --- Combine All Parameter Names ---
    names_param_total = (names_param_unif + names_param_norm + names_param_unique + 
                         names_param_switch + names_param_withlist + names_param_empirical)
    
    return names_param_total, sample_dataframe












def extend_parameters_withlists(list_values,size_uncertainty):
    
    intermediate = [[a]*size_uncertainty for a in list_values]
    flattened_list = list(itertools.chain(*intermediate))  

    return flattened_list




