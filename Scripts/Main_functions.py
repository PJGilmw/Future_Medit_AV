# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 11:45:17 2024

@author: pjouannais
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 11:54:48 2024

@author: pierre.jouannais


Script containing the main functions to compute the LCAs.

"""

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
import time

import matplotlib.pyplot as plt

import pandas as pd



from util_functions import *


import pickle


import math 


import datetime
import os

import json
import ray


import logging

import inspect

import sys

import _thread

from ray.util import inspect_serializability
import threading










def create_modif_arrays_para(act,
                              meth1,
                              name_db_foreground,
                              name_db_background,
                              name_db_additional_bio,
                              name_db_additional_bio_multi_margi,
                              list_totalindices,
                              dict_funct,
                              values_for_datapackages,
                              numberchunksforpara):
    
    """ Collects the original datapackages from ecoinvent and the foreground database and returns modified datapackages with computed stochastic values in the right positions."""
    
    fu, data_objs, _ = bd.prepare_lca_inputs(
        {act: 1}, method=meth1)  # Here are the original data objects
    
    
    
    # There should not be any duplicates, i.e, any flow which is affected by two functions.
    duplicates = set(x for x in list_totalindices if list_totalindices.count(x) > 1) 
    
    if len(duplicates)!=0:
        print("WARNING DUPLICATES", "\n",
              duplicates)
        sys.exit()
    
    
    
    
    # Due to parallel brightway import, the biosphere and the technosphere are not always in the same positions in the data_objs
    # little trick to make sure it always works:
       
        
    # Quick fix of the db names to ensure correspondance by replacing " " by "_"
    
    
    name_db_background=replace_spaces_with_underscore(name_db_background)
    name_db_foreground=replace_spaces_with_underscore(name_db_foreground)
    name_db_additional_bio=replace_spaces_with_underscore(name_db_additional_bio)
    name_db_additional_bio_multi_margi=replace_spaces_with_underscore(name_db_additional_bio_multi_margi)
    
    
    for data in data_objs:
        
        #print("XXXXXXXXXXX")

        #print(data.metadata["name"])
        #print(name_db_background)
        #print(data.metadata["name"]==name_db_foreground)
       
        #print(name_db_foreground)
        #print(name_db_additional_bio)
        #print(name_db_additional_bio_multi_margi)


       # #print(data.metadata)
        if data.metadata["name"]==name_db_background:
            
            print("BACKGROUND")
            # Technosphere background
    
            array_data_background, dict_ = data.get_resource(
                name_db_background+'_technosphere_matrix.data')
            array_flip_background, dict_ = data.get_resource(
                name_db_background+'_technosphere_matrix.flip')
            array_indice_background, dict_ = data.get_resource(
                name_db_background+'_technosphere_matrix.indices')
    
    
            # Biosphere
            array_data_biospherefromecoinvent, dict_ = data.get_resource(
                name_db_background+'_biosphere_matrix.data')
            array_indice_biospherefromecoinvent, dict_ = data.get_resource(
                name_db_background+'_biosphere_matrix.indices')
    
            
        elif data.metadata["name"]==name_db_foreground:
            
            print("FOREGROUND")

            array_data_foreground, dict_ = data.get_resource(
                name_db_foreground+'_technosphere_matrix.data')
            array_indice_foreground, dict_ = data.get_resource(
                name_db_foreground+'_technosphere_matrix.indices')
            array_flip_foreground, dict_ = data.get_resource(
                name_db_foreground+'_technosphere_matrix.flip')
    
            array_data_foreground_bio, dict_ = data.get_resource(
                name_db_foreground+'_biosphere_matrix.data')
            array_indice_foreground_bio, dict_ = data.get_resource(
                name_db_foreground+'_biosphere_matrix.indices')        
    
        elif data.metadata["name"]==name_db_additional_bio:
            
            print("ADDI BIO")

            array_data_additional_bio, dict_ = data.get_resource(
                name_db_additional_bio+'_technosphere_matrix.data')
            array_indice_additional_bio, dict_ = data.get_resource(
                name_db_additional_bio+'_technosphere_matrix.indices')
            

    
        elif data.metadata["name"]==name_db_additional_bio_multi_margi:
            
            print("ADDI BIO MULTI MARGI")

            array_data_additional_bio_multi_margi, dict_ = data.get_resource(
                name_db_additional_bio_multi_margi+'_technosphere_matrix.data')
            array_indice_additional_bio_multi_margi, dict_ = data.get_resource(
                name_db_additional_bio_multi_margi+'_technosphere_matrix.indices')
            

    
    # list_totalindices calculated in "parameters and functions script"
    
    # indices_strucutures contains all the tuples of indices that will require some modifications.
    
    # They can be in the ecoinvent tehcnosphere, the biosphere or the foreground 
    
    indices_structured = np.array(
        list_totalindices, dtype=array_indice_background.dtype)
    
    
    mask_foreground = np.isin(array_indice_foreground, indices_structured)
    #print(sum(mask_foreground))
    
    
    mask_background = np.isin(array_indice_background, indices_structured)
    #cc=sum(mask_background)
    #print(sum(mask_background))

    
    mask_biosphere_background = np.isin(array_indice_biospherefromecoinvent, indices_structured)
    #ii=sum(mask_biosphere_background)
    #print(sum(mask_biosphere_background))

    
    mask_biosphere_foreground = np.isin(array_indice_foreground_bio, indices_structured)
    #ee=sum(mask_biosphere_foreground)
    #print(sum(mask_biosphere_foreground))

    
    mask_additional_biosphere = np.isin(array_indice_additional_bio, indices_structured)
    #oo=sum(mask_additional_biosphere)
    #print(sum(mask_additional_biosphere))

    
    
    mask_additional_biosphere_multi_margi = np.isin(array_indice_additional_bio_multi_margi, indices_structured)
    #vv=sum(mask_additional_biosphere_multi_margi)
    #print(sum(mask_additional_biosphere_multi_margi))

    
    
    # These arrays contain all the data, indices and flip that will be modified by the functions, either in the foreground or the background
    
    # Technosphere
    
    data_array_total_init_techno = np.concatenate(
        (array_data_foreground[mask_foreground], 
         array_data_background[mask_background],
         array_data_additional_bio[mask_additional_biosphere],
         array_data_additional_bio_multi_margi[mask_additional_biosphere_multi_margi]
    ), axis=0)
    
    
    
    indices_array_total_init_techno = np.concatenate(
        (array_indice_foreground[mask_foreground],
         array_indice_background[mask_background],
         array_indice_additional_bio[mask_additional_biosphere],
         array_indice_additional_bio_multi_margi[mask_additional_biosphere_multi_margi]), axis=0)
    
    flip_array_total_init_techno = np.concatenate(
        (array_flip_foreground[mask_foreground],
         array_flip_background[mask_background],

         
         ), axis=0)
    
    # Biosphere
    
    
    data_array_total_init_bio = np.concatenate(
        (array_data_foreground_bio[mask_biosphere_foreground],
         array_data_biospherefromecoinvent[mask_biosphere_background]
    ), axis=0)
    
    
    
    indices_array_total_init_bio = np.concatenate(
        (array_indice_foreground_bio[mask_biosphere_foreground],
         array_indice_biospherefromecoinvent[mask_biosphere_background]), axis=0)
    
    
    
    
    
    # Divide the dictionnary of stochastic values for the parameters into chunks to be processed in parallel
    
    
    #print("RRR")
    
    ##print(values_for_datapackages)
    #print(numberchunksforpara)
    
    
    
    listof_values_for_datapackages, list_chunk_sizes = divide_dict(values_for_datapackages, numberchunksforpara)

    #print("UUU")

    
    """ Create the modified data arrays """
    
    
    list_arrays_for_datapackages = []
    
    for values_for_datapackages_chunk in listof_values_for_datapackages:
        
        data_array_total_modif_techno = data_array_total_init_techno
        data_array_total_modif_bio = data_array_total_init_bio
        
        
        # going through the functions in dict_func and apply them where needed
        
        list_indices=[]
        for a in dict_funct:
            #print(a)
    
            #print(dict_funct[a]["indices"])
            # build a mask to know here the function applies
            mask_techno = check_positions(indices_array_total_init_techno.tolist(), dict_funct[a]["indices"])
            mask_bio = check_positions(indices_array_total_init_bio.tolist(), dict_funct[a]["indices"])
            #print("mask")

            if sum(mask_bio)==0 and sum(mask_techno)==0: # problem
                print("NOWHERE")
                sys.exit("NOWHERE")
            
            list_indices.append(dict_funct[a]["indices"])
            
            if len(data_array_total_modif_techno)!=0: # if there are some modifications to apply on the technosphere
                
                # the data array is modified by applying the function (vectorized) over the stochasitic inputs in "values_for_datapackages_chunk"
                # Applied whee it should thangs to "map" and the mask. 
                # The function automatically picks the necessary input parameters in the dictionnary "values_for_datapackages_chunk" according to their names.
                
                data_array_total_modif_techno = np.where(mask_techno, dict_funct[a]["func"](
                    data_array_total_modif_techno, values_for_datapackages_chunk), data_array_total_modif_techno)
            
            if len(data_array_total_modif_bio)!=0:  # if there are some modifications to apply on the biosphere
        
                data_array_total_modif_bio = np.where(mask_bio, dict_funct[a]["func"](
                    data_array_total_modif_bio, values_for_datapackages_chunk), data_array_total_modif_bio)
            #print("apply")

        
    
        
        list_arrays_for_datapackages.append([data_array_total_modif_bio,
                             data_array_total_modif_techno,
                             indices_array_total_init_techno,
                             flip_array_total_init_techno,
                             indices_array_total_init_bio])



    return list_arrays_for_datapackages,values_for_datapackages,list_chunk_sizes












    
@ray.remote
def compute_stochastic_lcas_1worker(constant_inputs,
                            array_for_dp,
                            chunk_size):
    
    """Computes stochatic lcas by calling the new parameterized, stochastic datapackages. 
    We need to load the bw objects in the function."""
    
    
    # UNPACK
    
    # print("ZZZZ")

    
    # print("UUUU")
    #print("UU")
    [list_fu,
    list_meth,
    uncert,
    C_matrixes,
    act,
    meth1,
    background_db_name,
    biosphere_db_name,
    foreground_db_name,
    name_project,
    indices_array_fix,
    data_array_fix] = constant_inputs
    
    #print("OO")

    [data_array_total_modif_bio,
    data_array_total_modif_techno,
    indices_array_total_init_techno,
    flip_array_total_init_techno,
    indices_array_total_init_bio] = array_for_dp
    
    
    """ Load project, db, methods etc."""
    
    bd.projects.set_current(name_project)
    
    
    Ecoinvent = bd.Database(background_db_name)
    
    biosphere = bd.Database(biosphere_db_name)
    
    
    foregroundAVS = bd.Database(foreground_db_name)

    # print("UUU")


    fu, data_objs, _ = bd.prepare_lca_inputs(
        {act: 1}, method=meth1)  # Here are the original data objects
        
    
    # Save datapackage Static dp fix
    
    dp_static_fix = bwp.create_datapackage()
    
    
    
    dp_static_fix.add_persistent_vector(
        matrix='technosphere_matrix',
        indices_array=indices_array_fix,
        data_array=data_array_fix,
        name='techno static fix')
    
    
    
    #print("PP")

    # CREATE DATAPACKAGE FROM MATRIXES ARRAYS
    
    
    def create_dp_modif(data_array_total_modif_bio,
                         data_array_total_modif_techno,
                         indices_array_total_init_techno,
                         flip_array_total_init_techno,
                         indices_array_total_init_bio) : 
        """Returns a datapackage with the input arrays"""

       
        dp_modif = bwp.create_datapackage(sequential=True)
        
        # Transpose for the right structure
        data_array_total_modif_bio_T = data_array_total_modif_bio.T
        data_array_total_modif_techno_T = data_array_total_modif_techno.T
        
        
        if len(data_array_total_modif_techno_T)!=0:
            dp_modif.add_persistent_array(
                matrix='technosphere_matrix',
                indices_array=indices_array_total_init_techno,
                data_array=data_array_total_modif_techno_T,
                flip_array=flip_array_total_init_techno,
                name='techno modif')
        
        if len(data_array_total_modif_bio_T)!=0:
        
            dp_modif.add_persistent_array(
                matrix='biosphere_matrix',
                indices_array=indices_array_total_init_bio,
                data_array=data_array_total_modif_bio_T,
                name='bio modif')

        return dp_modif
    


    dp_modif = create_dp_modif(data_array_total_modif_bio,
                          data_array_total_modif_techno,
                          indices_array_total_init_techno,
                          flip_array_total_init_techno,
                          indices_array_total_init_bio)
    
    


    

    lca = bc.LCA(fu, data_objs = data_objs + [dp_static_fix]  + [dp_modif],  use_arrays=True,use_distributions=uncert)
    lca.lci()
    lca.lcia()

    
    time1=time.time()  
    
    
    # Initalize the list of results 
    
    listarray_mc_sample = [np.array([[0]*len(list_fu)]*chunk_size,dtype="float32") for meth in range(len(list_meth))]
    
    
    for it in range(chunk_size):
        
        next(lca)
    
    
        for i in range(0,len(list_fu)):
            demand=list_fu[i][0]
            lca.redo_lcia({demand:1})  # redo with new FU
            
            index_array_method=-1
            
            # technosphere_matrix=lca.technosphere_matrix
            # print(technosphere_matrix.shape)
            # print("AVS_crop_main output",technosphere_matrix[(AVS_elec_main_single.id,AVS_elec_main_single.id)])
            # print("AVS_crop_main_single output",technosphere_matrix[(105282, 105282)])
            # print("AVS_crop_main_single input pv insta",technosphere_matrix[(105264, 105282)])
            # print("AVS_crop_main input pv insta",technosphere_matrix[(105262, 105238)])
                  
            
    
            for m in list_meth:
            
                #print("ok3",m)
    
                index_array_method+=1
            
                
                listarray_mc_sample[index_array_method][it,i]=(C_matrixes[m]*lca.inventory).sum() # This calculates the LCIA
                
    
    time2=time.time()
                
    tot_time = time2-time1 
    #print(tot_time)
    
    list_tables = [pd.DataFrame(listarray_mc_sample[i],columns=[fu_[1] for fu_ in list_fu]) for i in range(len(list_meth))]
    
    
    
    # The last row actually corresponds to the first one
    
    for table_index in range(len(list_tables)):
        #  add the last row as first row
        list_tables[table_index] = pd.concat([pd.DataFrame(list_tables[table_index].iloc[-1]).T, list_tables[table_index]], ignore_index=True)
    
        #  delete the last row 
        list_tables[table_index] = list_tables[table_index].drop(list_tables[table_index].index[-1])
    
    return list_tables


    
    
def compute_stochastic_lcas_para(
                            list_arrays_for_datapackages,
                            list_fu,
                            act,
                            list_meth,
                            uncert,
                            list_chunk_sizes,
                            background_db_name,
                            biosphere_db_name,
                            foreground_db_name,
                            name_project,
                            indices_array_fix,
                            data_array_fix):
    
    """Computes stochastic LCAs by calling the new parameterized, stochastic datapackages"""
    
    # Characterization matrices necessary for computations
    Lca = bc.LCA({act: 1}, list_meth[0])
    Lca.lci()
    Lca.lcia()
    C_matrixes = load_characterization_matrix(list_meth, Lca)

    # Determine the number of chunks based on the number of available CPUs


    # Start parallel Monte Carlo simulation
    start_time = time.time()
  
    

    
    ray.shutdown()

    
    
    
    # Get the absolute path of the current working directory
    current_directory = os.getcwd()


    ray.init()
    
            
            
        

    
        
    
    constant_inputs = ray.put([list_fu,
                                list_meth,
                                uncert,
                                C_matrixes,
                                act,
                                list_meth[0],
                                background_db_name,
                                biosphere_db_name,
                                foreground_db_name,
                                name_project,
                                indices_array_fix,
                                data_array_fix])

    
    

    print("Started parallel computations")
    results_ray = [compute_stochastic_lcas_1worker.remote(constant_inputs,array_for_dp,chunk_size) for array_for_dp,chunk_size in zip(list_arrays_for_datapackages,list_chunk_sizes)]
    results = ray.get(results_ray)
    end_time = time.time()
    
    ray.shutdown()




    # Combine results from all chunks
    combined_results = [np.vstack([result[i] for result in results]) for i in range(len(list_meth))]
    list_tables = [pd.DataFrame(combined_results[i], columns=[fu_[1] for fu_ in list_fu]) for i in range(len(list_meth))]


    print(f"Total computation time: {end_time - start_time} seconds")
    
    
    return list_tables   
    
    
    
    
    
    
    
def divide_dict(dict_total, x):
    
    ###
    """ Tahes the dictionnary containing the stochastic values for each parameter
    and divides into  a list of dictionnaries with chunks of the stochastic samples, to be processed in parallel """
    ###
    
    # Initialize the list of dictionaries to be returned
    dicts_list = [{} for _ in range(x)]

    # Iterate over each variable in the original dictionary
    for var, data in dict_total.items():
        
        #print(var,data)
        values = data["values"]
        n = len(values)
        chunk_size = n // x
        
        
        remainder = n % x
        
        # Split values into x chunks
        chunks = [values[i*chunk_size + min(i, remainder):(i+1)*chunk_size + min(i+1, remainder)] for i in range(x)]
        
        # Assign chunks to the new dictionaries
        for i in range(x):
            dicts_list[i][var] = {"values": chunks[i]}
    
    
    list_chunk_sizes=[len(chunk) for chunk in chunks]

    
    return dicts_list,list_chunk_sizes















# Functions to vectorize the application of the functions over the matrices.

def correct_init_whenreplace(init):
    init_correct_sign = np.sign(init)
    init_correct_sign[init == 0] = 1  
    return init_correct_sign

def get_argument_names(func):
    signature = inspect.signature(func)
    return [param.name for param in signature.parameters.values()]

def applyfunctiontothedictvalues(func, dict_, x_array):
    argument_names = get_argument_names(func)
    parameter_values = [dict_[key]["values"] for key in argument_names if key != "init"]
    array_value = np.array([func(x, *params) for x, *params in zip(x_array, *parameter_values)])
    return array_value

def vectorize_function(func):
    def func_vector(x, dict):
        number_iterations = len(next(iter(dict.values()))["values"])
        x_array = extend_init(x, number_iterations)
        array_values = applyfunctiontothedictvalues(func, dict, x_array)
        return array_values
    func_vector.__name__ = f"{func.__name__}_vector"
    return func_vector

def extend_init(x, number_iterations):
    if not isinstance(x[0], np.ndarray):
        x_array = [x] * number_iterations
    else:
        x_array = [x[0]] * number_iterations
    return np.array(x_array)


def check_positions(listtupples, positions):
    
    return [t in positions for t in listtupples]



def Merge(dict_1, dict_2):
	result = dict_1 | dict_2
	return result



def convert_cat_parameters_switch_act(list_input_cat):
    
    
    
    uniq = np.unique(list_input_cat).tolist()

    dict_= {}
    for input_ in uniq:
        
        values=[(input_==i)*1 for i in list_input_cat]
        
        dict_[input_]={"values":values}
        
    return dict_ 







def replace_spaces_with_underscore(input_string):
    return input_string.replace(" ", "_")
