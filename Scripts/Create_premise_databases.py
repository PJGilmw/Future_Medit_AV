# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 13:52:25 2024

@author: pjouannais

This script allows creating the premise databases from the ecoinvent 3.10 consequential database and its biosphere
It starts from a basis bw project with ecoinvent 3.10 and its biosphere that is then copied and populated with premise databases.
It creates one project per combination of SSP and IAM and 6 premise databases are created for each project: 3 years [2050, 2070, 2090]] and two types of arguments to identify marginal mixes.




"""


from bw2data.parameters import *



import bw2data as bd
import bw2io as bi

from premise import *

import pandas as pd


import matplotlib.pyplot as plt


from exploration_functions_bw import *




"""Functions """


def list_names_bd_premise(originaldb_name,scenarios,argtype):
    """Creates list of names for the premise databases"""
    
    list_names=  [originaldb_name+"_model"+dic["model"]+'_path'+ dic["pathway"]+'_year'+str(dic["year"])+"_arg"+argtype for dic in scenarios ]
    
    return list_names


def extend_scenario_for_all_years(scenarios,list_years):
    """Creates list of names for the premise databases"""

    scenarios_all_years = []  

    for dic in scenarios:
        for year in list_years:

            dic_new = {"model":dic["model"],"pathway":dic["pathway"],"year":year}
            scenarios_all_years.append(dic_new)
            
    return scenarios_all_years





def create_write_prosp_db(scenarios,
                          source_db,
                          source_version,
                          system_args_short,
                          system_args_long,
                          biosphere_name,
                          originaldb_name):
    
    ndb_short = NewDatabase(
        scenarios = scenarios,
        source_db=source_db,
        source_version=source_version,
        key='tUePmX_S5B8ieZkkM7WUU2CnO8SmShwmAeWK9x2rTFo=',
        system_model="consequential",
        system_args=system_args_short,
        biosphere_name=biosphere_name
    )
    
    ndb_short.update("electricity")
    ndb_short.update("fuels")
    ndb_short.update("biomass")
    ndb_short.update("steel")

    list_names_short = list_names_bd_premise(originaldb_name,scenarios,"short_slope")
    
    
    ndb_short.write_db_to_brightway(name=list_names_short)
    
    
    


    for scenario in ndb_short.scenarios:
        if "cache" in scenario:
            scenario["cache"] = {}
        if "index" in scenario:
            scenario["index"] = {}
    
    ###
    
    ndb_long = NewDatabase(
        scenarios = scenarios,
        source_db=source_db,
        source_version=source_version,
        key='tUePmX_S5B8ieZkkM7WUU2CnO8SmShwmAeWK9x2rTFo=',
        system_model="consequential",
        system_args=system_args_long,
        biosphere_name=biosphere_name
    )
    
    ndb_long.update("electricity")
    ndb_long.update("fuels")
    ndb_long.update("biomass")
    ndb_long.update("steel")

    list_names_long= list_names_bd_premise("ei_consequential_3.10",scenarios,"long_area")
    
    
    ndb_long.write_db_to_brightway(name=list_names_long)




def create_write_prosp_db_short(scenarios,
                          source_db,
                          source_version,
                          system_args_short,
                          system_args_long,
                          biosphere_name,
                          originaldb_name):
    
    ndb_short = NewDatabase(
        scenarios = scenarios,
        source_db=source_db,
        source_version=source_version,
        key='tUePmX_S5B8ieZkkM7WUU2CnO8SmShwmAeWK9x2rTFo=',
        system_model="consequential",
        system_args=system_args_short,
        biosphere_name=biosphere_name
    )
    
    ndb_short.update("electricity")
    ndb_short.update("fuels")
    ndb_short.update("biomass")
    ndb_short.update("steel")

    list_names_short = list_names_bd_premise(originaldb_name,scenarios,"short_slope")
    
    
    ndb_short.write_db_to_brightway(name=list_names_short)
    
    
    


    for scenario in ndb_short.scenarios:
        if "cache" in scenario:
            scenario["cache"] = {}
        if "index" in scenario:
            scenario["index"] = {}
    
    



def create_write_prosp_db_long(scenarios,
                          source_db,
                          source_version,
                          system_args_short,
                          system_args_long,
                          biosphere_name,
                          originaldb_name):
    
   
    ###
    
    ndb_long = NewDatabase(
        scenarios = scenarios,
        source_db=source_db,
        source_version=source_version,
        key='tUePmX_S5B8ieZkkM7WUU2CnO8SmShwmAeWK9x2rTFo=',
        system_model="consequential",
        system_args=system_args_long,
        biosphere_name=biosphere_name
    )
    
    ndb_long.update("electricity")
    ndb_long.update("fuels")
    ndb_long.update("biomass")
    ndb_long.update("steel")

    list_names_long= list_names_bd_premise("ei_consequential_3.10",scenarios,"long_area")
    
    
    ndb_long.write_db_to_brightway(name=list_names_long)









""" Four types of arguments to identify the marginal mix"""



args_image_short_slope = {
    "range time":2,
    "duration":0,  # short-lasting change
    "foresight":False,  #IMAGE # average lead time after chnage in demand
    "lead time":False,   # always false
    "capital replacement rate":True,   
    "measurement": 0,  # slope
    "weighted slope start": 0.75,  # only used for method 3
    "weighted slope end": 1.00   # only used for method 3
}



args_image_long_area = {
    "range time":0,
    "duration":5,  # long-lasting change
    "foresight":False,  #IMAGE # average lead time after chnage in demand
    "lead time":False,   # always false
    "capital replacement rate":True,   
    "measurement": 2,  # area
    "weighted slope start": 0.75,  # only used for method 3
    "weighted slope end": 1.00   # only used for method 3
}


args_remind_short_slope = {
    "range time":2,
    "duration":0,  # short-lasting change
    "foresight":True,  # rewind # no lead time, production just after the change in demand (perfect anticipation)
    "lead time":False,   # always false
    "capital replacement rate":True,   
    "measurement": 0,  # slope
    "weighted slope start": 0.75,  # only used for method 3
    "weighted slope end": 1.00   # only used for method 3
}



args_remind_long_area = {
    "range time":0,
    "duration":5,  # long-lasting change
    "foresight":True,  # rewind # no lead time, production just after the change in demand (perfect anticipation)
    "lead time":False,   # always false
    "capital replacement rate":True,   
    "measurement": 2,  # area
    "weighted slope start": 0.75,  # only used for method 3
    "weighted slope end": 1.00   # only used for method 3
}


# une base de données premise tous les 10 ans


list_years =list(range(2050, 2100, 20))

# list_years = [2030,2050,2090]






""" Create databases"""





#SSP2_Npi remind long

bd.projects.set_current('eco-3.10_conseq_prem_1511') # your project


bd.projects.copy_project('eco-3.10_conseq_prem_remind_SSP2_NPi_remind_long0404')

bd.projects.set_current('eco-3.10_conseq_prem_remind_SSP2_NPi_remind_long0404') # your project

scenarios= [
    {"model": "remind", "pathway":"SSP2-NPi", "year": 2050}]

scenarios_all_years = extend_scenario_for_all_years(scenarios,list_years)


create_write_prosp_db_long(scenarios_all_years,
                          "ecoinvent-3.10-consequential",
                          "3.10",
                          args_remind_short_slope,
                          args_remind_long_area,
                          "ecoinvent-3.10-biosphere",
                          "ei_consequential_3.10"
                          )


#SSP2_Npi remind short

bd.projects.set_current('eco-3.10_conseq_prem_1511') # your project


bd.projects.copy_project('eco-3.10_conseq_prem_remind_SSP2_NPi_remind_short0404')

bd.projects.set_current('eco-3.10_conseq_prem_remind_SSP2_NPi_remind_short0404') # your project

scenarios= [
    {"model": "remind", "pathway":"SSP2-NPi", "year": 2050}]

scenarios_all_years = extend_scenario_for_all_years(scenarios,list_years)


create_write_prosp_db_short(scenarios_all_years,
                          "ecoinvent-3.10-consequential",
                          "3.10",
                          args_remind_short_slope,
                          args_remind_long_area,
                          "ecoinvent-3.10-biosphere",
                          "ei_consequential_3.10"
                          )






#SSP2_base remind long

bd.projects.set_current('eco-3.10_conseq_prem_1511') # your project


bd.projects.copy_project('eco-3.10_conseq_prem_remind_SSP2_base_remind_long0404')

bd.projects.set_current('eco-3.10_conseq_prem_remind_SSP2_base_remind_long0404') # your project

scenarios= [
    {"model": "remind", "pathway":"SSP2-Base", "year": 2050}]

scenarios_all_years = extend_scenario_for_all_years(scenarios,list_years)


create_write_prosp_db_long(scenarios_all_years,
                          "ecoinvent-3.10-consequential",
                          "3.10",
                          args_remind_short_slope,
                          args_remind_long_area,
                          "ecoinvent-3.10-biosphere",
                          "ei_consequential_3.10"
                          )


#SSP2 base remind short

bd.projects.set_current('eco-3.10_conseq_prem_1511') # your project


bd.projects.copy_project('eco-3.10_conseq_prem_remind_SSP2_base_remind_short0404')

bd.projects.set_current('eco-3.10_conseq_prem_remind_SSP2_base_remind_short0404') # your project

scenarios= [
    {"model": "remind", "pathway":"SSP2-Base", "year": 2050}]

scenarios_all_years = extend_scenario_for_all_years(scenarios,list_years)


create_write_prosp_db_short(scenarios_all_years,
                          "ecoinvent-3.10-consequential",
                          "3.10",
                          args_remind_short_slope,
                          args_remind_long_area,
                          "ecoinvent-3.10-biosphere",
                          "ei_consequential_3.10"
                          )









#SSP2_base image long


bd.projects.set_current('eco-3.10_conseq_prem_1511') # your project


bd.projects.copy_project('eco-3.10_conseq_prem_image_SSP2_base_image_long0404')

bd.projects.set_current('eco-3.10_conseq_prem_image_SSP2_base_image_long0404') # your project

scenarios= [
    {"model": "image", "pathway":"SSP2-Base", "year": 2050}]

scenarios_all_years = extend_scenario_for_all_years(scenarios,list_years)


create_write_prosp_db_long(scenarios_all_years,
                          "ecoinvent-3.10-consequential",
                          "3.10",
                          args_remind_short_slope,
                          args_remind_long_area,
                          "ecoinvent-3.10-biosphere",
                          "ei_consequential_3.10"
                          )


#SSP2 base image short

bd.projects.set_current('eco-3.10_conseq_prem_1511') # your project


bd.projects.copy_project('eco-3.10_conseq_prem_image_SSP2_base_image_short0404')

bd.projects.set_current('eco-3.10_conseq_prem_image_SSP2_base_image_short0404') # your project

scenarios= [
    {"model": "image", "pathway":"SSP2-Base", "year": 2050}]

scenarios_all_years = extend_scenario_for_all_years(scenarios,list_years)


create_write_prosp_db_short(scenarios_all_years,
                          "ecoinvent-3.10-consequential",
                          "3.10",
                          args_remind_short_slope,
                          args_remind_long_area,
                          "ecoinvent-3.10-biosphere",
                          "ei_consequential_3.10"
                          )






#SSP5_NDC remind 







#SSP5_NDC remind long


bd.projects.set_current('eco-3.10_conseq_prem_1511') # your project


bd.projects.copy_project('eco-3.10_conseq_prem_remind_SSP5_ndc_remind_long0404')

bd.projects.set_current('eco-3.10_conseq_prem_remind_SSP5_ndc_remind_long0404') # your project

scenarios= [
    {"model": "remind", "pathway":"SSP5-NDC", "year": 2050}]

scenarios_all_years = extend_scenario_for_all_years(scenarios,list_years)


create_write_prosp_db_long(scenarios_all_years,
                          "ecoinvent-3.10-consequential",
                          "3.10",
                          args_remind_short_slope,
                          args_remind_long_area,
                          "ecoinvent-3.10-biosphere",
                          "ei_consequential_3.10"
                          )


#SSP5_NDC remind short

bd.projects.set_current('eco-3.10_conseq_prem_1511') # your project


bd.projects.copy_project('eco-3.10_conseq_prem_remind_SSP5_ndc_remind_short0404')

bd.projects.set_current('eco-3.10_conseq_prem_remind_SSP5_ndc_remind_short0404') # your project


scenarios= [
    {"model": "remind", "pathway":"SSP5-NDC", "year": 2050}]


scenarios_all_years = extend_scenario_for_all_years(scenarios,list_years)


create_write_prosp_db_short(scenarios_all_years,
                          "ecoinvent-3.10-consequential",
                          "3.10",
                          args_remind_short_slope,
                          args_remind_long_area,
                          "ecoinvent-3.10-biosphere",
                          "ei_consequential_3.10"
                          )





# #SSP1_Base image


bd.projects.set_current('eco-3.10_conseq_prem_1511') # your project

bd.projects.copy_project('eco-3.10_conseq_prem_image_SSP1_base_image_2601')

bd.projects.set_current('eco-3.10_conseq_prem_image_SSP1_base_image_2601') # your project


scenarios= [
    {"model": "image", "pathway":"SSP1-Base", "year": 2040}]

scenarios_all_years = extend_scenario_for_all_years(scenarios,list_years)


create_write_prosp_db(scenarios_all_years,
                          "ecoinvent-3.10-consequential",
                          "3.10",
                          args_image_short_slope,
                          args_image_long_area,
                          "ecoinvent-3.10-biosphere",
                          "ei_consequential_3.10"
                          )









# # #SSP1_Base remind









bd.projects.set_current('eco-3.10_conseq_prem_1511') # your project

bd.projects.copy_project('eco-3.10_conseq_prem_remind_SSP1_base_remind_2601')

bd.projects.set_current('eco-3.10_conseq_prem_remind_SSP1_base_remind_2601') # your project


scenarios= [
    {"model": "remind", "pathway":"SSP1-Base", "year": 2040}]


scenarios_all_years = extend_scenario_for_all_years(scenarios,list_years)


create_write_prosp_db(scenarios_all_years,
                          "ecoinvent-3.10-consequential",
                          "3.10",
                          args_remind_short_slope,
                          args_remind_long_area,
                          "ecoinvent-3.10-biosphere",
                          "ei_consequential_3.10"
                          )

















