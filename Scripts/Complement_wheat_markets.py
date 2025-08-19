# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 14:48:56 2024

@author: pjouannais

Script which calculates the impact of the necessary global wheat market mixes across all scenarios and years.
Exports the results.
Will be necessary to  process results at the end.



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


from activities_parameters_and_functions import *

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



list(bd.projects)

# All bw projects

dict_res_databases = {"eco-3.10_conseq_prem_remind_SSP5_ndc_remind_short0404":[],
                      "eco-3.10_conseq_prem_remind_SSP5_ndc_remind_long0404":[],
                      "eco-3.10_conseq_prem_image_SSP2_base_image_short0404":[],
                      "eco-3.10_conseq_prem_image_SSP2_base_image_long0404":[],
                      "eco-3.10_conseq_prem_image_SSP2_base_image_0404":[],
                      "eco-3.10_conseq_prem_remind_SSP2_base_remind_long0404":[],
                      "eco-3.10_conseq_prem_remind_SSP2_base_remind_short0404":[],
                      "eco-3.10_conseq_prem_remind_SSP2_NPi_remind_short0404":[],
                      "eco-3.10_conseq_prem_remind_SSP2_NPi_remind_long0404":[],
                      "eco-3.10_conseq_prem_image_SSP1_base_image_2601":[],
                      "eco-3.10_conseq_prem_remind_SSP1_base_remind_2601":[]}




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


    
for key in dict_res_databases:
    
    bd.projects.set_current(key)  # your project
    
    dict_res = {}
    
    print(key)
    for db_name in list(bd.databases):
        print(db_name)

        if "ei_consequential_3.10_model" in db_name:
            print("ok")
            db_prem = bd.Database(db_name)

            try:
                wheat_row = db_prem.get([
                    act for act in db_prem
                    if act["name"] == 'wheat grain production' and act["location"] == "RoW"
                ][0]["code"])

                lca = bc.LCA({wheat_row: 1}, meth1)
                lca.lci()
                lca.lcia()

                dict_res[db_name] = lca.score

            except Exception as e:
                print(f"Error processing {db_name} in project {key}: {e}")
                continue
    
    dict_res_databases[key] = dict_res
    





for key in dict_res_databases:
    
    bd.projects.set_current(key)  # your project
    
    dict_res = {}
    
    print(key)
    for db_name in list(bd.databases):
        print(db_name)

        if "ei_consequential_3.10_model" in db_name:
            print("ok")
            db_prem = bd.Database(db_name)

            try:
                wheat_row = db_prem.get([
                    act for act in db_prem
                    if act["name"] == 'wheat grain production' and act["location"] == "RoW"
                ][0]["code"])


                
                
                FU=[{wheat_row:1}]
                bd.calculation_setups['wheatrow'] = {'inv':FU, 'ia': list_meth}
                mylca = bc.MultiLCA('wheatrow')
               
                dict_res[db_name] =  mylca.results.tolist()[0]

            except Exception as e:
                print(f"Error processing {db_name} in project {key}: {e}")
                continue
    
    dict_res_databases[key] = dict_res
        



# Compute wheat_market in current ecoinvent
name_project=  "eco-3.10_conseq_prem_image_SSP2_base_image_long0404"

bd.projects.set_current(name_project) # your project




ecoinvent_conseq = bd.Database('ecoinvent-3.10-consequential')

wheat_row = ecoinvent_conseq.get([
     act for act in ecoinvent_conseq
     if act["name"] == 'wheat grain production' and act["location"] == "RoW"
 ][0]["code"])


FU=[{wheat_row:1}]
bd.calculation_setups['wheatrow'] = {'inv':FU, 'ia': list_meth}
mylca = bc.MultiLCA('wheatrow')
   
dict_res_databases["Ecoinvent_310"] =  mylca.results.tolist()[0]


    

x = datetime.datetime.now()

month=str(x.month)
day=str(x.day)
microsec=str(x.strftime("%f"))
             


name_file_res ='Impacts_wheat_market'+"_"+month+"_"+day



export_pickle_2(dict_res_databases, name_file_res, "Results")





    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    