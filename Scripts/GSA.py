# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 14:57:33 2024

@author: pjouannais

Computes GSA indexes from the model's outputs

"""

import numpy as np
from SALib.sample import saltelli
from SALib.analyze import sobol
from util_functions import *
import matplotlib.pyplot as plt

from SALib.analyze import delta


# Replace with the corresponding file
res2 = importpickle("../Results/Server/Sens/Resmedi_4_4_364298size=sensii_uniquelifetime.pkl") 


res_uniquelifetime = res2["ei_consequential_3.10_modelremind_pathSSP1-Base_year2070_argshort_slope"]
results_uniquelifetime= res_uniquelifetime[1]  # For all impact categories
inputs_uniquelifetime=res_uniquelifetime[0]

# remove fixed parameters

# Drop columns where there is only one unique value

inputs_uniquelifetime_clean = inputs_uniquelifetime.loc[:, inputs_uniquelifetime.nunique() > 1]


for df in results_uniquelifetime:
        
    df["dif_avs_conv_elec"] = (df["AVS_elec_main_single"] - df["PV_ref_single"])/df["PV_ref_single"]
    df["dif_avs_conv_crop"] = (df["AVS_crop_main_single"] - df["wheat_fr_ref"])/df["wheat_fr_ref"]
   
    
    



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




"""Unique lifetime"""

dict_res_sensi_uniquelifetime = {}

for meth_index in range(len(list_meth)):
    
    res=results_uniquelifetime[meth_index]
    dict_res_meth = {}
    for output in list(res.columns):
        
        output_tostudy = np.array(res[output])
        
        # Create a minimal problem structure
        problem = {
            'num_vars': inputs_uniquelifetime_clean.shape[1],  # Number of variables
            'names': list(inputs_uniquelifetime_clean.columns)  # param names
        }
        
        Si = delta.analyze(problem, np.array(inputs_uniquelifetime_clean), output_tostudy, print_to_console=True)
        
        dict_res_meth[output]=Si
        
    dict_res_sensi_uniquelifetime[list_meth[meth_index][-1]] = dict_res_meth
        
        

x = datetime.datetime.now()

month=str(x.month)
day=str(x.day)
microsec=str(x.strftime("%f"))
             


name_file_res ='Sensi'+"_"+month+"_"+day+"_"+microsec+"uniquelifetime"



export_pickle_2(dict_res_sensi_uniquelifetime, name_file_res, "Results/Sens")
        



        

