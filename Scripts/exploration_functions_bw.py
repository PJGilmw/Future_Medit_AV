# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 10:32:59 2024

@author: pjouannais

Utility functions to explore activities

"""

import pandas as pd


def exp_act(activity,techno,bio):
    
    
    dict_techno={"input name":[],
           "input code":[],
           "input location":[],
           "input amount":[],
           "input unit":[]
               }
    
    dict_bio={"input name":[],
           "input code":[],
           "input categorie":[],
           "input amount":[],
           "input unit":[]
           
               }
    
    dict_production={"input name":[],
           "input code":[],
           "input location":[],
           "input unit":[]
               }
    for exc in list(activity.exchanges()):
        
        print(exc["name"])
        
        
        if exc["type"] == "production":
            input_ =  techno.get(exc["input"][1])
            dict_production["input name"].append(activity["name"])
            dict_production["input code"].append(activity["code"])
            dict_production["input location"].append(activity["location"])
            dict_production["input unit"].append(activity["unit"])
            
        
        elif exc["type"] == "technosphere":
            input_ =  techno.get(exc["input"][1])
            dict_techno["input name"].append(input_["name"])
            dict_techno["input code"].append(input_["code"])
            dict_techno["input location"].append(input_["location"])
            dict_techno["input amount"].append(exc["amount"])
            dict_techno["input unit"].append(input_["unit"])
    
    
        elif exc["type"] == "biosphere":
            
            input_ =  bio.get(exc["input"][1])
            dict_bio["input name"].append(input_["name"])
            dict_bio["input code"].append(input_["code"])
            dict_bio["input categorie"].append(input_["categories"])
            dict_bio["input amount"].append(exc["amount"])   
            dict_bio["input unit"].append(input_["unit"])
    
    dataframe_bio= pd.DataFrame.from_dict(dict_bio)
    dataframe_techno= pd.DataFrame.from_dict(dict_techno)
    dataframe_production= pd.DataFrame.from_dict(dict_production)
    
    
    
    def export_dataframes_to_csv(df_bio, df_techno, df_production, filename):
        # Create a list to hold all parts for concatenation
        parts = []
        
        # Add the name and DataFrame for 'production'
        parts.append(pd.DataFrame([["Production"]]))  # Section header
        parts.append(df_production)
        
        # Add the name and DataFrame for 'technosphere'
        parts.append(pd.DataFrame([["Technosphere"]]))  # Section header
        parts.append(df_techno)
        parts.append(pd.DataFrame([[""]]*2))  # Two empty rows
    
    
    
        # Add the name and DataFrame for 'biosphere'
        parts.append(pd.DataFrame([["Biosphere"]]))  # Section header
        parts.append(df_bio)
        parts.append(pd.DataFrame([[""]]*2))  # Two empty rows
    
    
        # Concatenate all parts into a single DataFrame
        export_df = pd.concat(parts, ignore_index=True)
    
        # Export to CSV
        export_df.to_csv(filename, index=False, header=False,sep=",")
        
    filename =  activity["name"]+activity["database"]+activity["location"]+'.csv'  
    export_dataframes_to_csv(dataframe_bio, dataframe_techno, dataframe_production, filename)
    
    print("PRODUCTION \n",dataframe_production)
    print("TECHNO \n",dataframe_techno)
    print("BIO \n",dataframe_bio)

    return [dataframe_production,dataframe_techno,dataframe_bio]
   
    
   