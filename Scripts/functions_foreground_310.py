# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 13:04:02 2024

@author: pjouannais

ONLY for keeping ecoinvent 3.0 as background 

Contains functions that create the foreground databases corresponding to specific backgrounds. 


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
import sys




 


def create_foreground_db(name_foreground,
                         name_background,
                         name_background_2,
                         name_biosphere,
                         name_additional_biosphere,
                         exists_already):
    
    
    if not exists_already:
        print("CREATE NEW FOREGROUND")
    
            
        
        foregroundAVS = bd.Database(name_foreground)
        background_db = bd.Database(name_background)
        
        background_db_2 = bd.Database(name_background_2)
        
        
        biosphere_db = bd.Database(name_biosphere)
        additional_biosphere_db = bd.Database(name_additional_biosphere)
        
        foregroundAVS.write({
            (name_foreground, "AVS_crop_main"): {  
                'name': 'AVS_crop_main',
                'unit': 'kg',
                'exchanges': [{
                        'input': (name_foreground, 'AVS_crop_main'),
                        'amount': 1,
                        'unit': 'kg',
                        'type': 'production'}]},
            
            (name_foreground, "AVS_elec_main"): {  
                'name': 'AVS_elec_main',
                'unit': 'kWh',
                'exchanges': [{
                        'input': (name_foreground, 'AVS_elec_main'),
                        'amount': 1,
                        'unit': 'kWh',
                        'type': 'production'}]},
            
            (name_foreground, "PV_ref"): {  
                'name': 'PV_ref',
                'unit': 'kWh',
                'exchanges': [{
                        'input': (name_foreground, 'PV_ref'),
                        'amount': 1,
                        'unit': 'kWh',
                        'type': 'production'}]
                },   
            
            
            (name_foreground, "LUCmarket_AVS"): {  
                'name': 'LUCmarket_AVS',
                'unit': 'ha.y',
                'exchanges': [{
                        'input': (name_foreground, 'LUCmarket_AVS'),
                        'amount': 1,
                        'unit': 'ha.y',
                        'type': 'production'}]
                },
            
            (name_foreground, "LUCmarket_PVref"): {  
                'name': 'LUCmarket_PVref',
                'unit': 'ha.y',
                'exchanges': [{
                        'input': (name_foreground, 'LUCmarket_PVref'),
                        'amount': 1,
                        'unit': 'ha.y',
                        'type': 'production'}]
                },
            
            (name_foreground, "LUCmarket_cropref"): {  
                'name': 'LUCmarket_cropref',
                'unit': 'ha.y',
                'exchanges': [{
                        'input': (name_foreground, 'LUCmarket_cropref'),
                        'amount': 1,
                        'unit': 'ha.y',
                        'type': 'production'}]
                },
            
            (name_foreground, "iluc"): {  
                'name': 'iluc',
                'unit': 'ha.y',
                'exchanges': [{
                        'input': (name_foreground, 'iluc'),
                        'amount': 1,
                        'unit': 'ha.y',
                        'type': 'production'}]
                },
            
            (name_foreground, "c_soil_accu"): { 
                'name': 'c_soil_accu',
                'unit': 'kg C',
                'exchanges': [{
                        'input': (name_foreground, 'c_soil_accu'),
                        'amount': 1,
                        'unit': 'ha.y',
                        'type': 'production'}]
                }})
        
            
            
        
        # Collects previously creates activities
        
        AVS_elec_main = foregroundAVS.get("AVS_elec_main")
        AVS_crop_main = foregroundAVS.get("AVS_crop_main")
        
        PV_ref =foregroundAVS.get("PV_ref")
        
        LUCmarket_PVref = foregroundAVS.get("LUCmarket_PVref")
        LUCmarket_cropref = foregroundAVS.get("LUCmarket_cropref")
        LUCmarket_AVS = foregroundAVS.get("LUCmarket_AVS")
        
        iluc = foregroundAVS.get("iluc")
        
        c_soil_accu = foregroundAVS.get("c_soil_accu")
    
    
    
    
    
    
    
        
        
        # Collects the background_db crop activities
        # aa= background_db.random()
        
        wheat_fr= background_db.get([act for act in background_db if act["name"] =='wheat grain production' and act["location"] == "FR"][0]["code"])
        
        # for exc in list(wheat_fr.exchanges()):
        #     print(exc)
        # wheat_fr = background_db.get(
        #     "98ad08a169bac30f68833a6261d73e73")     # Wheat grain fr
        
        
        
            
        # soy_ch = background_db.get(
        #     "c97395c63e38cf03e87f0e3763b1a9b7")     # soy  ch
        
        
        
        soy_ch= background_db.get([act for act in background_db if act["name"] =='soybean production' and act["location"] == 'CH'][0]["code"])
        
        
        # alfalfa_ch = background_db.get(
        #     "971ad0c601b5bb02b7a16450c48b28d5")     # alfalfa grain ch
        
        
        # alfalfa_ch["name"]
        # alfalfa_ch["location"]
        
        alfalfa_ch= background_db.get([act for act in background_db if act["name"] =='alfalfa-grass mixture production, Swiss integrated production' and act["location"] == 'CH'][0]["code"])
        
        
        
        maize_ch = background_db.get([act for act in background_db if "maize grain production, Swiss integrated production" in act["name"] and act["location"]== "CH"][0]["code"])

        def delete_transfo_occu(act):
            
            """ Deletes the transformation and the occupation exhhanges of an activity, and returns the original transformation flow """
            
            for exc in list(act.exchanges()):
                if exc["type"]=="biosphere_db_db":
                    
                    actinput=biosphere_db.get(exc["input"][1])
        
                    if "Transformation, to" in actinput["name"]:
                        
                        transfo_amount = exc["amount"]
                        
                        exc.delete()
                        
                    elif "Transformation, from" in actinput["name"]:
                                    
                        exc.delete()   
                        
                    elif "Occupation" in actinput["name"]:
                                    
                        exc.delete() 
            act.save()     
               
            return act,transfo_amount
        
        
        def collect_transfo(act):
            
            """  Collects the transformation amount of an act """
            
            for exc in list(act.exchanges()):
                if exc["type"]=="biosphere":
                    
                    actinput=biosphere_db.get(exc["input"][1])
        
                    if "Transformation, to" in actinput["name"]:
                        
                        transfo_amount = exc["amount"]
                        
                    
            return transfo_amount
        
        
        def rescale_to_ha(act,prodrescale,transfo_amount):
            
            """ Rescales a farming activity to 1 ha. """
            
            for exc in list(act.exchanges()):
                if exc["type"]!="production" or prodrescale:
                    exc["amount"] = exc["amount"]*10000/transfo_amount          
                    exc.save()
                    
            act.save()  
            
            return act
        
        
        def prepare_act_agri(original_act):
            
            """ Creates the crop activitites ready to be used in the AVS model"""
            
            act_ref = original_act.copy() # The conv zctivity
            
            act_AVS_elec_main = original_act.copy() # the crop virtual activity for when elec is the main product of AVS
            act_AVS_crop_main = original_act.copy() # the crop virtual activity for when crop is the main product of AVS
            
        
            act_ref_ori_code = act_ref["code"]
        
            # Collect yield (via transfo)
            
            transfo_amount = collect_transfo(act_AVS_crop_main)
            
            # The AVS agri act produces 1 Unit of virtural hectare.
            # the crop production becomes an output (substitution)
            
            for exc in list(act_AVS_elec_main.exchanges()):
                if exc["type"]=="production":
                    prod_amount=exc["amount"]
                    exc["amount"]=1 
                    exc["unit"] = "virtual hectare unit"
                    exc.save()
                    act_AVS_elec_main.new_exchange(amount=-prod_amount, input=act_ref,type="technosphere").save()
            
        
            # The AVS crop_main act produces 1 Unit of virtural hectare.
            # No substitution
            for exc in list(act_AVS_crop_main.exchanges()):
                if exc["type"]=="production":
                    prod_amount=exc["amount"]
                    exc["amount"]=1 
                    exc["unit"] = "virtual hectare unit"
                    exc.save()    
            
            
            act_AVS_elec_main["unit"] = "virtual hectare unit"
            act_AVS_elec_main.save()
            
            act_AVS_crop_main["unit"] = "virtual hectare unit"
            act_AVS_crop_main.save()
            
            
            
            
            # Rescale to 1 ha
            
            act_ref = rescale_to_ha(act_ref,True,transfo_amount)
            
            act_AVS_crop_main = rescale_to_ha(act_AVS_crop_main,False,transfo_amount)
            
        
          
            # This also rescales  act_AVS_elec_main to ha
            for exc in list(act_AVS_elec_main.exchanges()):
                if exc["type"]!="production":
                    for exc2 in list(act_ref.exchanges()):
                        if exc["input"]==exc2["input"]:
                            if exc2["input"][1]==act_ref_ori_code:
                                #print(exc2["input"])
                                exc["amount"] = - exc2["amount"]         
                                exc.save()
                                
                            else:
                                #print(exc2["input"])
                                exc["amount"] = exc2["amount"]         
                                exc.save()    
            act_AVS_elec_main.save()  
            
            
            
            return act_AVS_elec_main,act_ref,act_AVS_crop_main
            
        # Creates the crop activities with the function
        
        # Wheat
        
        # The "virtual" act for AVS with elec main, conv crop act, "virtual" for act for AVS with elec main
        wheat_fr_AVS_elec_main,wheat_fr_ref,wheat_fr_AVS_crop_main = prepare_act_agri(wheat_fr)  
        
        wheat_fr_AVS_elec_main["database"] = name_foreground
        wheat_fr_AVS_elec_main["name"] = 'wheat_fr_AVS_elec_main'
        wheat_fr_AVS_elec_main["code"] = 'wheat_fr_AVS_elec_main'
        wheat_fr_AVS_elec_main.save()
        
        wheat_fr_ref["database"] = name_foreground
        wheat_fr_ref["name"] = 'wheat_fr_ref'
        wheat_fr_ref["code"] = 'wheat_fr_ref'
        wheat_fr_ref.save()
        
        wheat_fr_AVS_crop_main["database"] = name_foreground
        wheat_fr_AVS_crop_main["name"] = 'wheat_fr_AVS_crop_main'
        wheat_fr_AVS_crop_main["code"] = 'wheat_fr_AVS_crop_main'
        wheat_fr_AVS_crop_main.save()
        
        
        
        # Soy
        
        soy_ch_AVS_elec_main,soy_ch_ref,soy_ch_AVS_crop_main = prepare_act_agri(soy_ch)
        
        soy_ch_AVS_elec_main["database"] = name_foreground
        soy_ch_AVS_elec_main["name"] = 'soy_ch_AVS_elec_main'
        soy_ch_AVS_elec_main["code"] = 'soy_ch_AVS_elec_main'
        soy_ch_AVS_elec_main.save()
        
        soy_ch_ref["database"] = name_foreground
        soy_ch_ref["name"] = 'soy_ch_ref'
        soy_ch_ref["code"] = 'soy_ch_ref'
        soy_ch_ref.save()
        
        soy_ch_AVS_crop_main["database"] = name_foreground
        soy_ch_AVS_crop_main["name"] = 'soy_ch_AVS_crop_main'
        soy_ch_AVS_crop_main["code"] = 'soy_ch_AVS_crop_main'
        soy_ch_AVS_crop_main.save()
        
        
        
        # Alfalfa
        
        alfalfa_ch_AVS_elec_main,alfalfa_ch_ref,alfalfa_ch_AVS_crop_main = prepare_act_agri(alfalfa_ch)
        
        alfalfa_ch_AVS_elec_main["database"] = name_foreground
        alfalfa_ch_AVS_elec_main["name"] = 'alfalfa_ch_AVS_elec_main'
        alfalfa_ch_AVS_elec_main["code"] = 'alfalfa_ch_AVS_elec_main'
        alfalfa_ch_AVS_elec_main.save()
        
        alfalfa_ch_ref["database"] = name_foreground
        alfalfa_ch_ref["name"] = 'alfalfa_ch_ref'
        alfalfa_ch_ref["code"] = 'alfalfa_ch_ref'
        alfalfa_ch_ref.save()
        
        alfalfa_ch_AVS_crop_main["database"] = name_foreground
        alfalfa_ch_AVS_crop_main["name"] = 'alfalfa_ch_AVS_crop_main'
        alfalfa_ch_AVS_crop_main["code"] = 'alfalfa_ch_AVS_crop_main'
        alfalfa_ch_AVS_crop_main.save()
        
        # Maize
        
        maize_ch_AVS_elec_main,maize_ch_ref,maize_ch_AVS_crop_main = prepare_act_agri(maize_ch)
        
        maize_ch_AVS_elec_main["database"] = name_foreground
        maize_ch_AVS_elec_main["name"] = 'maize_ch_AVS_elec_main'
        maize_ch_AVS_elec_main["code"] = 'maize_ch_AVS_elec_main'
        maize_ch_AVS_elec_main.save()
        
        maize_ch_ref["database"] = name_foreground
        maize_ch_ref["name"] = 'maize_ch_ref'
        maize_ch_ref["code"] = 'maize_ch_ref'
        maize_ch_ref.save()
        
        maize_ch_AVS_crop_main["database"] = name_foreground
        maize_ch_AVS_crop_main["name"] = 'maize_ch_AVS_crop_main'
        maize_ch_AVS_crop_main["code"] = 'maize_ch_AVS_crop_main'
        maize_ch_AVS_crop_main.save()        
        
        
        

        # Our AVS now needs to consider the associated production of crop as a coproduct.
        # 1 virtual unit of the three crops productions. This is the same as putting all the inputs and emissions of the crops to the AVS, and substuting with the crop output.
        
        
        AVS_elec_main.new_exchange(amount=1, input=wheat_fr_AVS_elec_main, type="technosphere").save()
        AVS_elec_main.new_exchange(amount=1, input=alfalfa_ch_AVS_elec_main, type="technosphere").save()
        AVS_elec_main.new_exchange(amount=1, input=soy_ch_AVS_elec_main, type="technosphere").save()
        AVS_elec_main.new_exchange(amount=1, input=maize_ch_AVS_elec_main, type="technosphere").save()
        
        
        
        AVS_crop_main.new_exchange(amount=1, input=wheat_fr_AVS_crop_main, type="technosphere").save()
        AVS_crop_main.new_exchange(amount=1, input=alfalfa_ch_AVS_crop_main, type="technosphere").save()
        AVS_crop_main.new_exchange(amount=1, input=soy_ch_AVS_crop_main, type="technosphere").save()
        AVS_crop_main.new_exchange(amount=1, input=maize_ch_AVS_crop_main, type="technosphere").save()
        
        
        
        # for exc in list(AVS_elec_main.exchanges()):
        #     print(exc)
            
           
            
        
        
        """ ilUC """
        
        # We also create virtual activities for iLUC. One each for AVS, Pvref and Wheatref
        
        
        
        
        
        
        
        # First, add an exhange of carbon  dioxide  to the virtual carbon accumulation act in the foreground.
        
        
        
        # Exchange: 1.42397953472488 kilogram 'Carbon dioxide, in air' (kilogram, None, ('natural resource', 'in air')) to 'wheat grain production' (kilogram, FR, None)>
        # Carbon dioxide, in air
        # cc6a1abb-b123-4ca6-8f16-38209df609be
        # Carbon_dioxide_to_soil_biomass_stock = biosphere_db.get('375bc95e-6596-4aa1-9716-80ff51b9da77')
        # Carbon_dioxide_to_soil_biomass_stock["name"]
        # Carbon_dioxide_to_soil_biomass_stock["categories"]
        
        Carbon_dioxide_to_soil_biomass_stock= biosphere_db.get([act for act in biosphere_db if act["name"] =='Carbon dioxide, to soil or biomass stock' and act["categories"] == ('soil',)][0]["code"])
        
        
        
        
        c_soil_accu.new_exchange(amount=44/12, input=Carbon_dioxide_to_soil_biomass_stock, type="biosphere").save()
        c_soil_accu.save()
        
        
        # Adds the virtual carbon accumulation activity to the different systems
        
        wheat_fr_ref.new_exchange(amount=1, input=c_soil_accu, type="technosphere").save()
        soy_ch_ref.new_exchange(amount=1, input=c_soil_accu, type="technosphere").save()
        alfalfa_ch_ref.new_exchange(amount=1, input=c_soil_accu, type="technosphere").save()
        maize_ch_ref.new_exchange(amount=1, input=c_soil_accu, type="technosphere").save()
        
        PV_ref.new_exchange(amount=1, input=c_soil_accu, type="technosphere").save()
        
        
        AVS_elec_main.new_exchange(amount=1, input=c_soil_accu, type="technosphere").save()
        AVS_crop_main.new_exchange(amount=1, input=c_soil_accu, type="technosphere").save()
    
        # The accelerated impact category has a biosphere output of a new emission flow "iluc CO2eq" which is already characterized.
        # to avoid double accounting, we create this new susbstance in a new biosphere database. 
        # We also create a new lcia method which is Recipe GWP100 modified to include a cf of 1 kg CO2 eq for our new flow.
        # Thus iLUC emissions are only characterized into GWP impact as we don't have access to the full model.
        
        # ('Carbon dioxide, from soil or biomass stock' (kilogram, None, ('air', 'low population density, long-term')),
            
        
        
        # We add this emission to our virtual act.
        iLUCspecificCO2eq=additional_biosphere_db.get("iLUCspecificCO2eq")
        
        iluc.new_exchange(
            amount=1, input=iLUCspecificCO2eq, type="biosphere").save()
        
        
        # Add virtual exchanges of iluc to the luc activities 
        LUCmarket_cropref.new_exchange(amount=1, input=iluc, type="technosphere").save()
        
        LUCmarket_PVref.new_exchange(amount=1, input=iluc, type="technosphere").save()
        
        LUCmarket_AVS.new_exchange(amount=1, input=iluc, type="technosphere").save()
        
        
        #  Add eLUC market exchanges to the AVS, PV prod and Crop ref
        
        wheat_fr_ref.new_exchange(amount=1, input=LUCmarket_cropref, type="technosphere").save()
        soy_ch_ref.new_exchange(amount=1, input=LUCmarket_cropref, type="technosphere").save()
        alfalfa_ch_ref.new_exchange(amount=1, input=LUCmarket_cropref, type="technosphere").save()
        maize_ch_ref.new_exchange(amount=1, input=LUCmarket_cropref, type="technosphere").save()
        
        AVS_elec_main.new_exchange(amount=1, input=LUCmarket_AVS, type="technosphere").save()
        
        AVS_crop_main.new_exchange(amount=1, input=LUCmarket_AVS, type="technosphere").save()
        
        PV_ref.new_exchange(amount=1, input=LUCmarket_PVref,type="technosphere").save()
        
        
        # AVS_crop_main substitutes either marginal or PV elec. To set up the database, we add an echnage of the 2 options.
        # One of them will be set to 0 for each MC iteration.
        
        
        # elec_marginal_fr = background_db.get("a3b594fa27de840e85cb577a3d63d11a")
        
        
        
        elec_marginal_fr= background_db.get([act for act in background_db if act["name"] =='market for electricity, high voltage' and act["location"] == "FR"][0]["code"])
        
        elec_marginal_fr_current= background_db_2.get([act for act in background_db_2 if act["name"] =='market for electricity, high voltage' and act["location"] == "FR"][0]["code"])
        elec_marginal_it_current= background_db_2.get([act for act in background_db_2 if act["name"] =='market for electricity, high voltage' and act["location"] == "IT"][0]["code"])
        elec_marginal_es_current= background_db_2.get([act for act in background_db_2 if act["name"] =='market for electricity, high voltage' and act["location"] == "ES"][0]["code"])
        
        elec_marginal_fr_copy = elec_marginal_fr.copy()
        
        
        elec_marginal_fr_copy["database"] = name_foreground
        elec_marginal_fr_copy["name"] = 'elec_marginal_fr_copy'
        elec_marginal_fr_copy["code"] = 'elec_marginal_fr_copy'
        elec_marginal_fr_copy.save()
        
        
        elec_marginal_fr_current_copy = elec_marginal_fr_current.copy()
        
        
        elec_marginal_fr_current_copy["database"] = name_foreground
        elec_marginal_fr_current_copy["name"] = 'elec_marginal_fr_current_copy'
        elec_marginal_fr_current_copy["code"] = 'elec_marginal_fr_current_copy'
        elec_marginal_fr_current_copy.save()
        
        
        elec_marginal_it_current_copy = elec_marginal_it_current.copy()
        
        
        elec_marginal_it_current_copy["database"] = name_foreground
        elec_marginal_it_current_copy["name"] = 'elec_marginal_it_current_copy'
        elec_marginal_it_current_copy["code"] = 'elec_marginal_it_current_copy'
        elec_marginal_it_current_copy.save()
        
        
        elec_marginal_es_current_copy = elec_marginal_es_current.copy()
        
        
        elec_marginal_es_current_copy["database"] = name_foreground
        elec_marginal_es_current_copy["name"] = 'elec_marginal_es_current_copy'
        elec_marginal_es_current_copy["code"] = 'elec_marginal_es_current_copy'
        elec_marginal_es_current_copy.save()
        
        
        

    
    
    else:
        print("FOREGROUND ALREADY EXISTS")

        foregroundAVS = bd.Database(name_foreground)

        
    return foregroundAVS
    
    
    
    
    
    
    
    
    


def finalize_foreground_db(name_foreground,
                           name_background,
                           name_biosphere,
                           name_additional_biosphere,
                           exists_already):
    
    foregroundAVS = bd.Database(name_foreground)
    background_db = bd.Database(name_background)
    biosphere_db = bd.Database(name_biosphere)
    additional_biosphere_db = bd.Database(name_additional_biosphere)
    
    if not exists_already:
        print("FINALIZE NEW FOREGROUND")

        photovoltaicslantedroofinstallation3kWpsingleSipanelmountedonroof_premise= background_db.get([act for act in background_db if act["name"] =='photovoltaic slanted-roof installation, 3kWp, single-Si, panel, mounted, on roof'and act["location"] == 'CH'][0]["code"])
        
        photovoltaicslantedroofinstallation3kWpmultiSipanelmountedonroof_premise= background_db.get([act for act in background_db if  act["name"] =='photovoltaic slanted-roof installation, 3kWp, multi-Si, panel, mounted, on roof' and act["location"] =="CH"][0]["code"])
        
        photovoltaicslantedroofinstallation3kWpCISpanelmountedonroof_premise= background_db.get([act for act in background_db if act["name"] =='photovoltaic slanted-roof installation, 3kWp, CIS, panel, mounted, on roof' and act["location"] == 'CH'][0]["code"])
      
        if background_db.name != "ecoinvent-3.10-consequential":
            photovoltaicslantedroofinstallation3kWpCdtepanelmountedonroof_premise= background_db.get([act for act in background_db if act["name"] =='photovoltaic slanted-roof installation, 3 kWp, CdTe, panel, mounted, on roof' and act["location"] == 'CH'][0]["code"])

        else:
            photovoltaicslantedroofinstallation3kWpCdtepanelmountedonroof_premise= background_db.get([act for act in background_db if "CdTe" in act["name"] and "photovoltaic slanted-roof installation" in act["name"] and act["location"] == 'CH'][0]["code"])

        # Collect original surface of panel for 3kWp, single
        for exc in list(photovoltaicslantedroofinstallation3kWpsingleSipanelmountedonroof_premise.exchanges()):
                print("AAAA")
                print(exc)
                print(exc["name"])
                if exc["name"]=='photovoltaic panel, single-Si wafer':
                    original_surface_panel_single = exc["amount"]
            
      
            
        # Collect original surface of panel for 3kWp, multi
        for exc in list(photovoltaicslantedroofinstallation3kWpmultiSipanelmountedonroof_premise.exchanges()):
                print("BBBB")
                print(exc)
                print(exc["name"])
                if exc["name"]=='photovoltaic panel, multi-Si wafer':
                    original_surface_panel_multi = exc["amount"]
            
            
        
                        
        # Collect original surface of panel for 3kWp, CIS
        for exc in list(photovoltaicslantedroofinstallation3kWpCISpanelmountedonroof_premise.exchanges()):
                print("CCC")
                print(exc)
                print(exc["name"])

                if exc["name"]=='photovoltaic panel, CIS':
                    original_surface_panel_cis= exc["amount"]
            
            
        # Collect original surface of panel for 3kWp, Cdte
        for exc in list(photovoltaicslantedroofinstallation3kWpCdtepanelmountedonroof_premise.exchanges()):
                print("DDD")
                print(exc)
                print(exc["name"])
                if exc["name"]=='photovoltaic laminate, CdTe':
                    original_surface_panel_cdte= exc["amount"]
            
    
        
        # old_efficiency_single,new_efficiency_single=get_PV_efficiency_indatabase(photovoltaicslantedroofinstallation3kWpsingleSipanelmountedonroof_premise)    
        
        # old_efficiency_multi,new_efficiency_multi=get_PV_efficiency_indatabase(photovoltaicslantedroofinstallation3kWpmultiSipanelmountedonroof_premise)    
        
        # old_efficiency_cis,new_efficiency_cis=get_PV_efficiency_indatabase(photovoltaicslantedroofinstallation3kWpCISpanelmountedonroof_premise)    
       
        # old_efficiency_cdte,new_efficiency_cdte=get_PV_efficiency_indatabase(photovoltaicslantedroofinstallation3kWpCdtepanelmountedonroof_premise)    
    
    
        # Just to collect the lifetime of the installation, won't be used later
        elecproduc_singleSIFR= background_db.get([act for act in background_db if act["name"] =='electricity production, photovoltaic, 3kWp slanted-roof installation, single-Si, panel, mounted' and act["location"] == 'FR'][0]["code"])
    
        # The lifetime parameter is not indicated for other technologies
    
        # correctedyield,lifetime=get_PV_yield_and_lifetime_indatabase(elecproduc_singleSIFR)    
    
        
        # Make a copy of the PV unit activity tha we rescale to 1m2 of panel
       
        # Mono/single (original model, to delete afterwards)
        
        photovoltaicmono_installation_perm2panel_AVS = photovoltaicslantedroofinstallation3kWpsingleSipanelmountedonroof_premise.copy()
        photovoltaicmono_installation_perm2panel_AVS["database"]=name_foreground
        photovoltaicmono_installation_perm2panel_AVS["name"] = 'photovoltaicmono_installation_perm2panel_AVS'
        photovoltaicmono_installation_perm2panel_AVS["code"] = 'photovoltaicmono_installation_perm2panel_AVS'
        photovoltaicmono_installation_perm2panel_AVS["unit"] = 'm2'    
        
        photovoltaicmono_installation_perm2panel_AVS.save()
        
        # Make a copy of the PV unit activity tha we rescale to 1m2 of panel
        photovoltaicmono_installation_perm2panel_PV = photovoltaicslantedroofinstallation3kWpsingleSipanelmountedonroof_premise.copy()
        photovoltaicmono_installation_perm2panel_PV["database"]=name_foreground
        photovoltaicmono_installation_perm2panel_PV["name"] = 'photovoltaicmono_installation_perm2panel_PV'
        photovoltaicmono_installation_perm2panel_PV["code"] = 'photovoltaicmono_installation_perm2panel_PV'
        photovoltaicmono_installation_perm2panel_PV["unit"] = 'm2' 
        
        photovoltaicmono_installation_perm2panel_PV.save()
        
        
        # Mono/single;
        
                
        photovoltaicmono_installation_perm2panel_AVS_2 = photovoltaicslantedroofinstallation3kWpsingleSipanelmountedonroof_premise.copy()
        photovoltaicmono_installation_perm2panel_AVS_2["database"]=name_foreground
        photovoltaicmono_installation_perm2panel_AVS_2["name"] = 'photovoltaicmono_installation_perm2panel_AVS_2'
        photovoltaicmono_installation_perm2panel_AVS_2["code"] = 'photovoltaicmono_installation_perm2panel_AVS_2'
        photovoltaicmono_installation_perm2panel_AVS_2["unit"] = 'm2'    
        
        photovoltaicmono_installation_perm2panel_AVS_2.save()
        
        photovoltaicmono_installation_perm2panel_PV_2 = photovoltaicslantedroofinstallation3kWpsingleSipanelmountedonroof_premise.copy()
        photovoltaicmono_installation_perm2panel_PV_2["database"]=name_foreground
        photovoltaicmono_installation_perm2panel_PV_2["name"] = 'photovoltaicmono_installation_perm2panel_PV_2'
        photovoltaicmono_installation_perm2panel_PV_2["code"] = 'photovoltaicmono_installation_perm2panel_PV_2'
        photovoltaicmono_installation_perm2panel_PV_2["unit"] = 'm2' 
        
        photovoltaicmono_installation_perm2panel_PV_2.save()

        
        # Multi
        
        photovoltaicmulti_installation_perm2panel_AVS = photovoltaicslantedroofinstallation3kWpmultiSipanelmountedonroof_premise.copy()
        photovoltaicmulti_installation_perm2panel_AVS["database"]=name_foreground
        photovoltaicmulti_installation_perm2panel_AVS["name"] = 'photovoltaicmulti_installation_perm2panel_AVS'
        photovoltaicmulti_installation_perm2panel_AVS["code"] = 'photovoltaicmulti_installation_perm2panel_AVS'
        photovoltaicmulti_installation_perm2panel_AVS["unit"] = 'm2'    
        
        photovoltaicmulti_installation_perm2panel_AVS.save()
        
        # Make a copy of the PV unit activity tha we rescale to 1m2 of panel
        photovoltaicmulti_installation_perm2panel_PV = photovoltaicslantedroofinstallation3kWpmultiSipanelmountedonroof_premise.copy()
        photovoltaicmulti_installation_perm2panel_PV["database"]=name_foreground
        photovoltaicmulti_installation_perm2panel_PV["name"] = 'photovoltaicmulti_installation_perm2panel_PV'
        photovoltaicmulti_installation_perm2panel_PV["code"] = 'photovoltaicmulti_installation_perm2panel_PV'
        photovoltaicmulti_installation_perm2panel_PV["unit"] = 'm2' 
        
        photovoltaicmulti_installation_perm2panel_PV.save()
        
        
        
        # CIS
        
        photovoltaicCIS_installation_perm2panel_AVS = photovoltaicslantedroofinstallation3kWpCISpanelmountedonroof_premise.copy()
        photovoltaicCIS_installation_perm2panel_AVS["database"]=name_foreground
        photovoltaicCIS_installation_perm2panel_AVS["name"] = 'photovoltaicCIS_installation_perm2panel_AVS'
        photovoltaicCIS_installation_perm2panel_AVS["code"] = 'photovoltaicCIS_installation_perm2panel_AVS'
        photovoltaicCIS_installation_perm2panel_AVS["unit"] = 'm2'    
        
        photovoltaicCIS_installation_perm2panel_AVS.save()
        
        # Make a copy of the PV unit activity tha we rescale to 1m2 of panel
        photovoltaicCIS_installation_perm2panel_PV = photovoltaicslantedroofinstallation3kWpCISpanelmountedonroof_premise.copy()
        photovoltaicCIS_installation_perm2panel_PV["database"]=name_foreground
        photovoltaicCIS_installation_perm2panel_PV["name"] = 'photovoltaicCIS_installation_perm2panel_PV'
        photovoltaicCIS_installation_perm2panel_PV["code"] = 'photovoltaicCIS_installation_perm2panel_PV'
        photovoltaicCIS_installation_perm2panel_PV["unit"] = 'm2' 
        
        photovoltaicCIS_installation_perm2panel_PV.save()
        
        
        
        # Cdte
        
        photovoltaicCdte_installation_perm2panel_AVS = photovoltaicslantedroofinstallation3kWpCdtepanelmountedonroof_premise.copy()
        photovoltaicCdte_installation_perm2panel_AVS["database"]=name_foreground
        photovoltaicCdte_installation_perm2panel_AVS["name"] = 'photovoltaicCdte_installation_perm2panel_AVS'
        photovoltaicCdte_installation_perm2panel_AVS["code"] = 'photovoltaicCdte_installation_perm2panel_AVS'
        photovoltaicCdte_installation_perm2panel_AVS["unit"] = 'm2'    
        
        photovoltaicCdte_installation_perm2panel_AVS.save()
        
        # Make a copy of the PV unit activity tha we rescale to 1m2 of panel
        photovoltaicCdte_installation_perm2panel_PV = photovoltaicslantedroofinstallation3kWpCdtepanelmountedonroof_premise.copy()
        photovoltaicCdte_installation_perm2panel_PV["database"]=name_foreground
        photovoltaicCdte_installation_perm2panel_PV["name"] = 'photovoltaicCdte_installation_perm2panel_PV'
        photovoltaicCdte_installation_perm2panel_PV["code"] = 'photovoltaicCdte_installation_perm2panel_PV'
        photovoltaicCdte_installation_perm2panel_PV["unit"] = 'm2' 
        
        photovoltaicCdte_installation_perm2panel_PV.save()
        
        photovoltaicCdte_installation_perm2panel_PV = foregroundAVS.get("photovoltaicCdte_installation_perm2panel_PV")

        # Remove transport  and heat waste (biosphere )for Cdte to make it the same as for others
        
        for exc in list(photovoltaicCdte_installation_perm2panel_PV.exchanges()):
            if exc["type"]=="technosphere":
                #print(exc["input"])
                input_= background_db.get(exc["input"][1])
                input_name = input_["name"]
                if "transport" in input_name:
                    
                    exc.delete()
                    
                    
        for exc in list(photovoltaicCdte_installation_perm2panel_AVS.exchanges()):
            if exc["type"]=="technosphere":
                #print(exc["input"])
                input_= background_db.get(exc["input"][1])
                input_name = input_["name"]
                if "transport" in input_name:
                    
                    exc.delete()            

        for exc in list(photovoltaicCdte_installation_perm2panel_AVS.exchanges()):
            if exc["type"]=="biosphere":
                #print(exc["input"])
                input_= biosphere_db.get(exc["input"][1])
                input_name = input_["name"]
                if "Heat" in input_name:
                    
                    exc.delete()     
                    
        for exc in list(photovoltaicCdte_installation_perm2panel_PV.exchanges()):
            if exc["type"]=="biosphere":
                #print(exc["input"])
                input_= biosphere_db.get(exc["input"][1])
                input_name = input_["name"]
                if "Heat" in input_name:
                    
                    exc.delete()                  

        
        
        def rescale_photovolt_act(act,
                                  background_db,
                                  original_surface_panel):
            
            for exc in list(act.exchanges()):
                    # exc_former = exc["amount"]
                    # exc["amount"] = exc_former/sum(list_inputs_electric_instal_weights)
                    # exc.save()
                    if exc["type"]=="technosphere":
            
                        input_= background_db.get(exc["input"][1])
                        input_name = input_["name"]
                        input_loc = input_["location"]
                        if "electricity" in input_name or "market for photovoltaic" in input_name or "treatment" in input_name or "Heat" in input_name:
                          #print(input_name)  
                          exc_former = exc["amount"]
            
                          exc["amount"] = exc_former/original_surface_panel
                          exc.save()
                          
        
        rescale_photovolt_act(photovoltaicmono_installation_perm2panel_AVS,
                                  background_db,
                                  original_surface_panel_single)
        

                      
        rescale_photovolt_act(photovoltaicmono_installation_perm2panel_PV,
                                  background_db,
                                  original_surface_panel_single)
        
        
                
        rescale_photovolt_act(photovoltaicmono_installation_perm2panel_AVS_2,
                                  background_db,
                                  original_surface_panel_single)
        

                      
        rescale_photovolt_act(photovoltaicmono_installation_perm2panel_PV_2,
                                  background_db,
                                  original_surface_panel_single)
        

        # Multi
                
        
        rescale_photovolt_act(photovoltaicmulti_installation_perm2panel_AVS,
                                  background_db,
                                  original_surface_panel_multi)
        


                      
        rescale_photovolt_act(photovoltaicmulti_installation_perm2panel_PV,
                                  background_db,
                                  original_surface_panel_multi)
        
        
        
        #CIS
        rescale_photovolt_act(photovoltaicCIS_installation_perm2panel_AVS,
                                  background_db,
                                  original_surface_panel_cis)
        
        rescale_photovolt_act(photovoltaicCIS_installation_perm2panel_PV,
                                  background_db,
                                  original_surface_panel_cis)
        
        
        # Cdte
        
        
        rescale_photovolt_act(photovoltaicCdte_installation_perm2panel_AVS,
                                  background_db,
                                  original_surface_panel_cdte)
        
        rescale_photovolt_act(photovoltaicCdte_installation_perm2panel_PV,
                                  background_db,
                                  original_surface_panel_cdte)
        
        
        
        
        # Remove mounting strucuture
        
        
        def remove_mounting_structure(act,background_db):
            
            #we remove the mounting systems in both PV and AVS installation
            
            
            for exc in list(act.exchanges()):
                    # exc_former = exc["amount"]
                    # exc["amount"] = exc_former/sum(list_inputs_electric_instal_weights)
                    # exc.save()
                    if exc["type"]=="technosphere":
            
                        input_= background_db.get(exc["input"][1])
                        input_name = input_["name"]
            
                        if "mounting system" in input_name:
                            
                        # exc["amount"] = exc_former/sum(list_inputs_electric_instal_weights)
                        # exc.save()
                            exc.delete()
                        
        
        remove_mounting_structure(photovoltaicmono_installation_perm2panel_AVS,background_db)
        remove_mounting_structure(photovoltaicmono_installation_perm2panel_PV,background_db)

        remove_mounting_structure(photovoltaicmono_installation_perm2panel_AVS_2,background_db)
        remove_mounting_structure(photovoltaicmono_installation_perm2panel_PV_2,background_db)
               

        remove_mounting_structure(photovoltaicmulti_installation_perm2panel_AVS,background_db)
        remove_mounting_structure(photovoltaicmulti_installation_perm2panel_PV,background_db)
       
        remove_mounting_structure(photovoltaicCIS_installation_perm2panel_AVS,background_db)
        remove_mounting_structure(photovoltaicCIS_installation_perm2panel_PV,background_db)
         
        remove_mounting_structure(photovoltaicCdte_installation_perm2panel_AVS,background_db)
        remove_mounting_structure(photovoltaicCdte_installation_perm2panel_PV,background_db)
         
       
        """Electric installation"""
        
        # Now we will create modifed version of the inverter, the electic installation and the panel

        
        
        photovoltaics_electric_installation_570kWpmodule= background_db.get([act for act in background_db if act["name"] =='photovoltaics, electric installation for 570kWp module, open ground'and act["location"] == 'GLO'][0]["code"])
        
        #electric installation 
        list_inputs_electric_instal_id = []
        list_inputs_electric_instal_weights = []
        
        for exc in list(photovoltaics_electric_installation_570kWpmodule.exchanges()):
          
                input_= background_db.get(exc["input"][1])
                input_name = input_["name"]
                input_loc = input_["location"]
                
                #print(input_name,input_loc, exc["amount"],exc["unit"])
                
                list_inputs_electric_instal_id.append(input_.id)
                if exc["amount"]>0 and exc["unit"]=="kilogram":
                    list_inputs_electric_instal_weights.append(exc["amount"])
            
        
        total_weight_electric_installation = sum(list_inputs_electric_instal_weights)
        
        mass_to_power_electric_installation = total_weight_electric_installation/570
        # Make a copy of the electric installation
        
        electricpvinstallation_kg= photovoltaics_electric_installation_570kWpmodule.copy()
        
        electricpvinstallation_kg["database"]=name_foreground
        electricpvinstallation_kg["name"] = 'electricpvinstallation_kg'
        electricpvinstallation_kg["code"] = 'electricpvinstallation_kg'
        electricpvinstallation_kg["unit"] = 'kg'
        
        electricpvinstallation_kg.save()
        
        # It is 1 kg of electrical installation. 
        
        for exc in list(electricpvinstallation_kg.exchanges()):
            if exc["type"]!="production":
                exc_former = exc["amount"]
                exc["amount"] = exc_former/total_weight_electric_installation
                
                
                exc.save()
          
        electricpvinstallation_kg.save()
        
        
        
        
        # We replace the input of installation by our new activity. 
        # So far we put 1, but the amount will be parameterized.
        
        

        
        def replace_electric_installation(act,background_db,electricpvinstallation_kg):
            
            #first delete original

            for exc in list(act.exchanges()):
                    
                    if exc["type"]=="technosphere":
            
                        input_= background_db.get(exc["input"][1])
                        input_name = input_["name"]
                        input_loc = input_["location"]
                        if "electric installation" in input_name:
                            
                          exc.delete()
                          
            # add the rescaled one              
            act.new_exchange(amount=1, input=electricpvinstallation_kg , type="technosphere",unit="kilogram").save()

        # Mono
        
        
        # PV    
        replace_electric_installation(photovoltaicmono_installation_perm2panel_PV,background_db,electricpvinstallation_kg)  
        replace_electric_installation(photovoltaicmono_installation_perm2panel_PV_2,background_db,electricpvinstallation_kg)  
        
        
        # AVS
        
        replace_electric_installation(photovoltaicmono_installation_perm2panel_AVS,background_db,electricpvinstallation_kg)  
        replace_electric_installation(photovoltaicmono_installation_perm2panel_AVS_2,background_db,electricpvinstallation_kg)  

        # # Now add exchange of the reolacement activity
        # photovoltaicmono_installation_perm2panel_AVS.new_exchange(amount=1, input=electricpvinstallation_kg , type="technosphere",unit="kilogram").save()
        # photovoltaicmono_installation_perm2panel_PV.new_exchange(amount=1, input=electricpvinstallation_kg , type="technosphere",unit="kilogram").save()
        
        
        # Multi

        
        # PV
        replace_electric_installation(photovoltaicmulti_installation_perm2panel_PV,background_db,electricpvinstallation_kg)  

        
                      
        # AVS
        
        replace_electric_installation(photovoltaicmulti_installation_perm2panel_AVS,background_db,electricpvinstallation_kg)  

        
        # CIS
        
        # PV
        replace_electric_installation(photovoltaicCIS_installation_perm2panel_PV,background_db,electricpvinstallation_kg)  

                      
        # AVS
        
        replace_electric_installation(photovoltaicCIS_installation_perm2panel_AVS,background_db,electricpvinstallation_kg)  

        
     
        # Cdte
        
        # PV
        replace_electric_installation(photovoltaicCdte_installation_perm2panel_PV,background_db,electricpvinstallation_kg)  

                      
        # AVS
        
        replace_electric_installation(photovoltaicCdte_installation_perm2panel_AVS,background_db,electricpvinstallation_kg)  

        
        
        """Inverter"""
        
        
        
        
        #Update inverter
        
        

            
        
                  
        
        # No need to interpolate cause the VERY minimum installed power will already be around 100 kwp
        # So let's just take the 500 kwp modified inverter 
        
        market_for_inverter_500kW= background_db.get([act for act in background_db if act["name"] =='market for inverter, 500kW'and act["location"] == 'GLO'][0]["code"])
        
        
        inverter_500_row= background_db.get([act for act in background_db if act["name"] =='inverter production, 500kW' and act["location"] == 'RoW'][0]["code"])
        
        
        
        
        inverter_500_rer= background_db.get([act for act in background_db if act["name"] =='inverter production, 500kW' and act["location"] == 'RER'][0]["code"])
        
        
        market_for_inverter_500kW_kg=market_for_inverter_500kW.copy()
        market_for_inverter_500kW_kg["database"]=name_foreground
        market_for_inverter_500kW_kg["name"] = "market_for_inverter_500kW_kg"
        market_for_inverter_500kW_kg["code"] = "market_for_inverter_500kW_kg"
        market_for_inverter_500kW_kg["unit"] = 'kilogram'
        
        market_for_inverter_500kW_kg.save()
        
        

        
        # This was done to rescale to 1 kg of inverter
        
        list_inputs_inverter_id = []
        list_inputs_inverter_weights = []
        df_=pd.DataFrame({'input':[], 
                'amount':[], 
                'unit':[] } )
        
        # Calculate total weightt
        
        
        for exc in list(inverter_500_rer.exchanges()):
          
                input_= background_db.get(exc["input"][1])
                input_name = input_["name"]
                input_loc = input_["location"]
                
                #print(input_name,input_loc, exc["amount"],exc["unit"])
                df_.loc[len(df_.index)] = [ input_name, exc["amount"], exc["unit"]]  
        
                list_inputs_inverter_id.append(input_.id)
                if exc["amount"]>0 and exc["type"]!="production" and exc["unit"]=="kilogram":
                    list_inputs_inverter_weights.append(exc["amount"])
        
        weigth_500kWrer= df_[(df_.unit == 'kilogram')&(df_.amount > 0)].amount.drop_duplicates().sum()
        
        weigth_500kWrer/500  # 5.97 kg inverter per kWP
        
        inverter_500_rer_kg=inverter_500_rer.copy()
        inverter_500_rer_kg["database"]=name_foreground
        inverter_500_rer_kg["name"] = "inverter_500_rer_kg"
        inverter_500_rer_kg["code"] = "inverter_500_rer_kg"
        inverter_500_rer_kg["unit"] = 'kilogram'
        
        inverter_500_rer_kg.save()
        
        # Collect inverter_specificweight kg.kWp
        inverter_specificweight = weigth_500kWrer/500
        
        
        
        for exc in list(inverter_500_rer_kg.exchanges()):
            if exc["type"]!="production":
                exc_former = exc["amount"]
                exc["amount"] = exc_former/weigth_500kWrer ####
                exc.save()
        
        
        
            
        
        # RoW
        
        inverter_500_row_kg=inverter_500_row.copy()
        inverter_500_row_kg["database"]=name_foreground
        inverter_500_row_kg["name"] = "inverter_500_row_kg"
        inverter_500_row_kg["code"] = "inverter_500_row_kg"
        inverter_500_row_kg["unit"] = 'kilogram'
        
        inverter_500_row_kg.save()
        
        list_inputs_inverter_id = []
        list_inputs_inverter_weights = []
        df_=pd.DataFrame({'input':[], 
                'amount':[], 
                'unit':[] } )
        # Calculate total weight
        for exc in list(inverter_500_row.exchanges()):
          
                input_= background_db.get(exc["input"][1])
                input_name = input_["name"]
                input_loc = input_["location"]
                
                #print(input_name,input_loc, exc["amount"],exc["unit"])
                df_.loc[len(df_.index)] = [ input_name, exc["amount"], exc["unit"]]  
        
                list_inputs_inverter_id.append(input_.id)
                if exc["amount"]>0 and exc["unit"]=="kilogram":
                    list_inputs_inverter_weights.append(exc["amount"])
        
        #print('Weight of the inverter : %.0f kg'%df_[(df_.unit == 'kilogram')&(df_.amount > 0)].amount.drop_duplicates().sum())
        weigth_500kWrow= df_[(df_.unit == 'kilogram')&(df_.amount > 0)].amount.drop_duplicates().sum()
        
        
        for exc in list(inverter_500_row_kg.exchanges()):
            if exc["type"]!="production":        
                exc_former = exc["amount"]
                exc["amount"] = exc_former/weigth_500kWrow
                exc.save()
                
                
            
                
        # Now we add the new inverters to the market
        
        
        for exc in list(market_for_inverter_500kW_kg.exchanges()):
                if exc["type"]!="production"    :
            
                    input_= background_db.get(exc["input"][1])
                    input_name = input_["name"]
                    input_loc = input_["location"]
                    
                    if input_loc =="RER":
                        exc_former = exc["amount"]
                        
                        market_for_inverter_500kW_kg.new_exchange(amount=exc_former,
                                                                      input=inverter_500_rer_kg, 
                                                                      type="technosphere",
                                                                      unit="kilogram").save()
                        exc.delete()
                    elif input_loc == "RoW":
                        exc_former = exc["amount"]
                        
                        market_for_inverter_500kW_kg.new_exchange(amount=exc_former,
                                                                      input=inverter_500_row_kg, 
                                                                      type="technosphere",
                                                                      unit="kilogram").save()
                        exc.delete()
                        
        market_for_inverter_500kW_kg.save()               
        
                
                
        # # Now add exchange new inverter of the reolacement activity

        
        
        def replace_inverter(act,
                             background_db,
                             foregroundAVS,
                             market_for_inverter_500kW_kg):
            
            
            
            for exc in list(act.exchanges()):
            
                    if exc["type"]=="technosphere":
                        #print("oook")
                        if exc["input"][0] ==name_background:
                            input_= background_db.get(exc["input"][1])
                        else:    
                            input_= foregroundAVS.get(exc["input"][1])
                            
                        input_name = input_["name"]
                        input_loc = input_["location"]
                        if "inverter" in input_name:
                            
                          exc.delete()
        
            act.new_exchange(amount=1, input=market_for_inverter_500kW_kg , type="technosphere",unit="kilogram").save()


                      

            
        # Mono


        # AVS
        replace_inverter(photovoltaicmono_installation_perm2panel_AVS,
                             background_db,
                             foregroundAVS,
                             market_for_inverter_500kW_kg)

        # PV
        replace_inverter(photovoltaicmono_installation_perm2panel_PV,
                             background_db,
                             foregroundAVS,
                             market_for_inverter_500kW_kg)
        
        # AVS
        replace_inverter(photovoltaicmono_installation_perm2panel_AVS_2,
                             background_db,
                             foregroundAVS,
                             market_for_inverter_500kW_kg)

        # PV
        replace_inverter(photovoltaicmono_installation_perm2panel_PV_2,
                             background_db,
                             foregroundAVS,
                             market_for_inverter_500kW_kg)
        
        
        
        # Multi


        # AVS
        replace_inverter(photovoltaicmulti_installation_perm2panel_AVS,
                             background_db,
                             foregroundAVS,
                             market_for_inverter_500kW_kg)

        # PV
        replace_inverter(photovoltaicmulti_installation_perm2panel_PV,
                             background_db,
                             foregroundAVS,
                             market_for_inverter_500kW_kg)
        
        
        
        # CIS


        # AVS
        replace_inverter(photovoltaicCIS_installation_perm2panel_AVS,
                             background_db,
                             foregroundAVS,
                             market_for_inverter_500kW_kg)

        # PV
        replace_inverter(photovoltaicCIS_installation_perm2panel_PV,
                             background_db,
                             foregroundAVS,
                             market_for_inverter_500kW_kg)
        


      
        # Cdte


        # AVS
        replace_inverter(photovoltaicCdte_installation_perm2panel_AVS,
                             background_db,
                             foregroundAVS,
                             market_for_inverter_500kW_kg)

        # PV
        replace_inverter(photovoltaicCdte_installation_perm2panel_PV,
                             background_db,
                             foregroundAVS,
                             market_for_inverter_500kW_kg)
                
        """ Modification of the mounting strucuture"""
        
        
        
        
        
        
        
        # 'photovoltaic mounting system production, for 570kWp open ground module' (square meter, GLO, None)
        
        
        
        mountingsystem= background_db.get([act for act in background_db if act["name"] =='photovoltaic mounting system production, for 570kWp open ground module'and act["location"] == 'GLO'][0]["code"])
        
        
    
                    
        mountingsystem_modif_PV = mountingsystem.copy()   
        
        mountingsystem_modif_PV["name"]="mountingsystem_modif_PV"
        mountingsystem_modif_PV["database"] = name_foreground
        mountingsystem_modif_PV["code"]="mountingsystem_modif_PV"
        mountingsystem_modif_PV.save()
        
        
                    
        mountingsystem_modif_AVS = mountingsystem.copy()   
        
        mountingsystem_modif_AVS["name"]="mountingsystem_modif_AVS"
        mountingsystem_modif_AVS["database"] = name_foreground
        mountingsystem_modif_AVS["code"]="mountingsystem_modif_AVS"
        mountingsystem_modif_AVS.save()
        
        

        # Delete occupation and transfo
        
        for exc in list(mountingsystem_modif_AVS.exchanges()):
            if exc["type"]== "biosphere":
                
                exc.delete()
        mountingsystem_modif_AVS.save()        
        
        for exc in list(mountingsystem_modif_PV.exchanges()):
            if exc["type"]== "biosphere":
                
                exc.delete()
        mountingsystem_modif_PV.save()        
        
        
        
        # Add inputs of panel and mounting structure to the AVS and PV production
        
        # Mounting structure
        
        
        
        
        # Here make copies for all technologies
        
        
        
        # Cdte
        
        
        
        AVS_elec_main= foregroundAVS.get("AVS_elec_main")
        
        AVS_elec_main_single = AVS_elec_main.copy()
        AVS_elec_main_single["name"] = "AVS_elec_main_single"
        AVS_elec_main_single["code"] = "AVS_elec_main_single"
        AVS_elec_main_single.save()
        
        
        AVS_elec_main_multi = AVS_elec_main.copy()
        AVS_elec_main_multi["name"] = "AVS_elec_main_multi"
        AVS_elec_main_multi["code"] = "AVS_elec_main_multi"
        AVS_elec_main_multi.save()
        
                
        AVS_elec_main_cis = AVS_elec_main.copy()
        AVS_elec_main_cis["name"] = "AVS_elec_main_cis"
        AVS_elec_main_cis["code"] = "AVS_elec_main_cis"
        AVS_elec_main_cis.save()
        
        AVS_elec_main_cdte = AVS_elec_main.copy()
        AVS_elec_main_cdte["name"] = "AVS_elec_main_cdte"
        AVS_elec_main_cdte["code"] = "AVS_elec_main_cdte"
        AVS_elec_main_cdte.save()


        # Crop main
        
        AVS_crop_main= foregroundAVS.get("AVS_crop_main")
        
                
        AVS_crop_main_single = AVS_crop_main.copy()
        AVS_crop_main_single["name"] = "AVS_crop_main_single"
        AVS_crop_main_single["code"] = "AVS_crop_main_single"
        AVS_crop_main_single.save()
        
        
        AVS_crop_main_multi = AVS_crop_main.copy()
        AVS_crop_main_multi["name"] = "AVS_crop_main_multi"
        AVS_crop_main_multi["code"] = "AVS_crop_main_multi"
        AVS_crop_main_multi.save()
        
                
        AVS_crop_main_cis = AVS_crop_main.copy()
        AVS_crop_main_cis["name"] = "AVS_crop_main_cis"
        AVS_crop_main_cis["code"] = "AVS_crop_main_cis"
        AVS_crop_main_cis.save()
        
        AVS_crop_main_cdte = AVS_crop_main.copy()
        AVS_crop_main_cdte["name"] = "AVS_crop_main_cdte"
        AVS_crop_main_cdte["code"] = "AVS_crop_main_cdte"
        AVS_crop_main_cdte.save()
        
        
        
        PV_ref= foregroundAVS.get("PV_ref")
        
        
        PV_ref_single = PV_ref.copy()
        PV_ref_single["name"] = "PV_ref_single"
        PV_ref_single["code"] = "PV_ref_single"
        PV_ref_single.save()
        
        
        PV_ref_multi = PV_ref.copy()
        PV_ref_multi["name"] = "PV_ref_multi"
        PV_ref_multi["code"] = "PV_ref_multi"
        PV_ref_multi.save()
        
                
        PV_ref_cis = PV_ref.copy()
        PV_ref_cis["name"] = "PV_ref_cis"
        PV_ref_cis["code"] = "PV_ref_cis"
        PV_ref_cis.save()
        
        PV_ref_cdte = PV_ref.copy()
        PV_ref_cdte["name"] = "PV_ref_cdte"
        PV_ref_cdte["code"] = "PV_ref_cdte"
        PV_ref_cdte.save()

        
        
        
        AVS_elec_main.new_exchange(amount=1, input=mountingsystem_modif_AVS, type="technosphere").save()
        AVS_elec_main_single.new_exchange(amount=1, input=mountingsystem_modif_AVS, type="technosphere").save()
        AVS_elec_main_multi.new_exchange(amount=1, input=mountingsystem_modif_AVS, type="technosphere").save()
        AVS_elec_main_cis.new_exchange(amount=1, input=mountingsystem_modif_AVS, type="technosphere").save()
        AVS_elec_main_cdte.new_exchange(amount=1, input=mountingsystem_modif_AVS, type="technosphere").save()
        
        AVS_crop_main.new_exchange(amount=1, input=mountingsystem_modif_AVS, type="technosphere").save()
        AVS_crop_main_single.new_exchange(amount=1, input=mountingsystem_modif_AVS, type="technosphere").save()
        AVS_crop_main_multi.new_exchange(amount=1, input=mountingsystem_modif_AVS, type="technosphere").save()
        AVS_crop_main_cis.new_exchange(amount=1, input=mountingsystem_modif_AVS, type="technosphere").save()
        AVS_crop_main_cdte.new_exchange(amount=1, input=mountingsystem_modif_AVS, type="technosphere").save()
        
        
        PV_ref.new_exchange(amount=1, input=mountingsystem_modif_PV, type="technosphere").save()
        PV_ref_single.new_exchange(amount=1, input=mountingsystem_modif_PV, type="technosphere").save()
        PV_ref_multi.new_exchange(amount=1, input=mountingsystem_modif_PV, type="technosphere").save()
        PV_ref_cis.new_exchange(amount=1, input=mountingsystem_modif_PV, type="technosphere").save()
        PV_ref_cdte.new_exchange(amount=1, input=mountingsystem_modif_PV, type="technosphere").save()
        
        
        
        # Panel instal
        
        
        AVS_elec_main.new_exchange(amount=1, input=photovoltaicmono_installation_perm2panel_AVS, type="technosphere").save()
        AVS_elec_main_single.new_exchange(amount=1, input=photovoltaicmono_installation_perm2panel_AVS_2, type="technosphere").save()
        AVS_elec_main_multi.new_exchange(amount=1, input=photovoltaicmulti_installation_perm2panel_AVS, type="technosphere").save()
        AVS_elec_main_cis.new_exchange(amount=1, input=photovoltaicCIS_installation_perm2panel_AVS, type="technosphere").save()
        AVS_elec_main_cdte.new_exchange(amount=1, input=photovoltaicCdte_installation_perm2panel_AVS, type="technosphere").save()
        
        AVS_crop_main.new_exchange(amount=1, input=photovoltaicmono_installation_perm2panel_AVS, type="technosphere").save()
        AVS_crop_main_single.new_exchange(amount=1, input=photovoltaicmono_installation_perm2panel_AVS_2, type="technosphere").save()
        AVS_crop_main_multi.new_exchange(amount=1, input=photovoltaicmulti_installation_perm2panel_AVS, type="technosphere").save()
        AVS_crop_main_cis.new_exchange(amount=1, input=photovoltaicCIS_installation_perm2panel_AVS, type="technosphere").save()
        AVS_crop_main_cdte.new_exchange(amount=1, input=photovoltaicCdte_installation_perm2panel_AVS, type="technosphere").save()
        
        
        PV_ref.new_exchange(amount=1, input=photovoltaicmono_installation_perm2panel_PV, type="technosphere").save()
        PV_ref_single.new_exchange(amount=1, input=photovoltaicmono_installation_perm2panel_PV_2, type="technosphere").save()
        PV_ref_multi.new_exchange(amount=1, input=photovoltaicmulti_installation_perm2panel_PV, type="technosphere").save()
        PV_ref_cis.new_exchange(amount=1, input=photovoltaicCIS_installation_perm2panel_PV, type="technosphere").save()
        PV_ref_cdte.new_exchange(amount=1, input=photovoltaicCdte_installation_perm2panel_PV, type="technosphere").save()
        
        

        
    
  

        # Add occupation of industrial area for PV ref.
        # AVS has kept the same occupation as the associated crop.
        occupation_industrial_code = [act["code"] for act in biosphere_db if 'Occupation, industrial area' in act["name"]][0]
        occupation_industrial=biosphere_db.get(occupation_industrial_code)
        
        PV_ref.new_exchange(amount=10000, input=occupation_industrial, type="biosphere").save()
        PV_ref_single.new_exchange(amount=10000, input=occupation_industrial, type="biosphere").save()
        PV_ref_multi.new_exchange(amount=10000, input=occupation_industrial, type="biosphere").save()
        PV_ref_cis.new_exchange(amount=10000, input=occupation_industrial, type="biosphere").save()
        PV_ref_cdte.new_exchange(amount=10000, input=occupation_industrial, type="biosphere").save()
        
        
        # Add elec prod for AVS_crop_main
        
        elec_marginal_fr_copy = foregroundAVS.get("elec_marginal_fr_copy")
        elec_marginal_fr_current_copy = foregroundAVS.get("elec_marginal_fr_current_copy")
        elec_marginal_it_current_copy = foregroundAVS.get("elec_marginal_it_current_copy")
        elec_marginal_es_current_copy = foregroundAVS.get("elec_marginal_es_current_copy")
        
        AVS_crop_main.new_exchange(amount=-1, input=elec_marginal_fr_copy, type="technosphere").save()
        AVS_crop_main_single.new_exchange(amount=-1, input=elec_marginal_fr_copy, type="technosphere").save()
        AVS_crop_main_multi.new_exchange(amount=-1, input=elec_marginal_fr_copy, type="technosphere").save()
        AVS_crop_main_cis.new_exchange(amount=-1, input=elec_marginal_fr_copy, type="technosphere").save()
        AVS_crop_main_cdte.new_exchange(amount=-1, input=elec_marginal_fr_copy, type="technosphere").save()


        AVS_crop_main.new_exchange(amount=-1, input=elec_marginal_fr_current_copy, type="technosphere").save()
        AVS_crop_main_single.new_exchange(amount=-1, input=elec_marginal_fr_current_copy, type="technosphere").save()
        AVS_crop_main_multi.new_exchange(amount=-1, input=elec_marginal_fr_current_copy, type="technosphere").save()
        AVS_crop_main_cis.new_exchange(amount=-1, input=elec_marginal_fr_current_copy, type="technosphere").save()
        AVS_crop_main_cdte.new_exchange(amount=-1, input=elec_marginal_fr_current_copy, type="technosphere").save()
        
        AVS_crop_main.new_exchange(amount=-1, input=elec_marginal_it_current_copy, type="technosphere").save()
        AVS_crop_main_single.new_exchange(amount=-1, input=elec_marginal_it_current_copy, type="technosphere").save()
        AVS_crop_main_multi.new_exchange(amount=-1, input=elec_marginal_it_current_copy, type="technosphere").save()
        AVS_crop_main_cis.new_exchange(amount=-1, input=elec_marginal_it_current_copy, type="technosphere").save()
        AVS_crop_main_cdte.new_exchange(amount=-1, input=elec_marginal_it_current_copy, type="technosphere").save()
  
        AVS_crop_main.new_exchange(amount=-1, input=elec_marginal_es_current_copy, type="technosphere").save()
        AVS_crop_main_single.new_exchange(amount=-1, input=elec_marginal_es_current_copy, type="technosphere").save()
        AVS_crop_main_multi.new_exchange(amount=-1, input=elec_marginal_es_current_copy, type="technosphere").save()
        AVS_crop_main_cis.new_exchange(amount=-1, input=elec_marginal_es_current_copy, type="technosphere").save()
        AVS_crop_main_cdte.new_exchange(amount=-1, input=elec_marginal_es_current_copy, type="technosphere").save()
        
        
        AVS_crop_main.new_exchange(amount=-1, input=PV_ref, type="technosphere").save()
        AVS_crop_main_single.new_exchange(amount=-1, input=PV_ref_single, type="technosphere").save()
        AVS_crop_main_multi.new_exchange(amount=-1, input=PV_ref_multi, type="technosphere").save()
        AVS_crop_main_cis.new_exchange(amount=-1, input=PV_ref_cis, type="technosphere").save()
        AVS_crop_main_cdte.new_exchange(amount=-1, input=PV_ref_cdte, type="technosphere").save()
        

        
        
        
        
        AVS_elec_main.save()
        AVS_elec_main_single.save()
        AVS_elec_main_multi.save()
        AVS_elec_main_cis.save()
        AVS_elec_main_cdte.save()
        
        AVS_crop_main.save()
        AVS_crop_main_single.save()
        AVS_crop_main_multi.save()
        AVS_crop_main_cis.save()
        AVS_crop_main_cdte.save()        

        PV_ref.save()
        PV_ref_single.save()
        PV_ref_multi.save()
        PV_ref_cis.save()
        PV_ref_cdte.save()
        
        
        
        
   
    else:
        
        print("FOREGROUND ALREADY EXISTS")
        print("UUUUUU",background_db.name)
        
        print([act for act in background_db if act["name"] =='photovoltaic slanted-roof installation, 3kWp, single-Si, panel, mounted, on roof'and act["location"] == 'CH'])
        photovoltaicslantedroofinstallation3kWpsingleSipanelmountedonroof_premise= background_db.get([act for act in background_db if act["name"] =='photovoltaic slanted-roof installation, 3kWp, single-Si, panel, mounted, on roof'and act["location"] == 'CH'][0]["code"])
        
        photovoltaicslantedroofinstallation3kWpmultiSipanelmountedonroof_premise= background_db.get([act for act in background_db if  act["name"] =='photovoltaic slanted-roof installation, 3kWp, multi-Si, panel, mounted, on roof' and act["location"] =="CH"][0]["code"])
        
        photovoltaicslantedroofinstallation3kWpCISpanelmountedonroof_premise= background_db.get([act for act in background_db if act["name"] =='photovoltaic slanted-roof installation, 3kWp, CIS, panel, mounted, on roof' and act["location"] == 'CH'][0]["code"])
        
        if background_db.name != "ecoinvent-3.10-consequential":
            photovoltaicslantedroofinstallation3kWpCdtepanelmountedonroof_premise= background_db.get([act for act in background_db if act["name"] =='photovoltaic slanted-roof installation, 3 kWp, CdTe, panel, mounted, on roof' and act["location"] == 'CH'][0]["code"])

        else:
            photovoltaicslantedroofinstallation3kWpCdtepanelmountedonroof_premise= background_db.get([act for act in background_db if "CdTe" in act["name"] and "photovoltaic slanted-roof installation" in act["name"] and act["location"] == 'CH'][0]["code"])


        # list(photovoltaicslantedroofinstallation3kWpsingleSipanelmountedonroof_premise.keys())
        
        # Collect original surface of panel for 3kWp
        for exc in list(photovoltaicslantedroofinstallation3kWpsingleSipanelmountedonroof_premise.exchanges()):
                print("AAA")

                print(exc)
                print(exc["name"])
                if exc["name"]=='photovoltaic panel, single-Si wafer':
                    print("OKKKKK")
                    original_surface_panel_single = exc["amount"]
            
    
        for exc in list(photovoltaicslantedroofinstallation3kWpmultiSipanelmountedonroof_premise.exchanges()):
                print("BBB")
                print(exc)
                print(exc["name"])

                if exc["name"]=='photovoltaic panel, multi-Si wafer':
                    original_surface_panel_multi = exc["amount"]
        
        # Collect original surface of panel for 3kWp, CIS
        for exc in list(photovoltaicslantedroofinstallation3kWpCISpanelmountedonroof_premise.exchanges()):
                print("CCC")
                print(exc)
                print(exc["name"])
                if exc["name"]=='photovoltaic panel, CIS':
                    original_surface_panel_cis= exc["amount"]
            
            
        # Collect original surface of panel for 3kWp, Cdte
        for exc in list(photovoltaicslantedroofinstallation3kWpCdtepanelmountedonroof_premise.exchanges()):
                print("DDD")
                print(exc)
                print(exc["name"])

                if exc["name"]=='photovoltaic laminate, CdTe':
                    original_surface_panel_cdte= exc["amount"]
            
        
        
        
        
        # old_efficiency_single,new_efficiency_single=get_PV_efficiency_indatabase(photovoltaicslantedroofinstallation3kWpsingleSipanelmountedonroof_premise)    
        
        # old_efficiency_multi,new_efficiency_multi=get_PV_efficiency_indatabase(photovoltaicslantedroofinstallation3kWpmultiSipanelmountedonroof_premise)    
        
        # old_efficiency_cis,new_efficiency_cis=get_PV_efficiency_indatabase(photovoltaicslantedroofinstallation3kWpCISpanelmountedonroof_premise)    
       
        # old_efficiency_cdte,new_efficiency_cdte=get_PV_efficiency_indatabase(photovoltaicslantedroofinstallation3kWpCdtepanelmountedonroof_premise)    
        
        # Just to collect the lifetime of the installation, won't be used later
        elecproduc_singleSIFR= background_db.get([act for act in background_db if act["name"] =='electricity production, photovoltaic, 3kWp slanted-roof installation, single-Si, panel, mounted' and act["location"] == 'FR'][0]["code"])
    
    
        # correctedyield,lifetime=get_PV_yield_and_lifetime_indatabase(elecproduc_singleSIFR)    
    
        

                
        # Now we will create modifed version of the inverter, the electic installation and the panel
        
        
        """Electric installation"""
        
        
        
        #('photovoltaics, electric installation for 570kWp module, open ground' (unit, GLO, None),'772b78c3f69621d3c9927903c8b56b37')    
        # photovoltaics_electric_installation_570kWpmodule = background_db.get("772b78c3f69621d3c9927903c8b56b37")
        
        # photovoltaics_electric_installation_570kWpmodule["name"]
        # photovoltaics_electric_installation_570kWpmodule["location"]
        
        photovoltaics_electric_installation_570kWpmodule= background_db.get([act for act in background_db if act["name"] =='photovoltaics, electric installation for 570kWp module, open ground'and act["location"] == 'GLO'][0]["code"])
        
        #electric installation 
        list_inputs_electric_instal_id = []
        list_inputs_electric_instal_weights = []
        
        for exc in list(photovoltaics_electric_installation_570kWpmodule.exchanges()):
          
                input_= background_db.get(exc["input"][1])
                input_name = input_["name"]
                input_loc = input_["location"]
                
                print(input_name,input_loc, exc["amount"],exc["unit"])
                
                list_inputs_electric_instal_id.append(input_.id)
                if exc["amount"]>0 and exc["unit"]=="kilogram":
                    list_inputs_electric_instal_weights.append(exc["amount"])
            
        
        total_weight_electric_installation = sum(list_inputs_electric_instal_weights)
        
        mass_to_power_electric_installation = total_weight_electric_installation/570

        
        

        
        
        """Inverter"""
        
        
        
        inverter_500_rer= background_db.get([act for act in background_db if act["name"] =='inverter production, 500kW' and act["location"] == 'RER'][0]["code"])
        
        # This was done to rescale to 1 kg of inverter
        
        list_inputs_inverter_id = []
        list_inputs_inverter_weights = []
        df_=pd.DataFrame({'input':[], 
                'amount':[], 
                'unit':[] } )
        
        # Calculate total weightt
        
        
        for exc in list(inverter_500_rer.exchanges()):
          
                input_= background_db.get(exc["input"][1])
                input_name = input_["name"]
                input_loc = input_["location"]
                
                #print(input_name,input_loc, exc["amount"],exc["unit"])
                df_.loc[len(df_.index)] = [ input_name, exc["amount"], exc["unit"]]  
        
                list_inputs_inverter_id.append(input_.id)
                if exc["amount"]>0 and exc["type"]!="production" and exc["unit"]=="kilogram":
                    list_inputs_inverter_weights.append(exc["amount"])
        
        weigth_500kWrer= df_[(df_.unit == 'kilogram')&(df_.amount > 0)].amount.drop_duplicates().sum()
        
        weigth_500kWrer/500  # 5.97 kg inverter per kWP
        

        
        # Collect inverter_specificweight kg.kWp
        inverter_specificweight = weigth_500kWrer/500
        
        
        
        
        
        """ Modification of the mounting strucuture"""
        
        
        
        
        
        
        
    parameters_database = {"original_surface_panel_single":original_surface_panel_single,
                          "original_surface_panel_multi":original_surface_panel_multi,
                          "original_surface_panel_cis":original_surface_panel_cis,
                          "original_surface_panel_cdte":original_surface_panel_cdte,
                          "inverter_specificweight":inverter_specificweight,
                          "mass_to_power_electric_installation":mass_to_power_electric_installation,
                          "new_efficiency_single":0.194,
                          "new_efficiency_multi":0.181,
                          "new_efficiency_cis": 0.156,
                          "new_efficiency_cdte":0.175,
                          "lifetime":30}     
        
        
        
        
        
    return foregroundAVS,parameters_database
       
        
        