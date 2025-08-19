# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 22:22:52 2024

@author: pjouannais

FOR GSA. The only difference with the classic file is the parameterization dictionaries.

Contains the functions that:
    -collect databases pass activities 
    -Build the parameterization dictionary 
    -Passes everyting to global  to be used in the main script
    
Alternative parametrization can be done by changing the parametrization dictionaries from line 1971.
    


"""

import pandas as pd
import decimal
from random import *
import pstats
from itertools import *
import itertools
from math import*
import csv
import copy
import numpy as np
import random


import math

import inspect

from scipy.stats.qmc import LatinHypercube



import bw_processing as bwp

import sys






def load_activities_dict_andfunctions(foregroundAVS,background_db,Biosphere_db,additional_biosphere_db,global_vars):

    


    wheat_fr_ref = foregroundAVS.get("wheat_fr_ref")
    soy_ch_ref = foregroundAVS.get("soy_ch_ref")
    alfalfa_ch_ref = foregroundAVS.get("alfalfa_ch_ref")
    maize_ch_ref = foregroundAVS.get("maize_ch_ref")

    AVS_elec_main = foregroundAVS.get("AVS_elec_main")
    AVS_elec_main_single = foregroundAVS.get("AVS_elec_main_single")
    AVS_elec_main_multi = foregroundAVS.get("AVS_elec_main_multi")
    AVS_elec_main_cis = foregroundAVS.get("AVS_elec_main_cis")
    AVS_elec_main_cdte = foregroundAVS.get("AVS_elec_main_cdte")
    
    AVS_crop_main = foregroundAVS.get("AVS_crop_main")
    AVS_crop_main_single = foregroundAVS.get("AVS_crop_main_single")
    AVS_crop_main_multi = foregroundAVS.get("AVS_crop_main_multi")
    AVS_crop_main_cis = foregroundAVS.get("AVS_crop_main_cis")
    AVS_crop_main_cdte = foregroundAVS.get("AVS_crop_main_cdte")
    
    
    PV_ref  = foregroundAVS.get("PV_ref")
    PV_ref_single  = foregroundAVS.get("PV_ref_single")
    PV_ref_multi  = foregroundAVS.get("PV_ref_multi")
    PV_ref_cis  = foregroundAVS.get("PV_ref_cis")
    PV_ref_cdte  = foregroundAVS.get("PV_ref_cdte")



    LUCmarket_AVS = foregroundAVS.get("LUCmarket_AVS")
    LUCmarket_PVref= foregroundAVS.get("LUCmarket_PVref")
    LUCmarket_cropref= foregroundAVS.get("LUCmarket_cropref")
    iLUCspecificCO2eq= additional_biosphere_db.get("iLUCspecificCO2eq")
    iluc=foregroundAVS.get("iluc")
    

    
    # Get the outputs of these reference crop activities
    output_wheat = [exc["amount"] for exc in list(wheat_fr_ref.exchanges()) if exc["type"]=="production"][0]
    output_soy = [exc["amount"] for exc in list(soy_ch_ref.exchanges()) if exc["type"]=="production"][0]
    output_alfalfa = [exc["amount"] for exc in list(alfalfa_ch_ref.exchanges()) if exc["type"]=="production"][0]
    output_maize = [exc["amount"] for exc in list(maize_ch_ref.exchanges()) if exc["type"]=="production"][0]
    
    
    
    
    
    wheat_fr_AVS_crop_main = foregroundAVS.get("wheat_fr_AVS_crop_main")
    soy_ch_AVS_crop_main = foregroundAVS.get("soy_ch_AVS_crop_main")
    alfalfa_ch_AVS_crop_main = foregroundAVS.get("alfalfa_ch_AVS_crop_main")
    maize_ch_AVS_crop_main = foregroundAVS.get("maize_ch_AVS_crop_main")
    
    wheat_fr_AVS_elec_main = foregroundAVS.get("wheat_fr_AVS_elec_main")
    soy_ch_AVS_elec_main = foregroundAVS.get("soy_ch_AVS_elec_main")
    alfalfa_ch_AVS_elec_main = foregroundAVS.get("alfalfa_ch_AVS_elec_main")
    maize_ch_AVS_elec_main = foregroundAVS.get("maize_ch_AVS_elec_main")
    
    
    c_soil_accu = foregroundAVS.get("c_soil_accu")
    
    # Market elec
    
    
    
    #elec_marginal_fr = background_db.get("a3b594fa27de840e85cb577a3d63d11a"),
    
    elec_marginal_fr= background_db.get([act for act in background_db if act["name"] =='market for electricity, high voltage' and act["location"] =="FR"][0]["code"])
    
    elec_marginal_fr_copy=foregroundAVS.get("elec_marginal_fr_copy")
    
    elec_marginal_fr_current_copy = foregroundAVS.get("elec_marginal_fr_current_copy")
    elec_marginal_it_current_copy = foregroundAVS.get("elec_marginal_it_current_copy")
    elec_marginal_es_current_copy = foregroundAVS.get("elec_marginal_es_current_copy")

    # Inverter
    
    
    
    mark_inv_500kW_kg=foregroundAVS.get("market_for_inverter_500kW_kg")
    
    # Install 1 m2 PV
    pv_insta_PV=foregroundAVS.get("photovoltaicmono_installation_perm2panel_PV")
    
    pv_insta_PV_single=foregroundAVS.get("photovoltaicmono_installation_perm2panel_PV_2")

    pv_insta_PV_multi=foregroundAVS.get("photovoltaicmulti_installation_perm2panel_PV")
    
    pv_insta_PV_cis=foregroundAVS.get("photovoltaicCIS_installation_perm2panel_PV")

    pv_insta_PV_cdte=foregroundAVS.get("photovoltaicCdte_installation_perm2panel_PV")
    
    
    pv_insta_AVS=foregroundAVS.get("photovoltaicmono_installation_perm2panel_AVS")
    
    pv_insta_AVS_single=foregroundAVS.get("photovoltaicmono_installation_perm2panel_AVS_2")
    
    pv_insta_AVS_multi=foregroundAVS.get("photovoltaicmulti_installation_perm2panel_AVS")
  
    pv_insta_AVS_cis=foregroundAVS.get("photovoltaicCIS_installation_perm2panel_AVS")
    
    pv_insta_AVS_cdte=foregroundAVS.get("photovoltaicCdte_installation_perm2panel_AVS")
    
    
    
    
    # Electric installation
    elecinstakg= foregroundAVS.get('electricpvinstallation_kg')
    
    
    # Input of aluminium in the PV panel productions
    
    
    #'photovoltaic panel production, single-Si wafer' (square meter, RER, None),
    #pvpanel_prod_rer=background_db.get('7ef0c463fcc7c391e9676e53743b977f'),
    
    
    pvpanel_prod_rer=background_db.get([act for act in background_db if act["name"] =='photovoltaic panel production, single-Si wafer' and act["location"] =='RER'][0]["code"])
    
    # 'photovoltaic panel production, single-Si wafer' (square meter, RoW, None),
    
    #pvpanel_prod_row=background_db.get('9e0b81cf2d44559f13849f03e7b3344d'),
    
    pvpanel_prod_row=background_db.get([act for act in background_db if act["name"] =='photovoltaic panel production, single-Si wafer' and act["location"] =='RoW'][0]["code"])
   
    pvpanel_prod_multi_row=background_db.get([act for act in background_db if act["name"] =='photovoltaic panel production, multi-Si wafer' and act["location"] =='RoW'][0]["code"])
   
    pvpanel_prod_multi_rer=background_db.get([act for act in background_db if act["name"] =='photovoltaic panel production, multi-Si wafer' and act["location"] =='RER'][0]["code"])
    
    

    #'market for aluminium alloy, AlMg3' (kilogram, GLO, None),
    # Same for RoW Panel
    
    #aluminium_panel=background_db.get("8a81f71bde65e4274ff4407cdf0c6320"),
    
    aluminium_panel=background_db.get([act for act in background_db if act["name"] =='market for aluminium alloy, AlMg3' and act["location"] =='GLO'][0]["code"])
    
    
    
    #wafer_row=background_db.get("69e0ae62d7d13ca6270e9634a0c40374"),
   
    # mono Si
    wafer_row=background_db.get([act for act in background_db if act["name"] =='photovoltaic cell production, single-Si wafer' and act["location"] =='RoW'][0]["code"])
    

    

    wafer_rer=background_db.get([act for act in background_db if act["name"] =='photovoltaic cell production, single-Si wafer' and act["location"] =='RER'][0]["code"])
    
    # Multi Si
    
    #'photovoltaic cell production, multi-Si wafer' (square meter, RER, None), 
    wafer_multisi_rer=background_db.get([act for act in background_db if act["name"] =='photovoltaic cell production, multi-Si wafer' and act["location"] =='RER'][0]["code"])
    wafer_multisi_row=background_db.get([act for act in background_db if act["name"] =='photovoltaic cell production, multi-Si wafer' and act["location"] =='RoW'][0]["code"])



   # Common to wafer for mono and multi Si
    
    elec_wafer_nz=background_db.get([act for act in background_db if act["name"] =='market for electricity, medium voltage' and act["location"] =='NZ'][0]["code"])
    elec_wafer_rla=background_db.get([act for act in background_db if act["name"] =='market group for electricity, medium voltage' and act["location"] =='RLA'][0]["code"])
    

    elec_wafer_raf=background_db.get([act for act in background_db if act["name"] =='market group for electricity, medium voltage' and act["location"] =='RAF'][0]["code"])
    elec_wafer_au=background_db.get([act for act in background_db if act["name"] =='market for electricity, medium voltage' and act["location"] =='AU'][0]["code"])
    elec_wafer_ci=background_db.get([act for act in background_db if act["name"] =='market for electricity, medium voltage' and act["location"] =='CI'][0]["code"])
    elec_wafer_rna=background_db.get([act for act in background_db if act["name"] =='market group for electricity, medium voltage' and act["location"] =='RNA'][0]["code"])
    elec_wafer_ras=background_db.get([act for act in background_db if act["name"] =='market group for electricity, medium voltage' and act["location"] =='RAS'][0]["code"])
    
    
    
    
    # RER
    def find_the_right_location_because_premise_is_strange_here(activity,medium_or_high,background):
        """Finds the medium or high voltage input to the activity as it is sometimes EUR or RER"""
        for exc in list(activity.exchanges()):
            if exc["type"]=="technosphere":
                # print(exc)
                
                #print(exc["input"])
                #print("ee")
                inp = background.get(exc["input"][1])
                if inp["name"]=="market group for electricity, "+medium_or_high:
                    if inp["location"]=="RER":
                        return "RER"
                    elif inp["location"]=="EUR":
                        return "EUR"
                    
                    elif inp["location"]=="WEU":
                        return "WEU"
                    
                    else:
                        sys.exit("wrong location elec")
    
    
    elec_wafer_rer_single=background_db.get([act for act in background_db if act["name"] =='market group for electricity, medium voltage' and act["location"] =='RER'][0]["code"])
    
    #For some reason, for multi-si , the electricty location is sometimes noted as EUR instead of RER in SOME premise databases.
    
    # if len([act for act in background_db if act["name"] =='market group for electricity, medium voltage' and act["location"] =='EUR'])!=0:
    #     elec_wafer_rer_multi=background_db.get([act for act in background_db if act["name"] =='market group for electricity, medium voltage' and act["location"] =='EUR'][0]["code"])
    # else: 
    #     elec_wafer_rer_multi=background_db.get([act for act in background_db if act["name"] =='market group for electricity, medium voltage' and act["location"] =='RER'][0]["code"])


    elec_wafer_rer_multi = background_db.get([act for act in background_db if act["name"] =='market group for electricity, medium voltage' and act["location"] ==find_the_right_location_because_premise_is_strange_here(wafer_multisi_rer,"medium voltage",background_db)][0]["code"]) 
    
    # Silicon prod
    

            
    si_sg_row=background_db.get([act for act in background_db if act["name"] =='silicon production, solar grade, modified Siemens process' and act["location"] =='RoW'][0]["code"])
    
    
    
    si_sg_rer=background_db.get([act for act in background_db if act["name"] =='silicon production, solar grade, modified Siemens process' and act["location"] =='RER'][0]["code"])
    
    
    elec_sili_raf=background_db.get([act for act in background_db if act["name"] =='market group for electricity, high voltage' and act["location"] =='RAF'][0]["code"])
    
    
    elec_sili_au=background_db.get([act for act in background_db if act["name"] =='market for electricity, high voltage' and act["location"] =='AU'][0]["code"])
    
    elec_sili_ci=background_db.get([act for act in background_db if act["name"] =='market for electricity, high voltage' and act["location"] =='CI'][0]["code"])
    
    elec_sili_nz=background_db.get([act for act in background_db if act["name"] =='market for electricity, high voltage' and act["location"] =='NZ'][0]["code"])
    
    elec_sili_ras=background_db.get([act for act in background_db if act["name"] =='market group for electricity, high voltage' and act["location"] =='RAS'][0]["code"])
    
    elec_sili_rna=background_db.get([act for act in background_db if act["name"] =='market group for electricity, high voltage' and act["location"] =='RNA'][0]["code"])
    
    elec_sili_rla=background_db.get([act for act in background_db if act["name"] =='market group for electricity, high voltage' and act["location"] =='RLA'][0]["code"])
    
    
    
    # Hydroelectricity to remove in the electricity input to the silicon production
    
    electricity_hydro_row_si_sg = background_db.get([act for act in background_db if act["name"] =='electricity production, hydro, run-of-river'and act["location"] =='RoW'][0]["code"])
    
    


    
    
    # if len([act for act in background_db if act["name"] =='market group for electricity, high voltage'and act["location"] =='RER'])!=0:
    #     electricity_rer_si_sg=background_db.get([act for act in background_db if act["name"] =='market group for electricity, high voltage' and act["location"] =='RER'][0]["code"])
    # else: 
    #     electricity_rer_si_sg=background_db.get([act for act in background_db if act["name"] =='market group for electricity, high voltage' and act["location"] =='EUR'][0]["code"])

    electricity_rer_si_sg=background_db.get([act for act in background_db if act["name"] =='market group for electricity, high voltage' and act["location"] ==find_the_right_location_because_premise_is_strange_here(si_sg_rer,"high voltage",background_db)][0]["code"])

    
    
    
    
    mount_system_AVS=foregroundAVS.get("mountingsystem_modif_AVS")
    
    
    
    mount_system_PV=foregroundAVS.get("mountingsystem_modif_PV")
    # for exc in list(mount_system_PV.exchanges(),),=
    #     print(exc),
    
    
    #concrete_mount=background_db.get("bc8c7359f05890c22e37ab303bdd987a")

    
    concrete_mount=background_db.get([act for act in background_db if act["name"] =='market group for concrete, normal strength' and act["location"] =='GLO'][0]["code"])
    
    #concrete_mount_waste_1=background_db.get("e399ae8908f7a23700aad8c833ece1ca")
    
    concrete_mount_waste_1=background_db.get([act for act in background_db if act["name"] =='market for waste reinforced concrete'and act["location"] =='CH'][0]["code"])
    
    
    #concrete_mount_waste_2=background_db.get("5df13a7c86091d82d261b14090992eae")
    
    concrete_mount_waste_2=background_db.get([act for act in background_db if act["name"] =='market for waste reinforced concrete'and act["location"] =='Europe without Switzerland'][0]["code"])
    
    
    
    #concrete_mount_waste_3=background_db.get("d751bebe3aa8f622fc868d3b6a4ea6c4")
    concrete_mount_waste_3=background_db.get([act for act in background_db if act["name"] =='market for waste reinforced concrete'and act["location"] =='RoW'][0]["code"])
    
    
    # aluminium_mount =background_db.get("b2a028e18853749b07a3631ef6eccd96") 
    
    
    
    aluminium_mount=background_db.get([act for act in background_db if act["name"] =='market for aluminium, wrought alloy'and act["location"] =='GLO'][0]["code"])
    
    # alu_extru_mount =background_db.get("0d52d26769eb8011ebc49e09efb7259a") 
    
    
    
    alu_extru_mount=background_db.get([act for act in background_db if act["name"] =='market for section bar extrusion, aluminium'and act["location"] =='GLO'][0]["code"])
    
    
    # alu_mount_scrap_1 =background_db.get("238a2639dfe3ecbcd10a54426cd01094") 
    alu_mount_scrap_1=background_db.get([act for act in background_db if act["name"] =='market for scrap aluminium'and act["location"] =='Europe without Switzerland'][0]["code"])
    
    
    
    # alu_mount_scrap_2 =background_db.get("b487525640db147e00a2ece275262d59") 
    
    
    alu_mount_scrap_2=background_db.get([act for act in background_db if act["name"] =='market for scrap aluminium' and act["location"] =='RoW'][0]["code"])
    
    
    
    # alu_mount_scrap_3 =background_db.get("b7acf0e07b901a83ab9071d87e881af2") 
    
    alu_mount_scrap_3=background_db.get([act for act in background_db if act["name"] =='market for scrap aluminium' and act["location"] =='CH'][0]["code"])
    
    
    # reinf_steel_mount =background_db.get("1737592f8b159167b376ff1ffe485e7e") 
    
    
    reinf_steel_mount=background_db.get([act for act in background_db if act["name"] =='market for reinforcing steel' and act["location"] =='GLO'][0]["code"])
    
    
    # chrom_steel_mount =background_db.get("ecc7db7759925cf080ec73926ca4470e") 
    
    
    chrom_steel_mount=background_db.get([act for act in background_db if act["name"] =='market for steel, chromium steel 18/8, hot rolled' and act["location"] =='GLO'][0]["code"])
    
    
    # steel_rolling_mount =background_db.get("a19e6d008c28760e4872a5e7e001be0f") 
    
    steel_rolling_mount=background_db.get([act for act in background_db if act["name"] =='market for section bar rolling, steel' and act["location"] =='GLO'][0]["code"])
    
    
    
    # wire_mount=background_db.get("bc2fb8454c1dcf38750e1272ee1aae10")
    
    
    wire_mount=background_db.get([act for act in background_db if act["name"] =='market for wire drawing, steel' and act["location"] =='GLO'][0]["code"])
    
    
    # steel_mount_scrap_1=background_db.get("d8deb03d160b32aff1962dd626878c1e")
    
    steel_mount_scrap_1=background_db.get([act for act in background_db if act["name"] =='market for scrap steel' and act["location"] =='RoW'][0]["code"])
    
    
    
    # steel_mount_scrap_2=background_db.get("b9e7ab6f52bf1a1c2079b7fd6da7139e")
    
    steel_mount_scrap_2=background_db.get([act for act in background_db if act["name"] =='market for scrap steel' and act["location"] =='Europe without Switzerland'][0]["code"])
    
    
    
    # steel_mount_scrap_3=background_db.get("f95f81d06894ce8cd3992b22592a7a2e")
    steel_mount_scrap_3=background_db.get([act for act in background_db if act["name"] =='market for scrap steel' and act["location"] =='CH'][0]["code"])
    
    
    # Corrugated carb box
    
    # cor_box_mount_1=background_db.get("3c297273fd9dd888ca5f562b60633b1c")
    cor_box_mount_1=background_db.get([act for act in background_db if act["name"] =='market for corrugated board box' and act["location"] =='CA-QC'][0]["code"])
    
    
    
    # cor_box_mount_2=background_db.get("88b4c06b619444f94dbadd73ba25a99a")
    cor_box_mount_2=background_db.get([act for act in background_db if act["name"] =='market for corrugated board box' and act["location"] =='RoW'][0]["code"])
    
    
    # cor_box_mount_3=background_db.get("bc6d87ea5c33af362a873002e89d0a67")
    
    cor_box_mount_3=background_db.get([act for act in background_db if act["name"] =='market for corrugated board box' and act["location"] =='RER'][0]["code"])
    


    
    # cor_box_mount_waste_1=background_db.get("0fbef1744ad94621c41e6a29eff631e6")
    cor_box_mount_waste_1=background_db.get([act for act in background_db if act["name"] =='market for waste paperboard, unsorted' and act["location"] =='CH'][0]["code"])
    
    
    
    # cor_box_mount_waste_2=background_db.get("07135843544f495acdec3805593597cd")
    
    cor_box_mount_waste_2=background_db.get([act for act in background_db if act["name"] =='market for waste paperboard, unsorted' and act["location"] =='RoW'][0]["code"])
    
    # cor_box_mount_waste_3=background_db.get("5adbcfff2d45da42b0c894714357be1d")
    
    cor_box_mount_waste_3=background_db.get([act for act in background_db if act["name"] =='market for waste paperboard, unsorted' and act["location"] =='Europe without Switzerland'][0]["code"])
    
    
    # poly_mount_1=background_db.get("fe37424fcb39751b8436bea56516d595")
    
    
    poly_mount_1=background_db.get([act for act in background_db if act["name"] =='market for polyethylene, high density, granulate' and act["location"] =='GLO'][0]["code"])
    
    
    # poly_mount_2=background_db.get("ede9bf12aa723d584c162ce6a6805974")
    poly_mount_2=background_db.get([act for act in background_db if act["name"] =='market for polystyrene, high impact' and act["location"] =='GLO'][0]["code"])
    
    
    
    # poly_mount_waste_1=background_db.get("47e42f31205177ca7ccd475552344b26")
    
    # poly_mount_waste_1["name"]
    # poly_mount_waste_1["location"]
    
    poly_mount_waste_1=background_db.get([act for act in background_db if act["name"] =='market for waste polyethylene/polypropylene product' and act["location"] =='CH'][0]["code"])
    
    
    # poly_mount_waste_2=background_db.get("429bdbc2829990503a76dcae0b79e98d")
    # poly_mount_waste_2["name"]
    # poly_mount_waste_2["location"]
    
    poly_mount_waste_2=background_db.get([act for act in background_db if act["name"] =='market for waste polyethylene/polypropylene product' and act["location"] =='Europe without Switzerland'][0]["code"])
    
    
    
    # poly_mount_waste_3=background_db.get("96206cf1b984ab9666c6c84ad37a9f62")
    # poly_mount_waste_3["name"]
    # poly_mount_waste_3["location"]
    
    poly_mount_waste_3=background_db.get([act for act in background_db if act["name"] =='market for waste polyethylene/polypropylene product' and act["location"] =='RoW'][0]["code"])
    
    
    
    # poly_mount_waste_4=background_db.get("5db45c95744227b5dbf5368b217666d7")
    
    # poly_mount_waste_4["name"]
    # poly_mount_waste_4["location"]
    
    poly_mount_waste_4=background_db.get([act for act in background_db if act["name"] =='market for waste polystyrene isolation, flame-retardant' and act["location"] =='RoW'][0]["code"])
    
    
    # poly_mount_waste_5=background_db.get("4ab43cbf5ef6529a43b65011a45b4352")
    
    
    # poly_mount_waste_5["name"]
    # poly_mount_waste_5["location"]
    
    poly_mount_waste_5=background_db.get([act for act in background_db if act["name"] =='market for waste polystyrene isolation, flame-retardant' and act["location"] =='Europe without Switzerland'][0]["code"])
    
    
    # poly_mount_waste_6=background_db.get("ada15830ef22c5f76c41c633cafe81aa")
    # poly_mount_waste_6["name"]
    # poly_mount_waste_6["location"]
    
    poly_mount_waste_6=background_db.get([act for act in background_db if act["name"] =='market for waste polystyrene isolation, flame-retardant' and act["location"] =='CH'][0]["code"])
    
    
    
    # zinc_coat_mount_1=background_db.get("76c32858d64d75927d5a94a5a8683571") 
    
    # zinc_coat_mount_1["name"]
    # zinc_coat_mount_1["location"]
    
    zinc_coat_mount_1=background_db.get([act for act in background_db if act["name"] =='market for zinc coat, coils' and act["location"] =='GLO'][0]["code"])
    
    
    # zinc_coat_mount_2=background_db.get("00d441215fd227f4563ff6c2f4ae3f02") 
    
    # zinc_coat_mount_2["name"]
    # zinc_coat_mount_2["location"]
    
    zinc_coat_mount_2=background_db.get([act for act in background_db if act["name"] =='market for zinc coat, pieces' and act["location"] =='GLO'][0]["code"])
    
    
    # Maize
    
    maize_ch = background_db.get([act for act in background_db if "maize grain production, Swiss integrated production" in act["name"] and act["location"]== "CH"][0]["code"])

    set_activity_exchanges_as_variables(maize_ch,background_db,Biosphere_db,"maize",False,True)

    
    # market_for_maize_seed_swiss_integrated_production_for_sowing_maize
    # market_for_packaging_for_pesticides_maize
    # market_for_ammonium_sulfate_maize
    # market_for_metolachlor_maize
    # combine_harvesting_maize
    # fertilising_by_broadcaster_maize
    # market_for_inorganic_nitrogen_fertiliser_as_n_maize
    # market_for_glyphosate_maize
    # market_for_urea_maize
    # market_for_pesticide_unspecified_maize
    # market_for_packaging_for_fertilisers_maize
    # market_for_phosphate_rock_beneficiated_maize
    # market_for_inorganic_phosphorus_fertiliser_as_p2o5_maize
    # hoeing_maize
    # market_for_irrigation_maize
    # green_manure_growing_swiss_integrated_production_until_april_maize
    # tillage_harrowing_by_spring_tine_harrow_maize
    # liquid_manure_spreading_by_vacuum_tanker_maize
    # sowing_maize
    # market_for_transport_tractor_and_trailer_agricultural_maize
    # tillage_ploughing_maize
    # solid_manure_loading_and_spreading_by_hydraulic_loader_and_spreader_maize
    # application_of_plant_protection_product_by_field_sprayer_maize
    # drying_of_maize_straw_and_whole_plant_maize
    # market_for_atrazine_maize
    # market_for_ammonium_nitrate_maize
    # glyphosate_maizesoilagricultural
    # atrazine_maizesoilagricultural
    # chromium_ion_maizewaterground
    # mercury_maizewaterground
    # lead_maizesoilagricultural
    # zinc_maizesoilagricultural
    # nitrate_maizewaterground
    # copper_ion_maizewatersurfacewater
    # chromium_maizesoilagricultural
    # carbon_dioxide_fossil_maizeairnonurbanairorfromhighstacks
    # water_maizewatersurfacewater
    # cadmium_maizesoilagricultural
    # water_maizewaterground
    # metolachlor_maizesoilagricultural
    # copper_ion_maizewaterground
    # nitrogen_oxides_maizeairnonurbanairorfromhighstacks
    # zinc_ion_maizewaterground
    # lead_maizewaterground
    # water_maizeair
    # zinc_ion_maizewatersurfacewater
    # cadmium_ion_maizewatersurfacewater
    # phosphate_maizewatersurfacewater
    # copper_maizesoilagricultural
    # chromium_ion_maizewatersurfacewater
    # energy_gross_calorific_value_in_biomass_maizenaturalresourcebiotic
    # dinitrogen_monoxide_maizeairnonurbanairorfromhighstacks
    # ammonia_maizeairnonurbanairorfromhighstacks
    # carbon_dioxide_in_air_maizenaturalresourceinair
    # lead_maizewatersurfacewater
    # nickel_maizesoilagricultural
    # transformation_from_annual_crop_non_irrigated_intensive_maizenaturalresourceland
    # mercury_maizesoilagricultural
    # phosphate_maizewaterground
    # occupation_annual_crop_non_irrigated_intensive_maizenaturalresourceland
    # cadmium_ion_maizewaterground
    # transformation_to_annual_crop_non_irrigated_intensive_maizenaturalresourceland
    # phosphorus_maizewatersurfacewater
    # mercury_maizewatersurfacewater
    # nickel_ion_maizewatersurfacewater
    # market_group_for_electricity_low_voltage_maize
    






    
    # Alfalfa input and ouput
    
    
    
    # Exchange= 6.6662e-05 hectare 'fertilising, by broadcaster' (hectare, CH, None), to 'alfalfa-grass mixture production, Swiss integrated production' (kilogram, CH, None),>
    # fertilising, by broadcaster
    # 302a1e9ca5f7b39a5c6f4a7e27bd56c0
    # fert_broadcaster_ch=background_db.get("302a1e9ca5f7b39a5c6f4a7e27bd56c0"),
    
    # fert_broadcaster_ch["name"]
    # fert_broadcaster_ch["location"]
    
    fert_broadcaster_ch=background_db.get([act for act in background_db if act["name"] =='fertilising, by broadcaster' and act["location"] =='CH'][0]["code"])
    
    
    
    
    # Exchange= 0.0042024 kilogram 'market for inorganic phosphorus fertiliser, as P2O5' (kilogram, CH, None), to 'alfalfa-grass mixture production, Swiss integrated production' (kilogram, CH, None),>
    # market for inorganic phosphorus fertiliser, as P2O5
    # adf4a377ba3fca72cb0d2090757a0bb1
    
    # ino_P205_ch=background_db.get("adf4a377ba3fca72cb0d2090757a0bb1"),
    
    
    # ino_P205_ch["name"]
    # ino_P205_ch["location"]
    
    ino_P205_ch=background_db.get([act for act in background_db if act["name"] =='market for inorganic phosphorus fertiliser, as P2O5' and act["location"] =='CH'][0]["code"])
    
    
    # Exchange= 0.0013332 cubic meter 'liquid manure spreading, by vacuum tanker' (cubic meter, CH, None), to 'alfalfa-grass mixture production, Swiss integrated production' (kilogram, CH, None),>
    # liquid manure spreading, by vacuum tanker
    # f8c1374cd1f5f1f800a873ee4feb590e
    
    # liquidmanure_spr_ch=background_db.get("f8c1374cd1f5f1f800a873ee4feb590e"),
    
    
    # liquidmanure_spr_ch["name"]
    # liquidmanure_spr_ch["location"]
    
    liquidmanure_spr_ch=background_db.get([act for act in background_db if act["name"] =='liquid manure spreading, by vacuum tanker' and act["location"] =='CH'][0]["code"])
    
    
    
    
    
    # Exchange= 0.0084048 kilogram 'market for packaging, for fertilisers' (kilogram, GLO, None), to 'alfalfa-grass mixture production, Swiss integrated production' (kilogram, CH, None),>
    # market for packaging, for fertilisers
    # a9254f81cc8ecaeb85ab44c32b2e02be
    # packaging_fert_glo =background_db.get("a9254f81cc8ecaeb85ab44c32b2e02be"),
    
    
    # packaging_fert_glo["name"]
    # packaging_fert_glo["location"]
    
    packaging_fert_glo=background_db.get([act for act in background_db if act["name"] =='market for packaging, for fertilisers' and act["location"] =='GLO'][0]["code"])
    
    
    # Exchange= 0.66662 kilogram 'solid manure loading and spreading, by hydraulic loader and spreader' (kilogram, CH, None), to 'alfalfa-grass mixture production, Swiss integrated production' (kilogram, CH, None),>
    # solid manure loading and spreading, by hydraulic loader and spreader
    # d9355b12099cf24dd94f024cfca32d35
    # solidmanure_spreading_ch=background_db.get("d9355b12099cf24dd94f024cfca32d35"),
    
    # solidmanure_spreading_ch["name"]
    # solidmanure_spreading_ch["location"]
    
    solidmanure_spreading_ch=background_db.get([act for act in background_db if act["name"] =='solid manure loading and spreading, by hydraulic loader and spreader' and act["location"] =='CH'][0]["code"])
    
    
    
    # Machinery
    # fodder_loading_ch= background_db.get("10f89cf01395e21928b4137637b523a8"),
    
    
    # fodder_loading_ch["name"]
    # fodder_loading_ch["location"]
    
    fodder_loading_ch=background_db.get([act for act in background_db if act["name"] =='fodder loading, by self-loading trailer' and act["location"] =='CH'][0]["code"])
    
    
    
    # rotary_mower_ch= background_db.get("16735e3c568e791b8074941193b3ba12"),
    
    # rotary_mower_ch["name"]
    # rotary_mower_ch["location"]
    
    rotary_mower_ch=background_db.get([act for act in background_db if act["name"] =='mowing, by rotary mower' and act["location"] =='CH'][0]["code"])
    
    
    
    
    # sowing_ch= background_db.get("324c1476b808b7149f0ed9d7c8f5afed"),
    
    
    # sowing_ch["name"]
    # sowing_ch["location"]
    
    sowing_ch=background_db.get([act for act in background_db if act["name"] =='sowing' and act["location"] =='CH'][0]["code"])
    
    
    
    # tillage_rotary_spring_tine_ch= background_db.get("d408ce3256fb75715ddad41fcc92cebc"),
    
    # tillage_rotary_spring_tine_ch["name"]
    # tillage_rotary_spring_tine_ch["location"]
    
    tillage_rotary_spring_tine_ch=background_db.get([act for act in background_db if act["name"] =='tillage, harrowing, by spring tine harrow' and act["location"] =='CH'][0]["code"])
    
    
    # tillage_ploughing_ch=background_db.get("457c5b6648cad5aecc781c9f5d5eca55"),
    
    
    # tillage_ploughing_ch["name"]
    # tillage_ploughing_ch["location"]
    
    tillage_ploughing_ch=background_db.get([act for act in background_db if act["name"] =='tillage, ploughing' and act["location"] =='CH'][0]["code"])
    
    
    # tillage_rolling_ch=background_db.get("11374fa3faed9fe1cbab183ece4fa00e"),
    
    # tillage_rolling_ch["name"]
    # tillage_rolling_ch["location"]
    
    tillage_rolling_ch=background_db.get([act for act in background_db if act["name"] =='tillage, rolling' and act["location"] =='CH'][0]["code"])
    
    
    
    
    
    
    # combine_harvesting_ch=background_db.get("967325bd96bf98cd0d180527060236f9"),
    
    # combine_harvesting_ch["name"]
    # combine_harvesting_ch["location"]
    
    combine_harvesting_ch = background_db.get([act for act in background_db if act["name"] =='combine harvesting' and act["location"] =='CH'][0]["code"])
    
    
    
    #"""Soy_ch"""
    
    
    # for exc in list(soy_ch_AVS.exchanges(),),=
    #     if exc["type"]=="biosphere" and "Water" in exc["name"]=
    #         print(exc["name"]),
            
    
    
    # for exc in list(alfalfa_ch_AVS.exchanges(),),=
    #     if exc["type"]=="biosphere" and "Water" in exc["name"]=
    #         print(exc["name"]),
            
            
    # for exc in list(wheat_fr_AVS.exchanges(),),=
    #     if exc["type"]=="biosphere"and "Water" in exc["name"]=
    #         print(exc["name"]),        
    # Exchange= 0.00026044 hectare 'fertilising, by broadcaster' (hectare, CH, None), to 'soybean production' (kilogram, CH, None),>
    # fertilising, by broadcaster
    # 302a1e9ca5f7b39a5c6f4a7e27bd56c0
    # fert_broadcaster_ch=background_db.get("302a1e9ca5f7b39a5c6f4a7e27bd56c0"),
    
    # fert_broadcaster_ch["name"]
    # fert_broadcaster_ch["location"]
    
    fert_broadcaster_ch=background_db.get([act for act in background_db if act["name"] =='fertilising, by broadcaster' and act["location"] =='CH'][0]["code"])
    
    
    
    #Exchange= 0.00026044 hectare 'green manure growing, Swiss integrated production, until January' (hectare, CH, None), to 'soybean production' (kilogram, CH, None),
    #green manure growing, Swiss integrated production, until January
    # green_manure_ch =background_db.get("7dc3dcdf0989c1fe66d1cc94ac057726"),
    
    
    # green_manure_ch["name"]
    # green_manure_ch["location"]
    
    green_manure_ch=background_db.get([act for act in background_db if act["name"] =='green manure growing, Swiss integrated production, until January' and act["location"] =='CH'][0]["code"])
    
    
    
    # Exchange= 0.00066337 kilogram 'nutrient supply from thomas meal' (kilogram, GLO, None), to 'soybean production' (kilogram, CH, None),>
    # nutrient supply from thomas meal
    # 550fc5c30f02c82d6a7f491a6cfbf4c4
    
    
    # nutrient_supply_thomas_meal_ch =background_db.get("550fc5c30f02c82d6a7f491a6cfbf4c4"),
    
    # nutrient_supply_thomas_meal_ch["name"]
    # nutrient_supply_thomas_meal_ch["location"]
    
    nutrient_supply_thomas_meal_ch=background_db.get([act for act in background_db if act["name"] =='nutrient supply from thomas meal' and act["location"] =='GLO'][0]["code"])
    
    
    
    # Exchange= 0.0011157 cubic meter 'liquid manure spreading, by vacuum tanker' (cubic meter, CH, None), to 'soybean production' (kilogram, CH, None),>
    # liquid manure spreading, by vacuum tanker
    # f8c1374cd1f5f1f800a873ee4feb590e
    # liquidmanure_spr_ch =background_db.get("f8c1374cd1f5f1f800a873ee4feb590e"),
    
    # liquidmanure_spr_ch["name"]
    # liquidmanure_spr_ch["location"]
    
    liquidmanure_spr_ch=background_db.get([act for act in background_db if act["name"] =='liquid manure spreading, by vacuum tanker' and act["location"] =='CH'][0]["code"])
    
    
    
    
    # Exchange= 0.03943066 kilogram 'market for packaging, for fertilisers' (kilogram, GLO, None), to 'soybean production' (kilogram, CH, None),>
    # market for packaging, for fertilisers
    # a9254f81cc8ecaeb85ab44c32b2e02be
    # packaging_fert_glo =background_db.get("a9254f81cc8ecaeb85ab44c32b2e02be"),
    
    
    # packaging_fert_glo["name"]
    # packaging_fert_glo["location"]
    
    packaging_fert_glo=background_db.get([act for act in background_db if act["name"] =='market for packaging, for fertilisers' and act["location"] =='GLO'][0]["code"])
    
    
    
    # Exchange= 0.009595625 kilogram 'market for phosphate rock, beneficiated' (kilogram, RER, None), to 'soybean production' (kilogram, CH, None),>
    # market for phosphate rock, beneficiated
    # 6ab3fd3b19187e0406419b03f3b88ff3
    # phosphate_rock_glo=background_db.get("6ab3fd3b19187e0406419b03f3b88ff3"),
    
    
    # phosphate_rock_glo["name"]
    # phosphate_rock_glo["location"]
    
    phosphate_rock_glo=background_db.get([act for act in background_db if act["name"] =='market for phosphate rock, beneficiated' and act["location"] =='RER'][0]["code"])
    
    
    
    # Exchange= 0.0156317446180305 kilogram 'market for potassium chloride' (kilogram, RER, None), to 'soybean production' (kilogram, CH, None),>
    # market for potassium chloride
    # 3c83fdd5c69822caadf209bdbf8a50f3
    # potassium_chloride_rer=background_db.get("3c83fdd5c69822caadf209bdbf8a50f3"),
    
    
    # potassium_chloride_rer["name"]
    # potassium_chloride_rer["location"]
    
    potassium_chloride_rer=background_db.get([act for act in background_db if act["name"] =='market for potassium chloride'  and act["location"] =='RER'][0]["code"])
    
    
    
    # Exchange= 0.00119833261913457 kilogram 'market for potassium sulfate' (kilogram, RER, None), to 'soybean production' (kilogram, CH, None),>
    # market for potassium sulfate
    # 06f1d7f08f051e8e466af8fd4faaadd9
    # potassium_sulfate_rer=background_db.get("06f1d7f08f051e8e466af8fd4faaadd9"),
    
    # potassium_sulfate_rer["name"]
    # potassium_sulfate_rer["location"]
    
    potassium_sulfate_rer=background_db.get([act for act in background_db if act["name"] =='market for potassium sulfate'  and act["location"] =='RER'][0]["code"])
    
    
    
    
    # Exchange= 0.00100714285714286 kilogram 'market for single superphosphate' (kilogram, RER, None), to 'soybean production' (kilogram, CH, None),>
    # market for single superphosphate
    # 61f2763127305038f68860164831764d
    # single_superphosphate_rer=background_db.get("61f2763127305038f68860164831764d"),
    
    
    
    # single_superphosphate_rer["name"]
    # single_superphosphate_rer["location"]
    
    single_superphosphate_rer=background_db.get([act for act in background_db if act["name"] =='market for single superphosphate'  and act["location"] =='RER'][0]["code"])
    
    
    
    
    # Exchange= 0.072403 kilogram 'solid manure loading and spreading, by hydraulic loader and spreader' (kilogram, CH, None), to 'soybean production' (kilogram, CH, None),>
    # solid manure loading and spreading, by hydraulic loader and spreader
    # d9355b12099cf24dd94f024cfca32d35
    # solidmanure_spreading_ch=background_db.get("d9355b12099cf24dd94f024cfca32d35"),
    
    
    
    # solidmanure_spreading_ch["name"]
    # solidmanure_spreading_ch["location"]
    
    solidmanure_spreading_ch=background_db.get([act for act in background_db if act["name"] =='solid manure loading and spreading, by hydraulic loader and spreader'  and act["location"] =='CH'][0]["code"])
    
    
    
    
    # Exchange= 0.0114058695652174 kilogram 'market for triple superphosphate' (kilogram, RER, None), to 'soybean production' (kilogram, CH, None),>
    # market for triple superphosphate
    # 10952b351896a1f0935e0dbd1fc98f49
    # triplesuperphosphate=background_db.get("10952b351896a1f0935e0dbd1fc98f49"),
    
    # triplesuperphosphate["name"]
    # triplesuperphosphate["location"]
    
    triplesuperphosphate=background_db.get([act for act in background_db if act["name"] =='market for triple superphosphate'  and act["location"] =='RER'][0]["code"])
    
    
    
    # Machinery
    # tillage_currying_weeder_ch= background_db.get("9fdce9892f57840bd2fa96d92ab577e3"),
    
    # tillage_currying_weeder_ch["name"]
    # tillage_currying_weeder_ch["location"]
    
    tillage_currying_weeder_ch=background_db.get([act for act in background_db if act["name"] =='tillage, currying, by weeder'  and act["location"] =='CH'][0]["code"])
    
    
    
    # tillage_rotary_spring_tine_ch= background_db.get("d408ce3256fb75715ddad41fcc92cebc"),
    
    # tillage_rotary_spring_tine_ch["name"]
    # tillage_rotary_spring_tine_ch["location"]
    
    tillage_rotary_spring_tine_ch=background_db.get([act for act in background_db if act["name"] =='tillage, harrowing, by spring tine harrow' and act["location"] =='CH'][0]["code"])
    
    
    
    
    # sowing_ch= background_db.get("324c1476b808b7149f0ed9d7c8f5afed"),
    
    # sowing_ch["name"]
    # sowing_ch["location"]
    
    sowing_ch=background_db.get([act for act in background_db if act["name"] =='sowing' and act["location"] =='CH'][0]["code"])
    
    
    
    # combine_harvesting_ch=background_db.get("967325bd96bf98cd0d180527060236f9"),
    
    # combine_harvesting_ch["name"]
    # combine_harvesting_ch["location"]
    
    combine_harvesting_ch=background_db.get([act for act in background_db if act["name"] =='combine harvesting'and act["location"] =='CH'][0]["code"])
    
    
    
    #"""Wheat fr"""
    
    
    # ammonium_nitrate=background_db.get("28acde92cd1aec3b5d5b8f58f50055ba"),
    
    # ammonium_nitrate["name"]
    # ammonium_nitrate["location"]
    
    ammonium_nitrate=background_db.get([act for act in background_db if act["name"] =='market for ammonium nitrate'and act["location"] =='RER'][0]["code"])
    
    
    
    
    # ammonium_sulfate=background_db.get("3e5d910e09135106f6c712c538182b71"),
    
    
    # ammonium_sulfate["name"]
    # ammonium_sulfate["location"]
    
    ammonium_sulfate=background_db.get([act for act in background_db if act["name"] =='market for ammonium sulfate'and act["location"] =='RER'][0]["code"])
    
    
    
    # urea=background_db.get("d729224f52da90228e5c4b7f864303b0"),
    
    # urea["name"]
    # urea["location"]
    
    urea=background_db.get([act for act in background_db if act["name"] =='market for urea'and act["location"] =='RER'][0]["code"])
    
    
    
    # Exchange= 0.00051831 hectare 'market for fertilising, by broadcaster' (hectare, GLO, None), to 'wheat grain production' (kilogram, FR, None),>
    # market for fertilising, by broadcaster
    # 2433809f3b7d23df8625099f845e381
    # fert_broadcaster=background_db.get("2433809f3b7d23df8625099f845e3814"),
    
    # fert_broadcaster["name"]
    # fert_broadcaster["location"]
    
    fert_broadcaster=background_db.get([act for act in background_db if act["name"] =='market for fertilising, by broadcaster'and act["location"] =='GLO'][0]["code"])
    
    
    
    
    # Exchange= 0.0104 kilogram 'market for inorganic phosphorus fertiliser, as P2O5' (kilogram, FR, None), to 'wheat grain production' (kilogram, FR, None),>
    
    
    # ino_P205_fr=background_db.get("12b793a4ed3da8e8e17209b4e788d9c0"),
    
    # ino_P205_fr["name"]
    # ino_P205_fr["location"]
    
    ino_P205_fr=background_db.get([act for act in background_db if act["name"] =='market for inorganic phosphorus fertiliser, as P2O5'and act["location"] =='FR'][0]["code"])
    
    
    
    # Exchange= 3.406e-07 kilogram 'market for organophosphorus-compound, unspecified' (kilogram, GLO, None), to 'wheat grain production' (kilogram, FR, None),>
    # market for organophosphorus-compound, unspecified
    # cbabef8729a2ba6a1286742273fa3a6e
    # org_P205=background_db.get("cbabef8729a2ba6a1286742273fa3a6e"),
    
    # org_P205["name"]
    # org_P205["location"]
    
    org_P205=background_db.get([act for act in background_db if act["name"] =='market for organophosphorus-compound, unspecified'and act["location"] =='GLO'][0]["code"])
    
    
    
    
    # Exchange= 0.094817 kilogram 'market for packaging, for fertilisers' (kilogram, GLO, None), to 'wheat grain production' (kilogram, FR, None),>
    # market for packaging, for fertilisers
    # a9254f81cc8ecaeb85ab44c32b2e02be
    
    # packaging_fert=background_db.get("a9254f81cc8ecaeb85ab44c32b2e02be"),
    
    
    # packaging_fert["name"]
    # packaging_fert["location"]
    
    packaging_fert=background_db.get([act for act in background_db if act["name"] =='market for packaging, for fertilisers'and act["location"] =='GLO'][0]["code"])
    
    
    
    
    
    # Exchange= 0.123183267669125 cubic meter 'market for irrigation' (cubic meter, FR, None), to 'wheat grain production' (kilogram, FR, None),>
    # market for irrigation
    # 432f8fdd083f6730c0f2ae6c8014656e
    # water_market_irrigation=background_db.get("432f8fdd083f6730c0f2ae6c8014656e"),
    
    # water_market_irrigation["name"]
    # water_market_irrigation["location"]
    
    water_market_irrigation=background_db.get([act for act in background_db if act["name"] =='market for irrigation'and act["location"] =='FR'][0]["code"])
    
    
    
    
    
    # Machinery
    # tillage_rotary_harrow_glo= background_db.get("cb6989c6601a864504c47d84b469704a"),
    
    # tillage_rotary_harrow_glo["name"]
    # tillage_rotary_harrow_glo["location"]
    
    tillage_rotary_harrow_glo=background_db.get([act for act in background_db if act["name"] =='market for tillage, harrowing, by rotary harrow'and act["location"] =='GLO'][0]["code"])
    
    
    # tillage_rotary_spring_tine_glo= background_db.get("d34f658ca1260bf83866b037f4c73306"),
    
    
    # tillage_rotary_spring_tine_glo["name"]
    # tillage_rotary_spring_tine_glo["location"]
    
    tillage_rotary_spring_tine_glo=background_db.get([act for act in background_db if act["name"] =='market for tillage, harrowing, by spring tine harrow'and act["location"] =='GLO'][0]["code"])
    
    
    # sowing_glo= background_db.get("bd78d677f0e04fe1038f135719578207"),
    
    # sowing_glo["name"]
    # sowing_glo["location"]
    
    sowing_glo=background_db.get([act for act in background_db if act["name"] =='market for sowing'and act["location"] =='GLO'][0]["code"])
    
    
    
    # tillage_ploughing_glo=background_db.get("bcd18cd662caba362f4dcbc8e05002d7"),
    
    
    # tillage_ploughing_glo["name"]
    # tillage_ploughing_glo["location"]
    
    tillage_ploughing_glo=background_db.get([act for act in background_db if act["name"] =='market for tillage, ploughing'and act["location"] =='GLO'][0]["code"])
    
    
    # combine_harvesting_glo=background_db.get("20966082098d6802dc2d3e1f0346aece"),
    
    
    # combine_harvesting_glo["name"]
    # combine_harvesting_glo["location"]
    
    combine_harvesting_glo=background_db.get([act for act in background_db if act["name"] =='market for combine harvesting'and act["location"] =='GLO'][0]["code"])
    

    #"""Biosphere_db outputs agri"""
    
    # Flow to add for Carbon storage differential
    
    #  ('Carbon dioxide, to soil or biomass stock' (kilogram, None, ('soil',),),,
    #  '375bc95e-6596-4aa1-9716-80ff51b9da77'),
    
    #[(flow,flow["code"]), for flow in Biosphere_db  if "dioxide" in flow["name"]]
    
    # Carbon_dioxide_to_soil_biomass_stock=Biosphere_db.get('375bc95e-6596-4aa1-9716-80ff51b9da77'),
    
    # Carbon_dioxide_to_soil_biomass_stock["name"]
    # Carbon_dioxide_to_soil_biomass_stock["categories"]
    
    
    Carbon_dioxide_to_soil_biomass_stock=Biosphere_db.get([act for act in Biosphere_db if act["name"] =='Carbon dioxide, to soil or biomass stock' and act["categories"] ==('soil',)][0]["code"])
    
    
    
    
    
    # ammonia=Biosphere_db.get("0f440cc0-0f74-446d-99d6-8ff0e97a2444"),
    
    # ammonia["name"]
    # ammonia["categories"]
    
    
    ammonia=Biosphere_db.get([act for act in Biosphere_db if act["name"] =='Ammonia' and act["categories"] ==('air', 'non-urban air or from high stacks')][0]["code"])
    
    
    
    
    # dinitrogen_monoxide=Biosphere_db.get("afd6d670-bbb0-4625-9730-04088a5b035e"),
    
    # dinitrogen_monoxide["name"]
    # dinitrogen_monoxide["categories"]
    
    
    dinitrogen_monoxide=Biosphere_db.get([act for act in Biosphere_db if act["name"] =='Dinitrogen monoxide' and act["categories"] ==('air', 'non-urban air or from high stacks')][0]["code"])
    
    
    # nitrogen_oxide=Biosphere_db.get("77357947-ccc5-438e-9996-95e65e1e1bce"),
    
    # nitrogen_oxide["name"]
    # nitrogen_oxide["categories"]
    
    
    nitrogen_oxide=Biosphere_db.get([act for act in Biosphere_db if act["name"] =='Nitrogen oxides' and act["categories"] ==('air', 'non-urban air or from high stacks')][0]["code"])
    
    
    
    
    # nitrate=Biosphere_db.get("b9291c72-4b1d-4275-8068-4c707dc3ce33"),
    
    # nitrate["name"]
    # nitrate["categories"]
    
    
    nitrate=Biosphere_db.get([act for act in Biosphere_db if act["name"] =='Nitrate' and act["categories"] ==('water', 'ground-')][0]["code"])
    
    
    
    
    # Exchange= 3.1767e-05 kilogram 'Phosphate' (kilogram, None, ('water', 'ground-'),), to 'wheat grain production' (kilogram, FR, None),>
    # Phosphate
    # phosphate_groundwater=Biosphere_db.get("329fc7d8-4011-4327-84e4-34ff76f0e42d"),
    
    # phosphate_groundwater["name"]
    # phosphate_groundwater["categories"]
    
    
    phosphate_groundwater=Biosphere_db.get([act for act in Biosphere_db if act["name"] =='Phosphate' and act["categories"] ==('water', 'ground-')][0]["code"])
    
    
    
    # phosphate_surfacewater["name"]
    # phosphate_surfacewater["categories"]
    
    
    phosphate_surfacewater=Biosphere_db.get([act for act in Biosphere_db if act["name"] =='Phosphate' and act["categories"] ==('water', 'surface water')][0]["code"])
    
    
    # phosphorus_surfacewater=Biosphere_db.get("b2631209-8374-431e-b7d5-56c96c6b6d79"),
    
    # phosphorus_surfacewater["name"]
    # phosphorus_surfacewater["categories"]
    
    
    phosphorus_surfacewater=Biosphere_db.get([act for act in Biosphere_db if act["name"] =='Phosphorus' and act["categories"] ==('water', 'surface water')][0]["code"])
    
    
    
    # Exchange= 0.0978691061631198 cubic meter 'Water' (cubic meter, None, ('air',),), to 'wheat grain production' (kilogram, FR, None),>
    # Water
    # 075e433b-4be4-448e-9510-9a5029c1ce94
    # water_air=Biosphere_db.get("075e433b-4be4-448e-9510-9a5029c1ce94"),
    
    # water_air["name"]
    # water_air["categories"]
    
    
    water_air=Biosphere_db.get([act for act in Biosphere_db if act["name"] =='Water' and act["categories"] ==('air',)][0]["code"])
    
    
    
    # water_ground=Biosphere_db.get("51254820-3456-4373-b7b4-056cf7b16e01")
    
    
    # water_ground["name"]
    # water_ground["categories"]
    
    
    water_ground=Biosphere_db.get([act for act in Biosphere_db if act["name"] =='Water' and act["categories"] ==('water', 'ground-')][0]["code"])
    
    
    
    # water_surface=Biosphere_db.get("db4566b1-bd88-427d-92da-2d25879063b9"),
    
    # water_surface["name"]
    # water_surface["categories"]
    
    
    water_surface=Biosphere_db.get([act for act in Biosphere_db if act["name"] =='Water' and act["categories"] ==('water', 'surface water')][0]["code"])
    
    
    # Exchange= 1.42397953472488 kilogram 'Carbon dioxide, in air' (kilogram, None, ('natural resource', 'in air'),), to 'wheat grain production' (kilogram, FR, None),>
    # Carbon dioxide, in air
    # cc6a1abb-b123-4ca6-8f16-38209df609be
    # carbon_dioxide_air_resource=Biosphere_db.get("cc6a1abb-b123-4ca6-8f16-38209df609be"),
    
    # carbon_dioxide_air_resource["name"]
    # carbon_dioxide_air_resource["categories"]
    
    
    carbon_dioxide_air_resource=Biosphere_db.get([act for act in Biosphere_db if act["name"] =='Carbon dioxide, in air' and act["categories"] ==('natural resource', 'in air')][0]["code"])
    
    
    # Exchange= 15.1230001449585 megajoule 'Energy, gross calorific value, in biomass' (megajoule, None, ('natural resource', 'biotic'),), to 'wheat grain production' (kilogram, FR, None),>
    # Energy, gross calorific value, in biomass
    # 01c12fca-ad8b-4902-8b48-2d5afe3d3a0f
    #energy_inbiomass=Biosphere_db.get("01c12fca-ad8b-4902-8b48-2d5afe3d3a0f"),
    
    
    # energy_inbiomass["name"]
    # energy_inbiomass["categories"]
    
    
    energy_inbiomass=Biosphere_db.get([act for act in Biosphere_db if act["name"] =='Energy, gross calorific value, in biomass' and act["categories"] ==('natural resource', 'biotic')][0]["code"])
    
    
    
    # carbon dioxide fossi from urea (only wheat),
    #carbondioxide_fossil_urea=Biosphere_db.get("aa7cac3a-3625-41d4-bc54-33e2cf11ec46"),
    
    # carbondioxide_fossil_urea["name"]
    # carbondioxide_fossil_urea["categories"]
    
    
    carbondioxide_fossil_urea=Biosphere_db.get([act for act in Biosphere_db if act["name"] =='Carbon dioxide, fossil' and act["categories"] ==('air', 'non-urban air or from high stacks')][0]["code"]) 
    
  
        
  
    
  
   # Get original agricultural activities
   
   
        
    AVS_elec_main_single = foregroundAVS.get("AVS_elec_main_single")
    AVS_elec_main_multi = foregroundAVS.get("AVS_elec_main_multi")
    AVS_elec_main_cis = foregroundAVS.get("AVS_elec_main_cis")
    AVS_elec_main_cdte = foregroundAVS.get("AVS_elec_main_cdte")
    
    AVS_crop_main = foregroundAVS.get("AVS_crop_main")
    AVS_crop_main_single = foregroundAVS.get("AVS_crop_main_single")
    AVS_crop_main_multi = foregroundAVS.get("AVS_crop_main_multi")
    AVS_crop_main_cis = foregroundAVS.get("AVS_crop_main_cis")
    AVS_crop_main_cdte = foregroundAVS.get("AVS_crop_main_cdte")
    
    
    PV_ref  = foregroundAVS.get("PV_ref")
    PV_ref_single  = foregroundAVS.get("PV_ref_single")
    PV_ref_multi  = foregroundAVS.get("PV_ref_multi")
    PV_ref_cis  = foregroundAVS.get("PV_ref_cis")
    PV_ref_cdte  = foregroundAVS.get("PV_ref_cdte")


    local_vars={
    # FG
    "AVS_elec_main" : AVS_elec_main,
    "AVS_elec_main_single" : AVS_elec_main_single,
    "AVS_elec_main_multi" : AVS_elec_main_multi,
    "AVS_elec_main_cis" : AVS_elec_main_cis,
    "AVS_elec_main_cdte" : AVS_elec_main_cdte,
    
    "AVS_crop_main" : AVS_crop_main,
    "AVS_crop_main_single" : AVS_crop_main_single,
    "AVS_crop_main_multi" : AVS_crop_main_multi,
    "AVS_crop_main_cis" : AVS_crop_main_cis,
    "AVS_crop_main_cdte" : AVS_crop_main_cdte,
    
    "PV_ref"  : PV_ref,
    "PV_ref_single" : PV_ref_single,
    "PV_ref_multi" : PV_ref_multi,
    "PV_ref_cis" : PV_ref_cis,
    "PV_ref_cdte" : PV_ref_cdte,
    
    
    
    "LUCmarket_AVS" : LUCmarket_AVS,
    "LUCmarket_PVref": LUCmarket_PVref,
    "LUCmarket_cropref": LUCmarket_cropref,
    "iLUCspecificCO2eq": iLUCspecificCO2eq,
    "iluc":iluc,
    
    "wheat_fr_ref" : wheat_fr_ref,
    "soy_ch_ref" : soy_ch_ref,
    "alfalfa_ch_ref" : alfalfa_ch_ref,
    "maize_ch_ref" :maize_ch_ref,
    
    # Get the outputs of these reference crop activities
    "output_wheat" : output_wheat,
    "output_soy" : output_soy,
    "output_alfalfa" : output_alfalfa,
    "output_maize" : output_maize,
    
    
    
    
    
    "wheat_fr_AVS_crop_main" : wheat_fr_AVS_crop_main,
    "soy_ch_AVS_crop_main" : soy_ch_AVS_crop_main,
    "alfalfa_ch_AVS_crop_main" : alfalfa_ch_AVS_crop_main,
    "maize_ch_AVS_crop_main" : maize_ch_AVS_crop_main,
    
    "wheat_fr_AVS_elec_main" : wheat_fr_AVS_elec_main,
    "soy_ch_AVS_elec_main" : soy_ch_AVS_elec_main,
    "maize_ch_AVS_elec_main" : maize_ch_AVS_elec_main,
    
    
    "c_soil_accu" : c_soil_accu,
    
    # Market elec
    
    
    
    #elec_marginal_fr : background_db.get("a3b594fa27de840e85cb577a3d63d11a"),
    
    "elec_marginal_fr": elec_marginal_fr,
    
    "elec_marginal_fr_copy":elec_marginal_fr_copy,
    "elec_marginal_fr_current_copy":elec_marginal_fr_current_copy,
    "elec_marginal_es_current_copy":elec_marginal_es_current_copy,
    "elec_marginal_it_current_copy":elec_marginal_it_current_copy,
    
    
    
    # Inverter
    
    
    
    "mark_inv_500kW_kg":mark_inv_500kW_kg,
    
    # Install 1 m2 PV
    "pv_insta_PV":pv_insta_PV,
    "pv_insta_PV_single":pv_insta_PV_single,
    "pv_insta_PV_multi":pv_insta_PV_multi,
    "pv_insta_PV_cis":pv_insta_PV_cis,
    "pv_insta_PV_cdte":pv_insta_PV_cdte,

    "pv_insta_AVS":pv_insta_AVS,
    "pv_insta_AVS_single":pv_insta_AVS_single,
    "pv_insta_AVS_multi":pv_insta_AVS_multi,
    "pv_insta_AVS_cis":pv_insta_AVS_cis,
    "pv_insta_AVS_cdte":pv_insta_AVS_cdte,
    
    
    
    
    # Electric installation
    "elecinstakg": elecinstakg,
    
    
    # Input of aluminium in the PV panel productions
    
    
    #'photovoltaic panel production, single-Si wafer' (square meter, RER, None),
    #pvpanel_prod_rer:background_db.get('7ef0c463fcc7c391e9676e53743b977f'),
    
    
    "pvpanel_prod_rer":pvpanel_prod_rer,
    
    # 'photovoltaic panel production, single-Si wafer' (square meter, RoW, None),
    
    #pvpanel_prod_row:background_db.get('9e0b81cf2d44559f13849f03e7b3344d'),
    
    "pvpanel_prod_row":pvpanel_prod_row,
    
    #'market for aluminium alloy, AlMg3' (kilogram, GLO, None),
    # Same for RoW Panel
    
    
    # Multi SI
    
    "pvpanel_prod_multi_row":pvpanel_prod_multi_row,
    
    "pvpanel_prod_multi_rer":pvpanel_prod_multi_rer,

    
    
    #aluminium_panel:background_db.get("8a81f71bde65e4274ff4407cdf0c6320"),
    
    "aluminium_panel":aluminium_panel,
    
    
    
    #wafer_row:background_db.get("69e0ae62d7d13ca6270e9634a0c40374"),
    "wafer_row":wafer_row,
    
    
    
    #wafer_rer:background_db.get("4ced0dbf5e0dbf56245b175b8171a6fb"),
    "wafer_rer":wafer_rer,
    
    "wafer_multisi_rer":wafer_multisi_rer,
    
    "wafer_multisi_row":wafer_multisi_row,
    
    
    "elec_wafer_nz":elec_wafer_nz,
    "elec_wafer_rla":elec_wafer_rla,
   

   
   "elec_wafer_raf":elec_wafer_raf,
   "elec_wafer_au":elec_wafer_au,
   "elec_wafer_ci":elec_wafer_ci,
   "elec_wafer_rna":elec_wafer_rna,
   "elec_wafer_ras":elec_wafer_ras,
   
   
   
   
   # RER
   
   
   "elec_wafer_rer_single":elec_wafer_rer_single,
   
   
    
    
    "elec_wafer_rer_multi":elec_wafer_rer_multi,
    # Silicon prod
    
    #si_sg_row:background_db.get("2843a8c71e81a4134b27c8918f018372"),
    
    
    "si_sg_row":si_sg_row,
    
    
    # si_sg_rer=background_db.get("7f5d2c4c04f1e4d8733f9ba103f08822")
    
    "si_sg_rer":si_sg_rer,
    
    
    
    # elec_sili_raf:background_db.get("23198afc58716e2f16033b5e83b2e60a"),
    "elec_sili_raf":elec_sili_raf,
    
   
   
    
    
    
    "elec_sili_au":elec_sili_au,
    
    "elec_sili_ci":elec_sili_ci,
    
    "elec_sili_nz":elec_sili_nz,
    
    "elec_sili_ras":elec_sili_ras,
    
    "elec_sili_rna":elec_sili_rna,
    
    "elec_sili_rla":elec_sili_rla,
        

    "electricity_hydro_row_si_sg" : electricity_hydro_row_si_sg,
    
    
    
    "electricity_rer_si_sg" : electricity_rer_si_sg,
    
    
    
    
    "mount_system_AVS":mount_system_AVS,
    
    
    
    "mount_system_PV":mount_system_PV,
    
    # for exc in list(mount_system_PV.exchanges(),),:
    #     print(exc),
    
    
    #concrete_mount:background_db.get("bc8c7359f05890c22e37ab303bdd987a"),
    "concrete_mount":concrete_mount,
    
    
    #concrete_mount_waste_1:background_db.get("e399ae8908f7a23700aad8c833ece1ca"),
    
    "concrete_mount_waste_1":concrete_mount_waste_1,
    
    
    #concrete_mount_waste_2=background_db.get("5df13a7c86091d82d261b14090992eae"),
    
    "concrete_mount_waste_2":concrete_mount_waste_2,
    
    
    
    #concrete_mount_waste_3=background_db.get("d751bebe3aa8f622fc868d3b6a4ea6c4"),
    "concrete_mount_waste_3":concrete_mount_waste_3,
    
    
    # aluminium_mount =background_db.get("b2a028e18853749b07a3631ef6eccd96"), 
    
    
    
    "aluminium_mount":aluminium_mount,
    
    # alu_extru_mount =background_db.get("0d52d26769eb8011ebc49e09efb7259a"), 
    
    
    
    "alu_extru_mount":alu_extru_mount,
    
    
    # alu_mount_scrap_1 =background_db.get("238a2639dfe3ecbcd10a54426cd01094"), 
    "alu_mount_scrap_1":alu_mount_scrap_1,
    
    
    
    # alu_mount_scrap_2 =background_db.get("b487525640db147e00a2ece275262d59"), 
    
    
    "alu_mount_scrap_2":alu_mount_scrap_2,
    
    
    
    # alu_mount_scrap_3 =background_db.get("b7acf0e07b901a83ab9071d87e881af2"), 
    
    "alu_mount_scrap_3":alu_mount_scrap_3,
    
    
    # reinf_steel_mount =background_db.get("1737592f8b159167b376ff1ffe485e7e"), 
    
    
    "reinf_steel_mount":reinf_steel_mount,
    
    
    # chrom_steel_mount =background_db.get("ecc7db7759925cf080ec73926ca4470e"), 
    
    
    "chrom_steel_mount":chrom_steel_mount,
    
    
    # steel_rolling_mount =background_db.get("a19e6d008c28760e4872a5e7e001be0f"), 
    
    "steel_rolling_mount":steel_rolling_mount,
    
    
    
    # wire_mount:background_db.get("bc2fb8454c1dcf38750e1272ee1aae10"),
    
    
    "wire_mount":wire_mount,
    
    
    # steel_mount_scrap_1:background_db.get("d8deb03d160b32aff1962dd626878c1e"),
    
    "steel_mount_scrap_1":steel_mount_scrap_1,
    
    
    
    # steel_mount_scrap_2:background_db.get("b9e7ab6f52bf1a1c2079b7fd6da7139e"),
    
    "steel_mount_scrap_2":steel_mount_scrap_2,
    
    
    
    # steel_mount_scrap_3:background_db.get("f95f81d06894ce8cd3992b22592a7a2e"),
    "steel_mount_scrap_3":steel_mount_scrap_3,
    
    
    # Corrugated carb box
    
    # cor_box_mount_1:background_db.get("3c297273fd9dd888ca5f562b60633b1c"),
    "cor_box_mount_1":cor_box_mount_1,
    
    
    
    
    # cor_box_mount_2:background_db.get("88b4c06b619444f94dbadd73ba25a99a"),
    "cor_box_mount_2":cor_box_mount_2,
    
    # cor_box_mount_3:background_db.get("bc6d87ea5c33af362a873002e89d0a67"),
    
    "cor_box_mount_3":cor_box_mount_3,
    

    
    # cor_box_mount_waste_1:background_db.get("0fbef1744ad94621c41e6a29eff631e6"),
    "cor_box_mount_waste_1":cor_box_mount_waste_1,
    
    
    
    # cor_box_mount_waste_2:background_db.get("07135843544f495acdec3805593597cd"),
    
    "cor_box_mount_waste_2":cor_box_mount_waste_2,
    
    # cor_box_mount_waste_3:background_db.get("5adbcfff2d45da42b0c894714357be1d"),
    
    "cor_box_mount_waste_3":cor_box_mount_waste_3,
    
    
    # poly_mount_1:background_db.get("fe37424fcb39751b8436bea56516d595"),
    
    
    "poly_mount_1":poly_mount_1,
    
    
    # poly_mount_2:background_db.get("ede9bf12aa723d584c162ce6a6805974"),
    "poly_mount_2":poly_mount_2,
    
    
    
    # poly_mount_waste_1:background_db.get("47e42f31205177ca7ccd475552344b26"),
    
    # poly_mount_waste_1["name"]
    # poly_mount_waste_1["location"]
    
    "poly_mount_waste_1":poly_mount_waste_1,
    
    
    # poly_mount_waste_2:background_db.get("429bdbc2829990503a76dcae0b79e98d"),
    # poly_mount_waste_2["name"]
    # poly_mount_waste_2["location"]
    
    "poly_mount_waste_2":poly_mount_waste_2,
    
    
    
    # poly_mount_waste_3:background_db.get("96206cf1b984ab9666c6c84ad37a9f62"),
    # poly_mount_waste_3["name"]
    # poly_mount_waste_3["location"]
    
    "poly_mount_waste_3":poly_mount_waste_3,
    
    
    
    # poly_mount_waste_4:background_db.get("5db45c95744227b5dbf5368b217666d7"),
    
    # poly_mount_waste_4["name"]
    # poly_mount_waste_4["location"]
    
    "poly_mount_waste_4":poly_mount_waste_4,
    
    
    # poly_mount_waste_5:background_db.get("4ab43cbf5ef6529a43b65011a45b4352"),
    
    
    # poly_mount_waste_5["name"]
    # poly_mount_waste_5["location"]
    
    "poly_mount_waste_5":poly_mount_waste_5,
    
    
    # poly_mount_waste_6:background_db.get("ada15830ef22c5f76c41c633cafe81aa"),
    # poly_mount_waste_6["name"]
    # poly_mount_waste_6["location"]
    
    "poly_mount_waste_6":poly_mount_waste_6,
    
    
    
    # zinc_coat_mount_1:background_db.get("76c32858d64d75927d5a94a5a8683571"), 
    
    # zinc_coat_mount_1["name"]
    # zinc_coat_mount_1["location"]
    
    "zinc_coat_mount_1":zinc_coat_mount_1,
    
    
    # zinc_coat_mount_2:background_db.get("00d441215fd227f4563ff6c2f4ae3f02"), 
    
    # zinc_coat_mount_2["name"]
    # zinc_coat_mount_2["location"]
    
    "zinc_coat_mount_2":zinc_coat_mount_2,
    
    
    
    
    # Alfalfa input and ouput
    
    
    
    # Exchange: 6.6662e-05 hectare 'fertilising, by broadcaster' (hectare, CH, None), to 'alfalfa-grass mixture production, Swiss integrated production' (kilogram, CH, None),>
    # fertilising, by broadcaster
    # 302a1e9ca5f7b39a5c6f4a7e27bd56c0
    # fert_broadcaster_ch=background_db.get("302a1e9ca5f7b39a5c6f4a7e27bd56c0"),
    
    # fert_broadcaster_ch["name"]
    # fert_broadcaster_ch["location"]
    
    "fert_broadcaster_ch":fert_broadcaster_ch,
    
    
    
    
    # Exchange: 0.0042024 kilogram 'market for inorganic phosphorus fertiliser, as P2O5' (kilogram, CH, None), to 'alfalfa-grass mixture production, Swiss integrated production' (kilogram, CH, None),>
    # market for inorganic phosphorus fertiliser, as P2O5
    # adf4a377ba3fca72cb0d2090757a0bb1
    
    # ino_P205_ch=background_db.get("adf4a377ba3fca72cb0d2090757a0bb1"),
    
    
    # ino_P205_ch["name"]
    # ino_P205_ch["location"]
    
    "ino_P205_ch":ino_P205_ch,
    
    
    # Exchange: 0.0013332 cubic meter 'liquid manure spreading, by vacuum tanker' (cubic meter, CH, None), to 'alfalfa-grass mixture production, Swiss integrated production' (kilogram, CH, None),>
    # liquid manure spreading, by vacuum tanker
    # f8c1374cd1f5f1f800a873ee4feb590e
    
    # liquidmanure_spr_ch=background_db.get("f8c1374cd1f5f1f800a873ee4feb590e"),
    
    
    # liquidmanure_spr_ch["name"]
    # liquidmanure_spr_ch["location"]
    
    "liquidmanure_spr_ch":liquidmanure_spr_ch,
    
    
    
    
    
    # Exchange: 0.0084048 kilogram 'market for packaging, for fertilisers' (kilogram, GLO, None), to 'alfalfa-grass mixture production, Swiss integrated production' (kilogram, CH, None),>
    # market for packaging, for fertilisers
    # a9254f81cc8ecaeb85ab44c32b2e02be
    # packaging_fert_glo =background_db.get("a9254f81cc8ecaeb85ab44c32b2e02be"),
    
    
    # packaging_fert_glo["name"]
    # packaging_fert_glo["location"]
    
    "packaging_fert_glo":packaging_fert_glo,
    
    
    # Exchange: 0.66662 kilogram 'solid manure loading and spreading, by hydraulic loader and spreader' (kilogram, CH, None), to 'alfalfa-grass mixture production, Swiss integrated production' (kilogram, CH, None),>
    # solid manure loading and spreading, by hydraulic loader and spreader
    # d9355b12099cf24dd94f024cfca32d35
    # solidmanure_spreading_ch:background_db.get("d9355b12099cf24dd94f024cfca32d35"),
    
    # solidmanure_spreading_ch["name"]
    # solidmanure_spreading_ch["location"]
    
    "solidmanure_spreading_ch":solidmanure_spreading_ch,
    
    
    
    # Machinery
    # fodder_loading_ch: background_db.get("10f89cf01395e21928b4137637b523a8"),
    
    
    # fodder_loading_ch["name"]
    # fodder_loading_ch["location"]
    
    "fodder_loading_ch":fodder_loading_ch,
    
    
    
    # rotary_mower_ch: background_db.get("16735e3c568e791b8074941193b3ba12"),
    
    # rotary_mower_ch["name"]
    # rotary_mower_ch["location"]
    
    "rotary_mower_ch":rotary_mower_ch,
    
    
    
    
    # sowing_ch: background_db.get("324c1476b808b7149f0ed9d7c8f5afed"),
    
    
    # sowing_ch["name"]
    # sowing_ch["location"]
    
    "sowing_ch":sowing_ch,
    
    
    
    # tillage_rotary_spring_tine_ch: background_db.get("d408ce3256fb75715ddad41fcc92cebc"),
    
    # tillage_rotary_spring_tine_ch["name"]
    # tillage_rotary_spring_tine_ch["location"]
    
    "tillage_rotary_spring_tine_ch":tillage_rotary_spring_tine_ch,
    
    
    # tillage_ploughing_ch:background_db.get("457c5b6648cad5aecc781c9f5d5eca55"),
    
    
    # tillage_ploughing_ch["name"]
    # tillage_ploughing_ch["location"]
    
    "tillage_ploughing_ch":tillage_ploughing_ch,
    
    
    # tillage_rolling_ch:background_db.get("11374fa3faed9fe1cbab183ece4fa00e"),
    
    # tillage_rolling_ch["name"]
    # tillage_rolling_ch["location"]
    
    "tillage_rolling_ch":tillage_rolling_ch,
    
    
    
    
    
    
    # combine_harvesting_ch:background_db.get("967325bd96bf98cd0d180527060236f9"),
    
    # combine_harvesting_ch["name"]
    # combine_harvesting_ch["location"]
    
    "combine_harvesting_ch" :combine_harvesting_ch,
    
    
    
    #"""Soy_ch"""
    
    
    # for exc in list(soy_ch_AVS.exchanges(),),:
    #     if exc["type"]=="biosphere" and "Water" in exc["name"]:
    #         print(exc["name"]),
            
    
    
    # for exc in list(alfalfa_ch_AVS.exchanges(),),:
    #     if exc["type"]=="biosphere" and "Water" in exc["name"]:
    #         print(exc["name"]),
            
            
    # for exc in list(wheat_fr_AVS.exchanges(),),:
    #     if exc["type"]=="biosphere"and "Water" in exc["name"]:
    #         print(exc["name"]),        
    # Exchange: 0.00026044 hectare 'fertilising, by broadcaster' (hectare, CH, None), to 'soybean production' (kilogram, CH, None),>
    # fertilising, by broadcaster
    # 302a1e9ca5f7b39a5c6f4a7e27bd56c0
    # fert_broadcaster_ch=background_db.get("302a1e9ca5f7b39a5c6f4a7e27bd56c0"),
    
    # fert_broadcaster_ch["name"]
    # fert_broadcaster_ch["location"]
    
    "fert_broadcaster_ch":fert_broadcaster_ch,
    
    
    
    #Exchange: 0.00026044 hectare 'green manure growing, Swiss integrated production, until January' (hectare, CH, None), to 'soybean production' (kilogram, CH, None),
    #green manure growing, Swiss integrated production, until January
    # green_manure_ch =background_db.get("7dc3dcdf0989c1fe66d1cc94ac057726"),
    
    
    # green_manure_ch["name"]
    # green_manure_ch["location"]
    
    "green_manure_ch":green_manure_ch,
    
    
    
    # Exchange: 0.00066337 kilogram 'nutrient supply from thomas meal' (kilogram, GLO, None), to 'soybean production' (kilogram, CH, None),>
    # nutrient supply from thomas meal
    # 550fc5c30f02c82d6a7f491a6cfbf4c4
    
    
    # nutrient_supply_thomas_meal_ch =background_db.get("550fc5c30f02c82d6a7f491a6cfbf4c4"),
    
    # nutrient_supply_thomas_meal_ch["name"]
    # nutrient_supply_thomas_meal_ch["location"]
    
    "nutrient_supply_thomas_meal_ch":nutrient_supply_thomas_meal_ch,
    
    
    
    # Exchange: 0.0011157 cubic meter 'liquid manure spreading, by vacuum tanker' (cubic meter, CH, None), to 'soybean production' (kilogram, CH, None),>
    # liquid manure spreading, by vacuum tanker
    # f8c1374cd1f5f1f800a873ee4feb590e
    # liquidmanure_spr_ch =background_db.get("f8c1374cd1f5f1f800a873ee4feb590e"),
    
    # liquidmanure_spr_ch["name"]
    # liquidmanure_spr_ch["location"]
    
    "liquidmanure_spr_ch":liquidmanure_spr_ch,
    
    
    
    
    # Exchange: 0.03943066 kilogram 'market for packaging, for fertilisers' (kilogram, GLO, None), to 'soybean production' (kilogram, CH, None),>
    # market for packaging, for fertilisers
    # a9254f81cc8ecaeb85ab44c32b2e02be
    # packaging_fert_glo =background_db.get("a9254f81cc8ecaeb85ab44c32b2e02be"),
    
    
    # packaging_fert_glo["name"]
    # packaging_fert_glo["location"]
    
    "packaging_fert_glo":packaging_fert_glo,
    
    
    
    # Exchange: 0.009595625 kilogram 'market for phosphate rock, beneficiated' (kilogram, RER, None), to 'soybean production' (kilogram, CH, None),>
    # market for phosphate rock, beneficiated
    # 6ab3fd3b19187e0406419b03f3b88ff3
    # phosphate_rock_glo:background_db.get("6ab3fd3b19187e0406419b03f3b88ff3"),
    
    
    # phosphate_rock_glo["name"]
    # phosphate_rock_glo["location"]
    
    "phosphate_rock_glo":phosphate_rock_glo,
    
    
    
    # Exchange: 0.0156317446180305 kilogram 'market for potassium chloride' (kilogram, RER, None), to 'soybean production' (kilogram, CH, None),>
    # market for potassium chloride
    # 3c83fdd5c69822caadf209bdbf8a50f3
    # potassium_chloride_rer:background_db.get("3c83fdd5c69822caadf209bdbf8a50f3"),
    
    
    # potassium_chloride_rer["name"]
    # potassium_chloride_rer["location"]
    
    "potassium_chloride_rer":potassium_chloride_rer,
    
    
    
    # Exchange: 0.00119833261913457 kilogram 'market for potassium sulfate' (kilogram, RER, None), to 'soybean production' (kilogram, CH, None),>
    # market for potassium sulfate
    # 06f1d7f08f051e8e466af8fd4faaadd9
    # potassium_sulfate_rer:background_db.get("06f1d7f08f051e8e466af8fd4faaadd9"),
    
    # potassium_sulfate_rer["name"]
    # potassium_sulfate_rer["location"]
    
    "potassium_sulfate_rer":potassium_sulfate_rer,
    
    
    
    
    # Exchange: 0.00100714285714286 kilogram 'market for single superphosphate' (kilogram, RER, None), to 'soybean production' (kilogram, CH, None),>
    # market for single superphosphate
    # 61f2763127305038f68860164831764d
    # single_superphosphate_rer:background_db.get("61f2763127305038f68860164831764d"),
    
    
    
    # single_superphosphate_rer["name"]
    # single_superphosphate_rer["location"]
    
    "single_superphosphate_rer":single_superphosphate_rer,
    
    
    
    
    # Exchange: 0.072403 kilogram 'solid manure loading and spreading, by hydraulic loader and spreader' (kilogram, CH, None), to 'soybean production' (kilogram, CH, None),>
    # solid manure loading and spreading, by hydraulic loader and spreader
    # d9355b12099cf24dd94f024cfca32d35
    # solidmanure_spreading_ch:background_db.get("d9355b12099cf24dd94f024cfca32d35"),
    
    
    
    # solidmanure_spreading_ch["name"]
    # solidmanure_spreading_ch["location"]
    
    "solidmanure_spreading_ch":solidmanure_spreading_ch,
    
    
    
    
    # Exchange: 0.0114058695652174 kilogram 'market for triple superphosphate' (kilogram, RER, None), to 'soybean production' (kilogram, CH, None),>
    # market for triple superphosphate
    # 10952b351896a1f0935e0dbd1fc98f49
    # triplesuperphosphate:background_db.get("10952b351896a1f0935e0dbd1fc98f49"),
    
    # triplesuperphosphate["name"]
    # triplesuperphosphate["location"]
    
    "triplesuperphosphate":triplesuperphosphate,
    
    
    
    # Machinery
    # tillage_currying_weeder_ch: background_db.get("9fdce9892f57840bd2fa96d92ab577e3"),
    
    # tillage_currying_weeder_ch["name"]
    # tillage_currying_weeder_ch["location"]
    
    "tillage_currying_weeder_ch":tillage_currying_weeder_ch,
    
    
    
    # tillage_rotary_spring_tine_ch: background_db.get("d408ce3256fb75715ddad41fcc92cebc"),
    
    # tillage_rotary_spring_tine_ch["name"]
    # tillage_rotary_spring_tine_ch["location"]
    
    "tillage_rotary_spring_tine_ch":tillage_rotary_spring_tine_ch,
    
    
    
    
    # sowing_ch: background_db.get("324c1476b808b7149f0ed9d7c8f5afed"),
    
    # sowing_ch["name"]
    # sowing_ch["location"]
    
    "sowing_ch":sowing_ch,
    
    
    
    # combine_harvesting_ch:background_db.get("967325bd96bf98cd0d180527060236f9"),
    
    # combine_harvesting_ch["name"]
    # combine_harvesting_ch["location"]
    
    "combine_harvesting_ch":combine_harvesting_ch,
    
    
    
    #"""Wheat fr"""
    
    
    # ammonium_nitrate:background_db.get("28acde92cd1aec3b5d5b8f58f50055ba"),
    
    # ammonium_nitrate["name"]
    # ammonium_nitrate["location"]
    
    "ammonium_nitrate":ammonium_nitrate,
    
    
    
    
    # ammonium_sulfate:background_db.get("3e5d910e09135106f6c712c538182b71"),
    
    
    # ammonium_sulfate["name"]
    # ammonium_sulfate["location"]
    
    "ammonium_sulfate":ammonium_sulfate,
    
    
    
    # urea:background_db.get("d729224f52da90228e5c4b7f864303b0"),
    
    # urea["name"]
    # urea["location"]
    
    "urea":urea,
    
    
    
    # Exchange: 0.00051831 hectare 'market for fertilising, by broadcaster' (hectare, GLO, None), to 'wheat grain production' (kilogram, FR, None),>
    # market for fertilising, by broadcaster
    # 2433809f3b7d23df8625099f845e381
    # fert_broadcaster=background_db.get("2433809f3b7d23df8625099f845e3814"),
    
    # fert_broadcaster["name"]
    # fert_broadcaster["location"]
    
    "fert_broadcaster":fert_broadcaster,
    
    
    
    
    # Exchange: 0.0104 kilogram 'market for inorganic phosphorus fertiliser, as P2O5' (kilogram, FR, None), to 'wheat grain production' (kilogram, FR, None),>
    
    
    # ino_P205_fr=background_db.get("12b793a4ed3da8e8e17209b4e788d9c0"),
    
    # ino_P205_fr["name"]
    # ino_P205_fr["location"]
    
    "ino_P205_fr":ino_P205_fr,
    
    
    
    # Exchange: 3.406e-07 kilogram 'market for organophosphorus-compound, unspecified' (kilogram, GLO, None), to 'wheat grain production' (kilogram, FR, None),>
    # market for organophosphorus-compound, unspecified
    # cbabef8729a2ba6a1286742273fa3a6e
    # org_P205=background_db.get("cbabef8729a2ba6a1286742273fa3a6e"),
    
    # org_P205["name"]
    # org_P205["location"]
    
    "org_P205":org_P205,
    
    
    
    
    # Exchange: 0.094817 kilogram 'market for packaging, for fertilisers' (kilogram, GLO, None), to 'wheat grain production' (kilogram, FR, None),>
    # market for packaging, for fertilisers
    # a9254f81cc8ecaeb85ab44c32b2e02be
    
    # packaging_fert:background_db.get("a9254f81cc8ecaeb85ab44c32b2e02be"),
    
    
    # packaging_fert["name"]
    # packaging_fert["location"]
    
    "packaging_fert":packaging_fert,
    
    
    
    
    
    # Exchange: 0.123183267669125 cubic meter 'market for irrigation' (cubic meter, FR, None), to 'wheat grain production' (kilogram, FR, None),>
    # market for irrigation
    # 432f8fdd083f6730c0f2ae6c8014656e
    # water_market_irrigation:background_db.get("432f8fdd083f6730c0f2ae6c8014656e"),
    
    # water_market_irrigation["name"]
    # water_market_irrigation["location"]
    
    "water_market_irrigation":water_market_irrigation,
    
    
    
    
    
    # Machinery
    # tillage_rotary_harrow_glo: background_db.get("cb6989c6601a864504c47d84b469704a"),
    
    # tillage_rotary_harrow_glo["name"]
    # tillage_rotary_harrow_glo["location"]
    
    "tillage_rotary_harrow_glo":tillage_rotary_harrow_glo,
    
    
    # tillage_rotary_spring_tine_glo: background_db.get("d34f658ca1260bf83866b037f4c73306"),
    
    
    # tillage_rotary_spring_tine_glo["name"]
    # tillage_rotary_spring_tine_glo["location"]
    
    "tillage_rotary_spring_tine_glo":tillage_rotary_spring_tine_glo,
    
    
    # sowing_glo: background_db.get("bd78d677f0e04fe1038f135719578207"),
    
    # sowing_glo["name"]
    # sowing_glo["location"]
    
    "sowing_glo":sowing_glo,
    
    
    
    # tillage_ploughing_glo:background_db.get("bcd18cd662caba362f4dcbc8e05002d7"),
    
    
    # tillage_ploughing_glo["name"]
    # tillage_ploughing_glo["location"]
    
    "tillage_ploughing_glo":tillage_ploughing_glo,
    
    
    # combine_harvesting_glo:background_db.get("20966082098d6802dc2d3e1f0346aece"),
    
    
    # combine_harvesting_glo["name"]
    # combine_harvesting_glo["location"]
    
    "combine_harvesting_glo":combine_harvesting_glo,
    
    
    
    # Maize
    
    
    
    "market_for_irrigation_maize" : market_for_irrigation_maize,
    
    "water_maizewaterground" : water_maizewaterground,
    
    "water_maizewatersurfacewater" : water_maizewatersurfacewater,
    
    "hoeing_maize" : hoeing_maize,
    
    "tillage_harrowing_by_spring_tine_harrow_maize" : tillage_harrowing_by_spring_tine_harrow_maize,
    
    "sowing_maize" : sowing_maize,
    
    "market_for_transport_tractor_and_trailer_agricultural_maize" : market_for_transport_tractor_and_trailer_agricultural_maize,
   
    "tillage_ploughing_maize" : tillage_ploughing_maize,
    
    "nitrate_maizewaterground" : nitrate_maizewaterground,
    "nitrogen_oxides_maizeairnonurbanairorfromhighstacks" : nitrogen_oxides_maizeairnonurbanairorfromhighstacks,
    "phosphate_maizewatersurfacewater" : phosphate_maizewatersurfacewater,
    "dinitrogen_monoxide_maizeairnonurbanairorfromhighstacks" : dinitrogen_monoxide_maizeairnonurbanairorfromhighstacks,
    "ammonia_maizeairnonurbanairorfromhighstacks" : ammonia_maizeairnonurbanairorfromhighstacks,
    "phosphate_maizewaterground" : phosphate_maizewaterground,
    
    "market_for_ammonium_sulfate_maize" : market_for_ammonium_sulfate_maize,
    "fertilising_by_broadcaster_maize" : fertilising_by_broadcaster_maize,
    "market_for_inorganic_nitrogen_fertiliser_as_n_maize" : market_for_inorganic_nitrogen_fertiliser_as_n_maize,
    "market_for_urea_maize" : market_for_urea_maize,
    "market_for_packaging_for_fertilisers_maize" : market_for_packaging_for_fertilisers_maize,
    "market_for_phosphate_rock_beneficiated_maize" : market_for_phosphate_rock_beneficiated_maize,
    "market_for_inorganic_phosphorus_fertiliser_as_p2o5_maize" : market_for_inorganic_phosphorus_fertiliser_as_p2o5_maize,
    "green_manure_growing_swiss_integrated_production_until_april_maize" : green_manure_growing_swiss_integrated_production_until_april_maize,
    "liquid_manure_spreading_by_vacuum_tanker_maize" : liquid_manure_spreading_by_vacuum_tanker_maize,
    "solid_manure_loading_and_spreading_by_hydraulic_loader_and_spreader_maize" : solid_manure_loading_and_spreading_by_hydraulic_loader_and_spreader_maize,
    "market_for_ammonium_nitrate_maize" : market_for_ammonium_nitrate_maize,
    "carbon_dioxide_fossil_maizeairnonurbanairorfromhighstacks" : carbon_dioxide_fossil_maizeairnonurbanairorfromhighstacks,
    
    
    

    
    

    #"""Biosphere_db outputs agri"""
    
    # Flow to add for Carbon storage differential
    
    #  ('Carbon dioxide, to soil or biomass stock' (kilogram, None, ('soil',),),,
    #  '375bc95e-6596-4aa1-9716-80ff51b9da77'),
    
    #[(flow,flow["code"]), for flow in Biosphere_db  if "dioxide" in flow["name"]]
    
    # Carbon_dioxide_to_soil_biomass_stock:Biosphere_db.get('375bc95e-6596-4aa1-9716-80ff51b9da77'),
    
    # Carbon_dioxide_to_soil_biomass_stock["name"]
    # Carbon_dioxide_to_soil_biomass_stock["categories"]
    
    
    "Carbon_dioxide_to_soil_biomass_stock":Carbon_dioxide_to_soil_biomass_stock,
    
    
    
    
    
    
    # ammonia:Biosphere_db.get("0f440cc0-0f74-446d-99d6-8ff0e97a2444"),
    
    # ammonia["name"]
    # ammonia["categories"]
    
    
    "ammonia":ammonia,
    
    
    
    
    # dinitrogen_monoxide:Biosphere_db.get("afd6d670-bbb0-4625-9730-04088a5b035e"),
    
    # dinitrogen_monoxide["name"]
    # dinitrogen_monoxide["categories"]
    
    
    "dinitrogen_monoxide":dinitrogen_monoxide,
    
    
    # nitrogen_oxide:Biosphere_db.get("77357947-ccc5-438e-9996-95e65e1e1bce"),
    
    # nitrogen_oxide["name"]
    # nitrogen_oxide["categories"]
    
    
    "nitrogen_oxide":nitrogen_oxide,
    
    
    
    
    # nitrate:Biosphere_db.get("b9291c72-4b1d-4275-8068-4c707dc3ce33"),
    
    # nitrate["name"]
    # nitrate["categories"]
    
    
    "nitrate":nitrate,
    
    
    
    
    # Exchange: 3.1767e-05 kilogram 'Phosphate' (kilogram, None, ('water', 'ground-'),), to 'wheat grain production' (kilogram, FR, None),>
    # Phosphate
    # phosphate_groundwater=Biosphere_db.get("329fc7d8-4011-4327-84e4-34ff76f0e42d"),
    
    # phosphate_groundwater["name"]
    # phosphate_groundwater["categories"]
    
    
    "phosphate_groundwater":phosphate_groundwater,
    
    
    
    # phosphate_surfacewater=Biosphere_db.get("1727b41d-377e-43cd-bc01-9eaba946eccb"),
    
    # phosphate_surfacewater["name"]
    # phosphate_surfacewater["categories"]
    
    
    "phosphate_surfacewater":phosphate_surfacewater,
    
    
    # phosphorus_surfacewater=Biosphere_db.get("b2631209-8374-431e-b7d5-56c96c6b6d79"),
    
    # phosphorus_surfacewater["name"]
    # phosphorus_surfacewater["categories"]
    
    
    "phosphorus_surfacewater":phosphorus_surfacewater,
    
    
    
    # Exchange: 0.0978691061631198 cubic meter 'Water' (cubic meter, None, ('air',),), to 'wheat grain production' (kilogram, FR, None),>
    # Water
    # 075e433b-4be4-448e-9510-9a5029c1ce94
    # water_air:Biosphere_db.get("075e433b-4be4-448e-9510-9a5029c1ce94"),
    
    # water_air["name"]
    # water_air["categories"]
    
    
    "water_air":water_air,
    
    
    
    # water_ground:Biosphere_db.get("51254820-3456-4373-b7b4-056cf7b16e01"),
    
    
    # water_ground["name"]
    # water_ground["categories"]
    
    
    "water_ground":water_ground,
    
    
    
    # water_surface:Biosphere_db.get("db4566b1-bd88-427d-92da-2d25879063b9"),
    
    # water_surface["name"]
    # water_surface["categories"]
    
    
    "water_surface":water_surface,
    
    
    # Exchange: 1.42397953472488 kilogram 'Carbon dioxide, in air' (kilogram, None, ('natural resource', 'in air'),), to 'wheat grain production' (kilogram, FR, None),>
    # Carbon dioxide, in air
    # cc6a1abb-b123-4ca6-8f16-38209df609be
    # carbon_dioxide_air_resource:Biosphere_db.get("cc6a1abb-b123-4ca6-8f16-38209df609be"),
    
    # carbon_dioxide_air_resource["name"]
    # carbon_dioxide_air_resource["categories"]
    
    
    "carbon_dioxide_air_resource":carbon_dioxide_air_resource,
    
    
    # Exchange: 15.1230001449585 megajoule 'Energy, gross calorific value, in biomass' (megajoule, None, ('natural resource', 'biotic'),), to 'wheat grain production' (kilogram, FR, None),>
    # Energy, gross calorific value, in biomass
    # 01c12fca-ad8b-4902-8b48-2d5afe3d3a0f
    #energy_inbiomass:Biosphere_db.get("01c12fca-ad8b-4902-8b48-2d5afe3d3a0f"),
    
    
    # energy_inbiomass["name"]
    # energy_inbiomass["categories"]
    
    
    "energy_inbiomass":energy_inbiomass,
    
    
    
    # carbon dioxide fossi from urea (only wheat),
    #carbondioxide_fossil_urea:Biosphere_db.get("aa7cac3a-3625-41d4-bc54-33e2cf11ec46"),
    
    # carbondioxide_fossil_urea["name"]
    # carbondioxide_fossil_urea["categories"]
    
    
    "carbondioxide_fossil_urea":carbondioxide_fossil_urea }
        
        






    
    # Parameters distributions dictionnaries
    
    #
    # each parameter is assigned a list containing  :
       # [Distribution,unique value if unique, min,max,mode,sd, unit]
    
    # Distribution :
    #   - 'unique' if no distribution. The  algorithm considers the value "unique"
    #   - 'unif' for uniform, uses min and max
    #   - 'triang, uses mim max and mode with mode as a fracion of max-min
    
    #
    
    
    # # iLUC param
    # iluc_par_distributions = {
    
    #     "NPP_weighted_ha_y_eq_cropref":['unique', [1,0.5, 2, 0, 0],"."],
    #     "NPP_weighted_ha_y_eq_ratio_avs_cropref":['unique', [1,0.5, 2, 0, 0],"."],
    #     "NPP_weighted_ha_y_eq_ratio_pv_cropref":['unique', [1,0.5, 2, 0, 0],"."],
    #     # "iluc_lm_PV":['unif', [1800,300, 2200, 0, 0],"kg CO2 eq.ha eq-1"],
    #     # "iluc_lm_AVS":['unif', [300,1400, 2200, 0, 0],"kg CO2 eq.ha eq-1"],
    #     # "iluc_lm_cropref":['unif', [1800,1400, 2200, 0, 0],"kg CO2 eq.ha eq-1"]
    
    #     "iluc_lm_PV":['unique', [0,300, 2200, 0],"kg CO2 eq.ha eq-1"],
    #     "iluc_lm_AVS":['unique', [0,300, 2200, 0],"kg CO2 eq.ha eq-1"],
    #     "iluc_lm_cropref":['unique', [0,300, 2200, 0],"kg CO2 eq.ha eq-1"]
        
    #     }
    
    # # We collect the intitial output values for the crop activities and use them as parameters
    
    
    # output_wheat = [exc["amount"] for exc in list(wheat_fr_ref.exchanges()) if exc["type"]=="production"][0]
    # output_soy = [exc["amount"] for exc in list(soy_ch_ref.exchanges()) if exc["type"]=="production"][0]
    # output_alfalfa = [exc["amount"] for exc in list(alfalfa_ch_ref.exchanges()) if exc["type"]=="production"][0]
    # output_maize= [exc["amount"] for exc in list(maize_ch_ref.exchanges()) if exc["type"]=="production"][0]
    
    # # Agronomical param
    # Ag_par_distributions = {
    
    #     "prob_switch_crop":['switch',4,[0,1,0,0],["maize","wheat","soy","alfalfa"],"."],
    #     "crop_yield_upd_ref":['unique', [1,2, 1, 1],"."],
    #     "crop_yield_ratio_avs_ref":['unique', [1,1, 0.5, 1],"."],
    #     "crop_fert_upd_ref":['unique', [1,0.5, 2, 0, 0],"."],
    #     "crop_fert_ratio_avs_ref":['unif', [1,0.3, 2, 0, 0],"."],
    #     "crop_mach_upd_ref":['unif', [1,0.5, 2, 0, 0],"."],
    #     "crop_mach_ratio_avs_ref":['unique', [1,0.3, 2, 0, 0],"."],
    #     "water_upd_ref":['unique', [1,0.5, 2, 0, 0],"."],
    #     "water_ratio_avs_ref":['unique', [1,0.3, 2, 0, 0],"."],
    #     "carbon_accumulation_soil_ref":['unique', [0,0.8, 1.2, 0, 0],"kg C.ha-1.y-1"],
    #     "carbon_accumulation_soil_AVS":['unique', [0,0.8, 1.2, 0, 0],"kg C.ha-1.y-1"],
    #     "carbon_accumulation_soil_PV":['unique', [0,0.8, 1.2, 0, 0],"kg C.ha-1.y-1"],
        
    #     "init_yield_alfalfa":['unique', [output_alfalfa,0, 0, 0, 0],"."],
    #     "init_yield_soy":['unique', [output_soy,0, 0, 0, 0],"."],
    #     "init_yield_wheat":['unique', [output_wheat,0, 0, 0, 0],"."],
    #     "init_yield_maize":['unique', [output_maize,0, 0, 0, 0],"."],
        
    #     # conversion npp to yield
    #     "root_shoot_ratio": ['norm', [0.21,0, 0, 0.178, 0.04],"."],
    #     "straw_grain_ratio":['norm', [1.35,0, 0, 1.6398, 0.543],"."],
    #     "carbon_content_grain":['norm', [0.43,0, 0, 0.429, 0.015],"."],
    #     "carbon_content_straw":['unique', [0.446,0, 0, 0, 0],"."],
    #     "water_content_grain":['unique', [0.14,0, 0, 0, 0],"."],
    #     "water_content_straw":['unique', [0.14,0, 0, 0, 0],"."]}


        
        
    
    # # PV param
    # PV_par_distributions = {
    #     'annual_producible_PV': ['unique', [1200, 900, 2000, 0, 0],"kwh.kwp-1.y-1"],
    #     'annual_producible_ratio_avs_pv': ['unique', [1,0.6, 1.3, 0, 0],"."],
        
    #     'mass_to_power_ratio_electric_instal_PV': ['unique', [2.2,1.5, 8, 0, 0],"kg electric installation . kwp -1"],  # Besseau minimum 2.2
        
    #     'panel_efficiency_PV': ['unique', [0.228,0.10, 0.40, 0, 0],"."],   # Besseau maximum 22.8%
        
        
    #     'inverter_specific_weight_PV': ['unique', [0.85,0.4, 7, 0, 0],"kg inverter.kwp-1"],   # Besseau maximum 0.85
       
    #     'inverter_lifetime_PV': ['unique', [15,10, 30, 0, 0],"y"], # Besseau 15
    
    #     'plant_lifetime_PV': ['unique', [30,20, 40, 0, 0],"y"], # Besseau and ecoinvent 30
        
    #     'plant_lifetime_ratio_avs_pv': ['unique', [1,0.3, 2, 0, 0],"."],  
        
    #     'concrete_mount_upd_PV': ['unique', [1,0.1, 2, 0, 0],"."],
        
    #     'concrete_mount_ratio_avs_pv': ['unique', [1,0.1, 4, 0, 0],"."],   # here facotr 4 because could be way worse
        
    #     'aluminium_mount_upd_PV': ['unique', [1,0.1, 2, 0, 0],"."], # here besseau suggests 1.5 kilo and the original value is 3.98. So update = 1.5/3.98 =  0.38
            
    #     'aluminium_mount_ratio_avs_pv': ['unique', [1,0.1, 4, 0, 0],"."],
        
    #     'steel_mount_upd_PV': ['unique', [1,0.1, 2, 0, 0],"."],
            
    #     'steel_mount_ratio_avs_pv': ['unique', [1,0.1, 4, 0, 0],"."],
    
    #     'poly_mount_upd_PV': ['unique', [1,0.1, 2, 0, 0],"."],
            
    #     'poly_mount_ratio_avs_pv': ['unique', [1,0.1,4 , 0, 0],"."],
    
    
    #     'corrbox_mount_upd_PV': ['unique', [1,0.1, 2, 0, 0],"."],
            
    #     'corrbox_mount_ratio_avs_pv': ['unique', [1,0.1, 4, 0, 0],"."],
        
        
    #     'zinc_mount_upd_PV': ['unique', [1,0.1, 2, 0, 0],"."],
            
    #     'zinc_mount_ratio_avs_pv': ['unique', [1,0.1, 4, 0, 0],"."],
    
    
    #     'surface_cover_fraction_PV': ['unique', [0.4,0.20, 0.6, 0, 0],"m2panel.m-2"],
        
    #     'surface_cover_fraction_AVS': ['unique', [0.4,0.10, 0.6, 0, 0],"."],
        
        
        
        
    #     'aluminium_panel_weight_PV': ['unique', [1.5,0, 3, 0, 0],"kg aluminium.m-2 panel"],  #Besseau :The ecoinvent inventory model assumes 2.6 kg aluminum/m2 whereas more recent studies indicate 1.5 kg/m2.42 Some PV panels are even frameless
        
        
    
    
    #     'manufacturing_eff_wafer_upd_PV': ['unique', [3,0.1, 2, 0, 0],"."],
    
    #     'solargrade_electric_intensity_PV': ['unique', [110,20, 110, 0, 0],"."],
        
    
    #     "prob_switch_substielec":['switch',5,[0,0,0,1,0],["substit_margi","substit_PV","substit_margi_fr","substit_margi_it","substit_margi_es"],"."],
        
    #     "impact_update_margi":['unique',[1,0.01, 3, 0, 0],"."]
    
    #     }
    
    
    
    
    
    # iLUC param
    iluc_par_distributions = {
    
        "NPP_weighted_ha_y_eq_cropref":['unif', [1,0.72, 1.16, 0, 0],"."],
        "NPP_weighted_ha_y_eq_ratio_avs_cropref":['unique', [1,0.5, 2, 0, 0],"."],  # Same quality of land
        "NPP_weighted_ha_y_eq_ratio_pv_cropref":['unique', [1,0.5, 2, 0, 0],"."], # Same quality of land
        # "iluc_lm_PV":['unif', [1800,300, 2200, 0, 0],"kg CO2 eq.ha eq-1"],
        # "iluc_lm_AVS":['unif', [300,1400, 2200, 0, 0],"kg CO2 eq.ha eq-1"],
        # "iluc_lm_cropref":['unif', [1800,1400, 2200, 0, 0],"kg CO2 eq.ha eq-1"]
    
        "iluc_lm_PV":['unique', [1840,300, 2200, 0],"kg CO2 eq.ha eq-1"], # Same market of land

        "iluc_lm_AVS":['unique', [1840,300, 2200, 0],"kg CO2 eq.ha eq-1"],  # Same market of land
        "iluc_lm_cropref":['unique', [1840,300, 2200, 0],"kg CO2 eq.ha eq-1"] # Same market of land

        
        }
    
    # We collect the intitial output values for the crop activities and use them as parameters
    
    
    output_wheat = [exc["amount"] for exc in list(wheat_fr_ref.exchanges()) if exc["type"]=="production"][0]
    output_soy = [exc["amount"] for exc in list(soy_ch_ref.exchanges()) if exc["type"]=="production"][0]
    output_alfalfa = [exc["amount"] for exc in list(alfalfa_ch_ref.exchanges()) if exc["type"]=="production"][0]
    output_maize= [exc["amount"] for exc in list(maize_ch_ref.exchanges()) if exc["type"]=="production"][0]
    
    # Agronomical param
    Ag_par_distributions = {
    
        "prob_switch_crop":['switch',4,[0,1,0,0],["maize","wheat","soy","alfalfa"],"."],
        "crop_yield_upd_ref":['unif', [1,0.03, 1.5, 1],"."],  # Based on min and max yields in France and SPain
        "crop_yield_ratio_avs_ref":['unif', [1,0.75, 1.25, 1],"."],  
        "crop_fert_upd_ref":['unif', [1,0.03, 1.5, 0, 0],"."], 
        "crop_fert_ratio_avs_ref":['unif', [1,0.75, 1.25, 0, 0],"."], 
        "crop_mach_upd_ref":['unif', [1,0.03, 1.5, 0, 0],"."], # 
        "crop_mach_ratio_avs_ref":['unif', [1,0.75, 1.25, 0, 0],"."],
        "water_upd_ref":['unique', [0,0.8, 1.2, 0, 0],"."], # By putting 0 : removes irrigtion
        "water_ratio_avs_ref":['unique', [1,0.8, 2, 0, 0],"."],
        "carbon_accumulation_soil_ref":['unique', [0,0.8, 1.2, 0, 0],"kg C.ha-1.y-1"],
        "carbon_accumulation_soil_AVS":['unique', [0,0.8, 1.2, 0, 0],"kg C.ha-1.y-1"],
        "carbon_accumulation_soil_PV":['unique', [0,0.8, 1.2, 0, 0],"kg C.ha-1.y-1"],
        
        "init_yield_alfalfa":['unique', [output_alfalfa,0, 0, 0, 0],"."],
        "init_yield_soy":['unique', [output_soy,0, 0, 0, 0],"."],
        "init_yield_wheat":['unique', [output_wheat,0, 0, 0, 0],"."],
        "init_yield_maize":['unique', [output_maize,0, 0, 0, 0],"."],
        
        # conversion npp to yield
       #  "root_shoot_ratio": ['empirical',[0.21,0.14,0.13,0.17,0.25,0.17],"."],
       #  "straw_grain_ratio":['empirical', [1.564102564, 1.325581395, 1.941176471, 1.631578947, 1.564102564, 1.380952381, 0.639344262, 1.380952381, 1.702702703, 0.666666667,
       # 2.125,1.702702703,2.03030303,1.43902439,1.631578947,3.166666667,2.125,1.5],"."],
       #  "carbon_content_grain":['empirical', [0.413,0.418,0.434,0.426,0.455],"."],
       #  "carbon_content_straw":['unique', [0.446,0, 0, 0, 0],"."],
       #  "water_content_grain":['unique', [0.14,0, 0, 0, 0],"."],
       #  "water_content_straw":['unique', [0.14,0, 0, 0, 0],"."],
        "correc_orchidee":['unique', [1,0,0,0,0],"."] # ratios for france # For the sensitivity analysis, there is no need to use this parameter as we will directly overwrite the activity yield with the the yield * crop_ref_update
        }

# ['empirical', [0.12861133180556408,
#  0.14281805060936634,
#  0.1477037637920191,
#  0.132855884674253,
#  0.16628774291251025,
#  0.1214386731559089,
#  0.18153349903861868,
#  0.14371036702728812,
#  0.130195188408881,
#  0.12406485708265667,
#  0.14007099837301587],"."]

        
        
    
    # PV param
    PV_par_distributions = {
        'annual_producible_PV': ['unif', [1200, 900, 2000, 0, 0],"kwh.kwp-1.y-1"],
        'annual_producible_ratio_avs_pv': ['unique', [1,0.6, 1.3, 0, 0],"."],
        
        'mass_to_power_ratio_electric_instal_PV': ['unif', [2.2,2.2-0.25*2.2, 2.2+0.25*2.2, 0, 0],"kg electric installation . kwp -1"],  # Besseau minimum 2.2
        
        'panel_efficiency_PV': ['unif', [0.228,0.179, 0.270, 0, 0],"."],   # Besseau maximum 22.8%
        
        
        'inverter_specific_weight_PV': ['unif', [0.85,0.85-0.25*0.85, 0.85+0.25*0.85, 0, 0],"kg inverter.kwp-1"],   # Besseau maximum 0.85
       
        'inverter_lifetime_PV': ['unif', [15,15-0.25*15, 15+0.25*15, 0, 0],"y"], # Besseau 15
    
        'plant_lifetime_PV': ['unif', [30,30-0.25*30, 30-0.25*30, 0, 0],"y"], # Besseau and ecoinvent 30
        
        'plant_lifetime_ratio_avs_pv': ['unif', [1,1-0.25*1, 1+0.25*1, 0, 0],"."],  
        
        'concrete_mount_upd_PV': ['unif', [1,1-0.25*1, 1+0.25*1,0.5,0],"."], # TO BE DEFINED WITH ZIMMERMAN
        
        'concrete_mount_ratio_avs_pv': ['unif', [0,0, 2, 1,0],"."],   # here facotr 4 because could be way worse
        
        'aluminium_mount_upd_PV': ['unif', [0.38,0.38-0.25*0.38, 0.38+0.25*0.38, 0, 0],"."], # here besseau suggests 1.5 kilo and the original value is 3.98. So update = 1.5/3.98 =  0.38
            
        'aluminium_mount_ratio_avs_pv': ['unif', [0,0,2,1,0],"."],
        
        'steel_mount_upd_PV': ['unif', [1,1-0.25*1, 1+0.25*1,0.5,0],"."],
            
        'steel_mount_ratio_avs_pv': ['unif', [0,0,2,1,0],"."],
    
        'poly_mount_upd_PV': ['unif', [1,1-0.25*1,1+0.25*1,0.5,0],"."],
            
        'poly_mount_ratio_avs_pv': ['unif',[0,0,2,1,0],"."],
    
    
        'corrbox_mount_upd_PV': ['unif', [1,1-0.25*1,1+0.25*1,0.5],"."],
            
        'corrbox_mount_ratio_avs_pv': ['unif', [0,0,2,1,0],"."],
        
        
        'zinc_mount_upd_PV': ['unif', [1,1-0.25*1,1+0.25*1,0.5],"."],
            
        'zinc_mount_ratio_avs_pv': ['unif', [0,0,2,1,0],"."],
    
    
        'surface_cover_fraction_PV': ['unif', [0.4383,0.4383-0.25*0.4383,0.4383+0.25*0.4383, 0, 0],"m2panel.m-2"],
        
        'surface_cover_fraction_AVS': ['unif', [0.7617,0.7617-0.25*0.7617,0.7617+0.25*0.7617, 0, 0],"."],
        
        
        
        
        'aluminium_panel_weight_PV': ['unif', [1.5,1.5-0.25*1.5,1.5+0.25*1.5, 0, 0],"kg aluminium.m-2 panel"],  #Besseau :The ecoinvent inventory model assumes 2.6 kg aluminum/m2 whereas more recent studies indicate 1.5 kg/m2.42 Some PV panels are even frameless
        
        
    
    
        'manufacturing_eff_wafer_upd_PV': ['unif', [1,1-0.25*1,1+0.25*1, 0, 0],"."],
    
        'solargrade_electric_intensity_PV': ['unif', [0,11-0.25*11,11+0.25*11, 0, 0],"."],  # Besseau range  A CHECKER (110 in ecoivnent)
        
    
        "prob_switch_substielec":['switch',5,[1,0,0,0,0],["substit_margi","substit_PV","substit_margi_fr","substit_margi_it","substit_margi_es"],"."],
        
        "impact_update_margi":['unif',[1,1-0.25*1,1+0.25*1, 0, 0],"."]
    
        }
    
    
    
    
    
    
    def reset_unique(dict_):
        """Function that sets all the parameters of dict_ to their unique values."""
        
        for a in dict_:
            
            if dict_[a][0]=="unif":
                
                dict_[a][0]="unique"
                
                
        return(dict_)        
    
    
    
    
    
    # Put all the dictionnaries in one
    
    dictionnaries = {"Ag_par_distributions" : Ag_par_distributions,
                     "PV_par_distributions": PV_par_distributions,
                     "iluc_par_distributions":iluc_par_distributions}
    
    
    
    
    
    
    ####################
    """Define functions"""
    ######################
    
    
  
 
    
    
    
    #######
    """Functions parameterizing the LCA matrixes"""
    #######
    
    
    
    ######## PV
    
    
    # Mounting structure
    
      
    
    # 0.000541666666666667 ; initial input value (m3) for concrete
    def fconcrete_PV_mounting(init,
                               concrete_mount_upd_PV):
        """ Updates the amount of concrete for the mounting structure for conv PV"""
        
        new_value = init * concrete_mount_upd_PV  # init = original value in the matrix
        
        return new_value
    
    
    
    
    
    def fconcrete_AVS_mounting(init,
                               concrete_mount_ratio_avs_pv,
                               concrete_mount_upd_PV
                               ):
        """ Updates the amount of concrete for the mounting structure in the AVS"""
    
        new_value = init * concrete_mount_upd_PV * concrete_mount_ratio_avs_pv
        
        return new_value
    
    
        
      
    
    
    # aluminium
    
    def falum_PV_mounting(init,
                               aluminium_mount_upd_PV):
        """ Updates the amount of aluminium for the mounting structure for conv PV"""
    
        
        new_value = init * aluminium_mount_upd_PV    
        
        return new_value
    
    
    
    
    
    def falum_AVS_mounting(init,
                               aluminium_mount_ratio_avs_pv,
                               aluminium_mount_upd_PV
                               ):
        
        """ Updates the amount of aluminium for the mounting structure for AVS"""
    
        new_value = init * aluminium_mount_upd_PV * aluminium_mount_ratio_avs_pv
        
        return new_value
    
    
    
    
    
    # Steel
    
    
    def fsteel_PV_mounting(init,
                               steel_mount_upd_PV):
        
        """ Updates the amount of steel for the mounting structure for conv PV"""
    
        new_value = init * steel_mount_upd_PV   
        
        return new_value
    
    
    
    
    
    
    def fsteel_AVS_mounting(init,
                               steel_mount_ratio_avs_pv,
                               steel_mount_upd_PV
                               ):
        
        """ Updates the amount of steel for the mounting structure for AVS"""
       
        new_value = init * steel_mount_upd_PV * steel_mount_ratio_avs_pv
        
        return new_value
    
    
    # Corrugated board box
    
    
    
    def fcorrbox_PV_mounting(init,
                               corrbox_mount_upd_PV):
        
        """ Updates the amount of corrugated box for the mounting structure for conv PV"""
    
        new_value = init * corrbox_mount_upd_PV   
        
        return new_value
    
    
    
    
    
    
    def fcorrbox_AVS_mounting(init,
                               corrbox_mount_ratio_avs_pv,
                               corrbox_mount_upd_PV
                               ):
        """ Updates the amount of corrugated box for the mounting structure for AVS"""
    
        
        new_value = init * corrbox_mount_upd_PV * corrbox_mount_ratio_avs_pv
        
        return new_value
    
    
    # polytehylene et polystirene
    
    
    
    def fpoly_PV_mounting(init,
                               poly_mount_upd_PV):
        
        """ Updates the amount of plastic for the mounting structure for PV conv"""
    
        new_value = init * poly_mount_upd_PV 
        
        return new_value
    
    
    
    
    
    
    def fpoly_AVS_mounting(init,
                               poly_mount_ratio_avs_pv,
                               poly_mount_upd_PV
                               ):
        
        """ Updates the amount of plastic for the mounting structure for AVS"""
    
        
        new_value = init * poly_mount_upd_PV * poly_mount_ratio_avs_pv
        
        return new_value
    
    
    
    
    # zinc coat
    
    
    
    def fzinc_PV_mounting(init,
                               zinc_mount_upd_PV):
        
        """ Updates the amount of zinc for the mounting structure for conv PV"""
    
        
        new_value = init * zinc_mount_upd_PV    
    
        return new_value
    
    
    
    
    
    
    def fzinc_AVS_mounting(init,
                               zinc_mount_ratio_avs_pv,
                               zinc_mount_upd_PV
                               ):
        
        """ Updates the amount of zinc for the mounting structure for AVS"""
    
        new_value = init * zinc_mount_upd_PV * zinc_mount_ratio_avs_pv
        
        return new_value
    
    
    
    
    
    def f_mountingpersystem_AVS(init,
             
                      surface_cover_fraction_AVS,
                      surface_cover_fraction_PV,
                      plant_lifetime_PV,
                      plant_lifetime_ratio_avs_pv):
        
        """Sets the amount of mounting structure for 1 ha of AVS"""
    
        
        new_value = correct_init_whenreplace(init)* 10000 *surface_cover_fraction_AVS/(plant_lifetime_PV*plant_lifetime_ratio_avs_pv)
    
        return new_value
    
    
    def f_mountingpersystem_PV(init,
               
                      surface_cover_fraction_PV,
                      plant_lifetime_PV):
        
        """Sets the amount of mounting structure for 1 ha for conv PV"""
    
        
        new_value = correct_init_whenreplace(init) *10000 *surface_cover_fraction_PV /plant_lifetime_PV
    
        return new_value
    
    
    
    
    
    
    """ Electric installation"""
    
    
    # iF AVS or PV not mentioned ,same for both
    
    
    
    def f_electricinstallation_amount_perm2(init,
                                      mass_to_power_ratio_electric_instal_PV,
                                      panel_efficiency_PV):
        
        """Sets the amount of electric installation per m2 of panel"""
        
        new_value = correct_init_whenreplace(init)*mass_to_power_ratio_electric_instal_PV * panel_efficiency_PV
        
        # #print( mass_to_power_ratio_electric_instal_PV,panel_efficiency_PV,new_value )
        # #print("lEEEN", len(init))
        return new_value
    
    
    
    """Inverter"""
    
    # (25171, 25172)
    
    def f_input_inverter_m2_panel_AVS(init,
                                  plant_lifetime_PV,
                                  panel_efficiency_PV,
                                  inverter_specific_weight_PV,
                                  inverter_lifetime_PV,
                                  plant_lifetime_ratio_avs_pv
                                  ):
        
        """Sets the amount of inverter per m2 of panel for the AVS"""
    
        
        new_value = correct_init_whenreplace(init) *panel_efficiency_PV * inverter_specific_weight_PV * plant_lifetime_PV*plant_lifetime_ratio_avs_pv/inverter_lifetime_PV 
        
        return new_value
    
    
    # (25171, 25173)
    def f_input_inverter_m2_panel_PV(init,
                                  plant_lifetime_PV,
                                  panel_efficiency_PV,
                                  inverter_specific_weight_PV,
                                  inverter_lifetime_PV,
                                  ):
        
        """Sets the amount of inverter per m2 of panel for the conv PV"""
    
        new_value = correct_init_whenreplace(init) *panel_efficiency_PV * inverter_specific_weight_PV * plant_lifetime_PV/inverter_lifetime_PV 
        
        return new_value
    
    
    
    
    
    """ Panel """
    
    
    def f_outputAVS_crop_main(init,
                              # water_content_grain, carbon_content_grain,
                              # water_content_straw, straw_grain_ratio,
                              # carbon_content_straw, root_shoot_ratio,
                              wheat_switch,
                              soy_switch,
                              alfalfa_switch,
                              maize_switch,
                              crop_yield_ratio_avs_ref,
                              crop_yield_upd_ref,
                              init_yield_soy,
                              init_yield_wheat,
                              init_yield_alfalfa,
                              init_yield_maize,
                              correc_orchidee):
        
        """Sets the output of crop production for the AVS"""
        
        # This line computes the yield according to orchidee npp and the mean values of the conversion parameters
        new_value = correct_init_whenreplace(init) * (init_yield_alfalfa*alfalfa_switch + init_yield_soy*soy_switch + init_yield_wheat*wheat_switch+ init_yield_maize*maize_switch) * crop_yield_upd_ref*crop_yield_ratio_avs_ref
    
    
    
        
        """
        Calculate the new grain yield based on changes of conversion parameters.
        
    
        """
        # Apply changes to parameters
        
        # Apply changes to parameters
        
        
        # Original values
        
        # root_shoot_ratio_ori = 0.21
        # straw_grain_ratio_ori = 1.35
        # carbon_content_grain_ori = 0.4292
        # carbon_content_straw_ori = 0.446
        # water_content_grain_ori = 0.14
        # water_content_straw_ori = 0.14
        
        # # Helper function to calculate A
        # def calculate_A(wcg, ccg, wcs, sgr, ccs):
        #     return ((1 - wcg) * ccg + (1 - wcs) * sgr * ccs)
    
        # # Original A value
        # A_original = calculate_A(water_content_grain_ori, carbon_content_grain_ori, water_content_straw_ori, straw_grain_ratio_ori, carbon_content_straw_ori)
        
        # # New A value
        # A_new = calculate_A(water_content_grain, carbon_content_grain, water_content_straw, straw_grain_ratio, carbon_content_straw)
        
        # Calculate the new grain yield
        new_value = new_value * correc_orchidee
    
    
        return new_value
    
    
    
    def f_outputAVS_elec_main(init,
                      panel_efficiency_PV,
                      surface_cover_fraction_PV,
                      surface_cover_fraction_AVS,
                      annual_producible_PV,
                      annual_producible_ratio_avs_pv):
        
        """Sets the output of electricity production for the AVS"""
    
        
        new_value = correct_init_whenreplace(init) *10000 *surface_cover_fraction_AVS* panel_efficiency_PV * annual_producible_PV *annual_producible_ratio_avs_pv
    
    
        #print("f_outputAVS_elec_main",new_value)
    
        return new_value
    
    
    
    def f_outputAVS_elec_margi_crop_main(init,
                      panel_efficiency_PV,
                      surface_cover_fraction_PV,
                      surface_cover_fraction_AVS,
                      annual_producible_PV,
                      annual_producible_ratio_avs_pv,
                      substit_margi_switch):
        
        """Sets the output of electricity production for the AVS, when crop is the main product.
        For the substituion of marginal elec"""
    
        
        new_value = correct_init_whenreplace(init) *10000 *surface_cover_fraction_AVS* panel_efficiency_PV * annual_producible_PV *annual_producible_ratio_avs_pv
        
        
        new_value = new_value * substit_margi_switch
        
        ##print("f_outputAVS_elec_margi_crop_main",new_value)
    
        return new_value
    
    def f_outputAVS_elec_PV_crop_main(init,
                      panel_efficiency_PV,
                      surface_cover_fraction_PV,
                      surface_cover_fraction_AVS,
                      annual_producible_PV,
                      annual_producible_ratio_avs_pv,
                      substit_PV_switch):
        
        """Sets the output of electricity production for the AVS, when crop is the main product.
        For the substitution of conv PV"""
    
        
        new_value = correct_init_whenreplace(init) *10000 *surface_cover_fraction_AVS* panel_efficiency_PV * annual_producible_PV *annual_producible_ratio_avs_pv 
        
        new_value = new_value * substit_PV_switch
        
        # nan_count = sum(math.isnan(x) for x in new_value)
        # loc_nan = [math.isnan(x) for x in new_value]
        # if nan_count!=0:
        #     #print("loc_nan",loc_nan)
    
        ##print("nan_count",nan_count)
        
        ##print("f_outputAVS_elec_PV_crop_main",init)
    
        return new_value
        
    
    
    def f_outputAVS_elec_current_fr_crop_main(init,
                      panel_efficiency_PV,
                      surface_cover_fraction_PV,
                      surface_cover_fraction_AVS,
                      annual_producible_PV,
                      annual_producible_ratio_avs_pv,
                      substit_margi_fr_switch):
        
        """Sets the output of electricity production for the AVS, when crop is the main product.
        For the substitution of conv PV"""
    
        
        new_value = correct_init_whenreplace(init) *10000 *surface_cover_fraction_AVS* panel_efficiency_PV * annual_producible_PV *annual_producible_ratio_avs_pv 
        
        new_value = new_value * substit_margi_fr_switch
        
        # nan_count = sum(math.isnan(x) for x in new_value)
        # loc_nan = [math.isnan(x) for x in new_value]
        # if nan_count!=0:
        #     #print("loc_nan",loc_nan)
    
        ##print("nan_count",nan_count)
        
        ##print("f_outputAVS_elec_PV_crop_main",init)
    
        return new_value
    
    def f_outputAVS_elec_current_it_crop_main(init,
                      panel_efficiency_PV,
                      surface_cover_fraction_PV,
                      surface_cover_fraction_AVS,
                      annual_producible_PV,
                      annual_producible_ratio_avs_pv,
                      substit_margi_it_switch):
        
        """Sets the output of electricity production for the AVS, when crop is the main product.
        For the substitution of conv PV"""
    
        
        new_value = correct_init_whenreplace(init) *10000 *surface_cover_fraction_AVS* panel_efficiency_PV * annual_producible_PV *annual_producible_ratio_avs_pv 
        
        new_value = new_value * substit_margi_it_switch
        
        # nan_count = sum(math.isnan(x) for x in new_value)
        # loc_nan = [math.isnan(x) for x in new_value]
        # if nan_count!=0:
        #     #print("loc_nan",loc_nan)
    
        ##print("nan_count",nan_count)
        
        ##print("f_outputAVS_elec_PV_crop_main",init)
    
        return new_value
      
    def f_outputAVS_elec_current_es_crop_main(init,
                      panel_efficiency_PV,
                      surface_cover_fraction_PV,
                      surface_cover_fraction_AVS,
                      annual_producible_PV,
                      annual_producible_ratio_avs_pv,
                      substit_margi_es_switch):
        
        """Sets the output of electricity production for the AVS, when crop is the main product.
        For the substitution of conv PV"""
    
        
        new_value = correct_init_whenreplace(init) *10000 *surface_cover_fraction_AVS* panel_efficiency_PV * annual_producible_PV *annual_producible_ratio_avs_pv 
        
        new_value = new_value * substit_margi_es_switch
        
        # nan_count = sum(math.isnan(x) for x in new_value)
        # loc_nan = [math.isnan(x) for x in new_value]
        # if nan_count!=0:
        #     #print("loc_nan",loc_nan)
    
        ##print("nan_count",nan_count)
        
        ##print("f_outputAVS_elec_PV_crop_main",init)
    
        return new_value    
    
    
    def f_outputPV(init,
                      panel_efficiency_PV,
                      surface_cover_fraction_PV,
                      annual_producible_PV):
        
        """Sets the output of electricity production for the conv PV"""
    
        
        new_value = correct_init_whenreplace(init) *10000 *surface_cover_fraction_PV * panel_efficiency_PV* annual_producible_PV
        
    
        nan_count = sum(math.isnan(x) for x in new_value)
        if nan_count!=0:
            sys.exit("nan")
    
    
        return new_value
    
    
    def f_panelperAVS(init,
                      surface_cover_fraction_AVS,
                      plant_lifetime_PV,
                      plant_lifetime_ratio_avs_pv):
        
        """Sets the amount of m2 of pV panels, per ha, for the AVS"""
    
        
        new_value = correct_init_whenreplace(init) *10000 *surface_cover_fraction_AVS/(plant_lifetime_PV*plant_lifetime_ratio_avs_pv)
    
    
        return new_value
    
    def f_panelperPV(init,
                      surface_cover_fraction_PV,
                      plant_lifetime_PV):
        
        """Sets the amount of m2 of pV panels, per ha, for the conv PV"""
    
        new_value = correct_init_whenreplace(init) *10000 *surface_cover_fraction_PV /plant_lifetime_PV
        ##print("new_value",new_value)
        # new_value =correct_init_whenreplace(init)
        
        # #print(init)
        # #print("new_value",new_value)
        return new_value
    
    
    
    
    def f_aluminium_input_panel(init,
                                aluminium_panel_weight_PV):
        
        
        """Sets the amount of aluminium per panel frame"""
    
        new_value =  correct_init_whenreplace(init) *aluminium_panel_weight_PV 
    
        return new_value
    
    
    
    """wafer"""
    
    def f_inputelec_wafer(init,
                          manufacturing_eff_wafer_upd_PV): #"""both on row and rer"""
    
        """Sets the amount of elec per m2 of wafer"""
    
        
        new_value = init * manufacturing_eff_wafer_upd_PV
        
        return new_value
    
    
    
    
    
    
    
    
    """ Silicon production"""
    
     # As we remove the hydro elec input (does not make sense in most cases for marginal ) which was 65 out ok 110kwH, the new original consumption is 45
    def f_elec_intensity_solargraderow(init,
                                        solargrade_electric_intensity_PV):
        
        """ Updates the elec consumpotion for the silicon manufacture , for the row"""
        
        new_value = init*solargrade_electric_intensity_PV/45 # 45 kWh in total (without hydro) in the original act
    
        return new_value
    
    
    def f_elec_intensity_solargraderer(init,
                                        solargrade_electric_intensity_PV):
        
        """ Updates the elec consumpotion for the silicon manufacture , for the rer"""
    
        
        new_value = init*solargrade_electric_intensity_PV/45 
        
        return new_value
    
    
    
    
    
    
    """iLUC"""
    
    
    def f_NPP_weighted_ha_y_landmarket_cropref(init,
           NPP_weighted_ha_y_eq_cropref):
        
        """ Sets the amount of NPP equivalent for 1 ha used by the conventional crop """
        
        new_value = correct_init_whenreplace(init) * NPP_weighted_ha_y_eq_cropref
        
        return new_value
    
    
    
    
    def f_NPP_weighted_ha_y_landmarket_AVS(init,
           NPP_weighted_ha_y_eq_cropref,
           NPP_weighted_ha_y_eq_ratio_avs_cropref):
        
        """ Sets the amount of NPP equivalent for 1 ha used by the AVS """
    
        
        new_value = correct_init_whenreplace(init) * NPP_weighted_ha_y_eq_cropref * NPP_weighted_ha_y_eq_ratio_avs_cropref
        
        return new_value
    
    
    def f_NPP_weighted_ha_y_landmarket_PV(init,
           NPP_weighted_ha_y_eq_cropref,
           NPP_weighted_ha_y_eq_ratio_pv_cropref):
        
        """ Sets the amount of NPP equivalent for 1 ha used by the PV """
    
        
        new_value = correct_init_whenreplace(init) * NPP_weighted_ha_y_eq_cropref * NPP_weighted_ha_y_eq_ratio_pv_cropref
        
        return new_value
    
    
    
    
    def f_iluc_landmarket_PV(init,
           iluc_lm_PV):
        
        """ iluc GW impact for 1  NPP equivalent for the conv PV landmarket"""
    
        
        new_value = correct_init_whenreplace(init)*iluc_lm_PV
        
        return new_value
    
    
    def f_iluc_landmarket_AVS(init,
           iluc_lm_AVS):
        
        """ iluc GW impact for 1 NPP equivalent for the AVS landmarket"""
    
        
        new_value = correct_init_whenreplace(init)*iluc_lm_AVS
        
        return new_value
    
    
    def f_iluc_landmarket_cropref(init,
           iluc_lm_cropref):
        
        """ iluc GW impact for 1 NPP equivalent for the crop landmarket"""
    
        new_value = correct_init_whenreplace(init)*iluc_lm_cropref
        
        return new_value
    
    
    
    """ Agri """
    
    def f_switch_wheat(init,
                     wheat_switch):
        
        """ Assigns 0 or 1 to choose wheat as the modeled crop"""
    
        
        new_value = correct_init_whenreplace(init) * wheat_switch
        
        return new_value
    
    
    
    def f_switch_soy(init,
                     soy_switch):
        
        """ Assigns 0 or 1 to choose soy as the modeled crop"""
    
        new_value = correct_init_whenreplace(init) * soy_switch
        
        return new_value
    
    def f_switch_alfalfa(init,
                     alfalfa_switch):
        
        """ Assigns 0 or 1 to choose alfalfa as the modeled crop"""
    
        
        new_value = correct_init_whenreplace(init) * alfalfa_switch
        
        return new_value
    
    def f_switch_maize(init,
                     maize_switch):
        
        """ Assigns 0 or 1 to choose alfalfa as the modeled crop"""
    
        
        new_value = correct_init_whenreplace(init) * maize_switch
        
        return new_value    
    
    
    
    # def f_output_crop_ref(init,
    #                    crop_yield_upd_ref):
        
    #     """ Updates the output of the conv crop production"""
    
        
    #     new_value = init*crop_yield_upd_ref
    #     #print("f_output_crop_ref",new_value)
        
    #     return new_value
    
    
    
    

    
    def f_output_crop_ref(init,
                          # water_content_grain, carbon_content_grain,
                          # water_content_straw, straw_grain_ratio,
                          # carbon_content_straw, root_shoot_ratio,
                          crop_yield_upd_ref,
                          correc_orchidee):
        
        """ Updates the output of the conv crop production"""
        
        #We'll keep the npp
        
        
        new_value = init*crop_yield_upd_ref # init it the orinal value in ecoinvent, 
        
        #print("f_output_crop_ref",new_value)
        
        
        """
        Calculate the new grain yield based on changes of conversion parameters.
        
    
        """
        # Original values
        
        # root_shoot_ratio_ori = 0.21
        # straw_grain_ratio_ori = 1.35
        # carbon_content_grain_ori = 0.4292
        # carbon_content_straw_ori = 0.446
        # water_content_grain_ori = 0.14
        # water_content_straw_ori = 0.14
        
        # # Helper function to calculate A
        # def calculate_A(wcg, ccg, wcs, sgr, ccs):
        #     return ((1 - wcg) * ccg + (1 - wcs) * sgr * ccs)
    
        # # Original A value
        # A_original = calculate_A(water_content_grain_ori, carbon_content_grain_ori, water_content_straw_ori, straw_grain_ratio_ori, carbon_content_straw_ori)
        
        # # New A value
        # A_new = calculate_A(water_content_grain, carbon_content_grain, water_content_straw, straw_grain_ratio, carbon_content_straw)
        
        # Calculate the new grain yield
        new_value = new_value * correc_orchidee
            
        
        return new_value
    
    
    
    def f_output_crop_avs(init,
                          # water_content_grain,
                          # carbon_content_grain, 
                          # water_content_straw,
                          # straw_grain_ratio,
                          # carbon_content_straw,
                          # root_shoot_ratio,
                       crop_yield_upd_ref,
                       crop_yield_ratio_avs_ref,
                       correc_orchidee):
        
        """ Updates the output of the AVS crop production"""
    
        
        new_value = init*crop_yield_upd_ref *crop_yield_ratio_avs_ref
        
        #print("f_output_crop_avs",new_value)
        
        
        
        """
        Calculate the new grain yield based on changes of conversion parameters.
        
    
        """
        # Original values
        
        # root_shoot_ratio_ori = 0.21
        # straw_grain_ratio_ori = 1.35
        # carbon_content_grain_ori = 0.4292
        # carbon_content_straw_ori = 0.446
        # water_content_grain_ori = 0.14
        # water_content_straw_ori = 0.14
        
        # # Helper function to calculate A
        # def calculate_A(wcg, ccg, wcs, sgr, ccs):
        #     return ((1 - wcg) * ccg + (1 - wcs) * sgr * ccs)
    
        # # Original A value
        # A_original = calculate_A(water_content_grain_ori, carbon_content_grain_ori, water_content_straw_ori, straw_grain_ratio_ori, carbon_content_straw_ori)
        
        # # New A value
        # A_new = calculate_A(water_content_grain, carbon_content_grain, water_content_straw, straw_grain_ratio, carbon_content_straw)
        
        # Calculate the new grain yield
        new_value = new_value * correc_orchidee
            
        
        
    
        return new_value
    
    
    
    def f_fert_input_ref(init,
                          crop_fert_upd_ref):
        
        """ Updates the amount of fertilizer input to conv crop production"""
    
        
        new_value = init *crop_fert_upd_ref
        
        return new_value
    
    
    
    
    def f_fert_input_avs(init,
                           crop_fert_upd_ref,
                           crop_fert_ratio_avs_ref):
        
        """ Updates the amount of fertilizer input to AVS"""
    
        
        new_value =  init * crop_fert_upd_ref *crop_fert_ratio_avs_ref
        
        return new_value
    
    
    def f_nutri_emission_ref(init,
                       crop_fert_upd_ref):
        
        """ Updates the amount of emissions associated with fertilizer for conv crop prod"""
    
        
        new_value = init *crop_fert_upd_ref
        
        return new_value
    
    
    
    def f_nutri_emission_avs(init,
                       crop_fert_upd_ref,
                       crop_fert_ratio_avs_ref):
        
        """ Updates the amount of emissions associated with fertilizer for AVS"""
    
        
        new_value = init * crop_fert_upd_ref *crop_fert_ratio_avs_ref
        
        return new_value
    
    
    
    def f_machinery_ref(init,
                       crop_mach_upd_ref):
        
        """ Updates the amount of machinery for conv crop prod"""
    
        
        new_value = init * crop_mach_upd_ref 
        
        return new_value
    
    
    
    def f_machinery_avs(init,
                       crop_mach_upd_ref,
                       crop_mach_ratio_avs_ref):
        
        """ Updates the amount of machinery for AVS"""
    
        
        new_value = init * crop_mach_upd_ref * crop_mach_ratio_avs_ref
        
        return new_value
    
    
    
    def f_water_ref(init,
                       water_upd_ref):
        
        """ Updates the amount of irrigated water for conv crop"""
    
        
        new_value = init * water_upd_ref
        
        return new_value
    
    
    def f_water_avs(init,
                    water_upd_ref,
                       water_ratio_avs_ref):
        
        """ Updates the amount of irrigated water for avs"""
    
        
        new_value = init * water_upd_ref * water_ratio_avs_ref
        
        return new_value
    
    
    
    def f_carbon_soil_accumulation_ref(init,
                    carbon_accumulation_soil_ref):
        
        """ Sets the amount of carbon accumulated for the conv crop"""
    
        
        new_value = correct_init_whenreplace(init) * carbon_accumulation_soil_ref
        
        return new_value
    
    
    def f_carbon_soil_accumulation_avs(init,
                    carbon_accumulation_soil_AVS):
        
        """ Sets the amount of carbon accumulated for AVS"""
    
        
        new_value = correct_init_whenreplace(init) * carbon_accumulation_soil_AVS
        
        return new_value
    
    
    def f_carbon_soil_accumulation_pv(init,
                    carbon_accumulation_soil_PV):
        
        """ Sets the amount of carbon accumulated for PV"""
    
        
        new_value = correct_init_whenreplace(init) * carbon_accumulation_soil_PV
        
        return new_value
    
    
    
    
    
    # Multi marginal_elec
    
    def modif_impact_marginal_elec(init,
                    impact_update_margi):
        
        """  Updates the impact of  the marginal electricity"""
    
        
        new_value = init * (impact_update_margi-1)
        
        return new_value
    
    
    
    
    
    def fdelete(init,
                       cover_av,
                       lifetime):
        
        """deletes a flow"""
        
        new_value = 0
        
        return new_value
    
    

    # print(electricity_rer_si_sg)
    # print(si_sg_rer)
    
    
    """ Collects the functions in a dictionnary, together with the locations where they should be applied in the matrix """
    
    dict_funct = { "f1": {"func":vectorize_function(fconcrete_PV_mounting), # returns a vectorized function applying the function on the vector of stochastic parameters values
                          "indices":[(concrete_mount.id,mount_system_PV.id),  #   # where it should apply on the matrix (row, col) : the position of the flow to modify
                                    (concrete_mount_waste_1.id,mount_system_PV.id),
                                    (concrete_mount_waste_2.id,mount_system_PV.id),
                                    (concrete_mount_waste_3.id,mount_system_PV.id)]},    
    
                    "f2": {"func":vectorize_function(fconcrete_AVS_mounting),     
                                        "indices":[(concrete_mount.id,mount_system_AVS.id),
                                                    (concrete_mount_waste_1.id,mount_system_AVS.id),
                                                    (concrete_mount_waste_2.id,mount_system_AVS.id),
                                                    (concrete_mount_waste_3.id,mount_system_AVS.id)]},
    
                    "f3": {"func":vectorize_function(falum_PV_mounting), # ok 17.03
                                          "indices":[(aluminium_mount.id,mount_system_PV.id),
                                                    (alu_extru_mount.id,mount_system_PV.id),
                                                    (alu_mount_scrap_1.id,mount_system_PV.id),
                                                    (alu_mount_scrap_2.id,mount_system_PV.id),
                                                    (alu_mount_scrap_3.id,mount_system_PV.id)]},
                  
    
                    "f4": {"func":vectorize_function(falum_AVS_mounting),   # ok 17.03
                                          "indices":[(aluminium_mount.id,mount_system_AVS.id),
                                                    (alu_extru_mount.id,mount_system_AVS.id),
                                                    (alu_mount_scrap_1.id,mount_system_AVS.id),
                                                    (alu_mount_scrap_2.id,mount_system_AVS.id),
                                                    (alu_mount_scrap_3.id,mount_system_AVS.id)]},
                  
                    "f5": {"func":vectorize_function(fsteel_PV_mounting),   # ok 17.03
                                          "indices":[(reinf_steel_mount.id,mount_system_PV.id),
                                                    (chrom_steel_mount.id,mount_system_PV.id),
                                                    (steel_rolling_mount.id,mount_system_PV.id),
                                                    (wire_mount.id,mount_system_PV.id),
                                                    (steel_mount_scrap_1.id,mount_system_PV.id),
                                                    (steel_mount_scrap_2.id,mount_system_PV.id),
                                                    (steel_mount_scrap_3.id,mount_system_PV.id)]},
                  
                    "f6": {"func":vectorize_function(fsteel_AVS_mounting),   # ok 17.03
                                          "indices":[(reinf_steel_mount.id,mount_system_AVS.id),
                                                    (chrom_steel_mount.id,mount_system_AVS.id),
                                                    (steel_rolling_mount.id,mount_system_AVS.id),
                                                    (wire_mount.id,mount_system_AVS.id),
                                                    (steel_mount_scrap_1.id,mount_system_AVS.id),
                                                    (steel_mount_scrap_2.id,mount_system_AVS.id),
                                                    (steel_mount_scrap_3.id,mount_system_AVS.id)]},
                  
                    "f7": {"func":vectorize_function(fcorrbox_PV_mounting),  # ok 17.03
                                          "indices":[(cor_box_mount_1.id,mount_system_PV.id),
                                                    (cor_box_mount_2.id,mount_system_PV.id),
                                                    (cor_box_mount_3.id,mount_system_PV.id),
                                                    (cor_box_mount_waste_1.id,mount_system_PV.id),
                                                    (cor_box_mount_waste_2.id,mount_system_PV.id),
                                                    (cor_box_mount_waste_3.id,mount_system_PV.id)]},
                  
                    "f8": {"func":vectorize_function(fcorrbox_AVS_mounting), # ok 17.03
                                          "indices":[(cor_box_mount_1.id,mount_system_AVS.id),
                                                    (cor_box_mount_2.id,mount_system_AVS.id),
                                                    (cor_box_mount_3.id,mount_system_AVS.id),
                                                    (cor_box_mount_waste_1.id,mount_system_AVS.id),
                                                    (cor_box_mount_waste_2.id,mount_system_AVS.id),
                                                    (cor_box_mount_waste_3.id,mount_system_AVS.id)]},
                  
                    "f9": {"func":vectorize_function(fpoly_PV_mounting), # ok 17.03
                                          "indices":[(poly_mount_1.id,mount_system_PV.id),
                                                    (poly_mount_2.id,mount_system_PV.id),
                                                    (poly_mount_waste_1.id,mount_system_PV.id),
                                                    (poly_mount_waste_2.id,mount_system_PV.id),
                                                    (poly_mount_waste_3.id,mount_system_PV.id),
                                                    (poly_mount_waste_4.id,mount_system_PV.id),
                                                    (poly_mount_waste_5.id,mount_system_PV.id),
                                                    (poly_mount_waste_6.id,mount_system_PV.id)]},              
                  
                      "f10": {"func":vectorize_function(fpoly_AVS_mounting),  # ok 17.03
                                          "indices":[(poly_mount_1.id,mount_system_AVS.id),
                                                      (poly_mount_2.id,mount_system_AVS.id),
                                                      (poly_mount_waste_1.id,mount_system_AVS.id),
                                                      (poly_mount_waste_2.id,mount_system_AVS.id),
                                                      (poly_mount_waste_3.id,mount_system_AVS.id),
                                                      (poly_mount_waste_4.id,mount_system_AVS.id),
                                                      (poly_mount_waste_5.id,mount_system_AVS.id),
                                                      (poly_mount_waste_6.id,mount_system_AVS.id)]},
                  
                      "f11": {"func":vectorize_function(fzinc_PV_mounting), # ok 17.03
                                          "indices":[(zinc_coat_mount_1.id,mount_system_PV.id),
                                                      (zinc_coat_mount_2.id,mount_system_PV.id)]},
                  
                      "f12": {"func":vectorize_function(fzinc_AVS_mounting), # ok 17.03
                                          "indices":[(zinc_coat_mount_1.id,mount_system_AVS.id),
                                                      (zinc_coat_mount_2.id,mount_system_AVS.id)]},
                  
                  
                  
                   # Electric installation
                  
                  
                      "f13": {"func":vectorize_function(f_electricinstallation_amount_perm2),
                                           "indices":[(elecinstakg.id, pv_insta_AVS.id), # Into AVS 
                                                      (elecinstakg.id, pv_insta_AVS_single.id),
                                                      (elecinstakg.id, pv_insta_AVS_multi.id),
                                                      (elecinstakg.id, pv_insta_AVS_cis.id),
                                                      (elecinstakg.id, pv_insta_AVS_cdte.id),
                                                      
                                                      (elecinstakg.id, pv_insta_PV.id),
                                                      (elecinstakg.id, pv_insta_PV_single.id),
                                                      (elecinstakg.id, pv_insta_PV_multi.id),
                                                      (elecinstakg.id, pv_insta_PV_cis.id),
                                                      (elecinstakg.id, pv_insta_PV_cdte.id)]}, # Into PV  # Ok 17.03
                  
    
                  
                  
                  # #   # AVS 
                  
                        "f14": {"func":vectorize_function(f_outputAVS_elec_main),
                                             "indices":[(AVS_elec_main.id, AVS_elec_main.id),
                                                        (AVS_elec_main_single.id, AVS_elec_main_single.id),
                                                        (AVS_elec_main_multi.id, AVS_elec_main_multi.id),
                                                        (AVS_elec_main_cis.id, AVS_elec_main_cis.id),
                                                        (AVS_elec_main_cdte.id, AVS_elec_main_cdte.id)]},  # Ok  17.03
                  
                       "f15": {"func":vectorize_function(f_outputPV),
                                            "indices":[(PV_ref.id, PV_ref.id),
                                                       (PV_ref_single.id, PV_ref_single.id),
                                                       (PV_ref_multi.id, PV_ref_multi.id),
                                                       (PV_ref_cis.id, PV_ref_cis.id),
                                                       (PV_ref_cdte.id, PV_ref_cdte.id)]}, # Ok 17.03
                       
                       "f16": {"func":vectorize_function(f_outputAVS_elec_margi_crop_main),
                                            "indices":[(elec_marginal_fr_copy.id, AVS_crop_main.id),
                                                       (elec_marginal_fr_copy.id, AVS_crop_main_single.id),
                                                       (elec_marginal_fr_copy.id, AVS_crop_main_multi.id),
                                                       (elec_marginal_fr_copy.id, AVS_crop_main_cis.id),
                                                       (elec_marginal_fr_copy.id, AVS_crop_main_cdte.id)]}, # Ok 17.03      
                       
                    
                       "f17": {"func":vectorize_function(f_outputAVS_elec_PV_crop_main),
                                            "indices":[(PV_ref.id, AVS_crop_main.id),
                                                       (PV_ref_single.id, AVS_crop_main_single.id),
                                                       (PV_ref_multi.id, AVS_crop_main_multi.id),
                                                       (PV_ref_cis.id, AVS_crop_main_cis.id),
                                                       (PV_ref_cdte.id, AVS_crop_main_cdte.id)]}, # Ok 17.03      
                       
                       "f18": {"func":vectorize_function(f_outputAVS_elec_current_fr_crop_main),
                                            "indices":[(elec_marginal_fr_current_copy.id, AVS_crop_main.id),
                                                       (elec_marginal_fr_current_copy.id, AVS_crop_main_single.id),
                                                       (elec_marginal_fr_current_copy.id, AVS_crop_main_multi.id),
                                                       (elec_marginal_fr_current_copy.id, AVS_crop_main_cis.id),
                                                       (elec_marginal_fr_current_copy.id, AVS_crop_main_cdte.id)]}, # Ok 17.03      
                       
                       "f19": {"func":vectorize_function(f_outputAVS_elec_current_es_crop_main),
                                            "indices":[(elec_marginal_es_current_copy.id, AVS_crop_main.id),
                                                       (elec_marginal_es_current_copy.id, AVS_crop_main_single.id),
                                                       (elec_marginal_es_current_copy.id, AVS_crop_main_multi.id),
                                                       (elec_marginal_es_current_copy.id, AVS_crop_main_cis.id),
                                                       (elec_marginal_es_current_copy.id, AVS_crop_main_cdte.id)]}, # Ok 17.03      
                                          
                       "f20": {"func":vectorize_function(f_outputAVS_elec_current_it_crop_main),
                                            "indices":[(elec_marginal_it_current_copy.id, AVS_crop_main.id),
                                                       (elec_marginal_it_current_copy.id, AVS_crop_main_single.id),
                                                       (elec_marginal_it_current_copy.id, AVS_crop_main_multi.id),
                                                       (elec_marginal_it_current_copy.id, AVS_crop_main_cis.id),
                                                       (elec_marginal_it_current_copy.id, AVS_crop_main_cdte.id)]}, # Ok 17.03      
                                          
                    
    
                    
                    
                    
                    
                    
                    
                    
                  
                      "f21": {"func":vectorize_function(f_panelperAVS),   # Ok 17.03
                                            "indices":[(pv_insta_AVS.id, AVS_elec_main.id),
                                                       (pv_insta_AVS_single.id, AVS_elec_main_single.id),
                                                       (pv_insta_AVS_multi.id, AVS_elec_main_multi.id),
                                                       (pv_insta_AVS_cis.id, AVS_elec_main_cis.id),
                                                       (pv_insta_AVS_cdte.id, AVS_elec_main_cdte.id),
                                                       
                                                       (pv_insta_AVS.id, AVS_crop_main.id),
                                                       (pv_insta_AVS_single.id, AVS_crop_main_single.id),
                                                       (pv_insta_AVS_multi.id, AVS_crop_main_multi.id),
                                                       (pv_insta_AVS_cis.id, AVS_crop_main_cis.id),
                                                       (pv_insta_AVS_cdte.id, AVS_crop_main_cdte.id)]},
                  
    
                      "f22": {"func":vectorize_function(f_panelperPV),
                                             "indices":[(pv_insta_PV.id, PV_ref.id),
                                                        (pv_insta_PV_single.id, PV_ref_single.id),
                                                        (pv_insta_PV_multi.id, PV_ref_multi.id),
                                                        (pv_insta_PV_cis.id, PV_ref_cis.id),
                                                        (pv_insta_PV_cdte.id, PV_ref_cdte.id)]}, # Ok 17.03
       
                      "f23": {"func":vectorize_function(f_mountingpersystem_AVS),
                                             "indices":[(mount_system_AVS.id, AVS_elec_main.id),
                                                        (mount_system_AVS.id, AVS_elec_main_single.id),
                                                        (mount_system_AVS.id, AVS_elec_main_multi.id),
                                                        (mount_system_AVS.id, AVS_elec_main_cis.id),
                                                        (mount_system_AVS.id, AVS_elec_main_cdte.id),
                                                        
                                                        (mount_system_AVS.id, AVS_crop_main.id),
                                                        (mount_system_AVS.id, AVS_crop_main_single.id),
                                                        (mount_system_AVS.id, AVS_crop_main_multi.id),
                                                        (mount_system_AVS.id, AVS_crop_main_cis.id),
                                                        (mount_system_AVS.id, AVS_crop_main_cdte.id)]}, # Ok 17.03
                  
                      "f24": {"func":vectorize_function(f_mountingpersystem_PV),
                                             "indices":[(mount_system_PV.id, PV_ref.id),
                                                        (mount_system_PV.id, PV_ref_single.id),
                                                        (mount_system_PV.id, PV_ref_multi.id),
                                                        (mount_system_PV.id, PV_ref_cis.id),
                                                        (mount_system_PV.id, PV_ref_cdte.id)]}, # Ok 17.03
    
    
    
    
                     #PV
                  
                  
                        "f25": {"func":vectorize_function(f_aluminium_input_panel), # Ok 17.03
                                              "indices":[(aluminium_panel.id, pvpanel_prod_row.id),
                                                        (aluminium_panel.id, pvpanel_prod_rer.id),
                                                        (aluminium_panel.id, pvpanel_prod_multi_rer.id),
                                                        (aluminium_panel.id, pvpanel_prod_multi_row.id)]},
                  
                        "f26": {"func":vectorize_function(f_inputelec_wafer), # Ok 17.03
                                             "indices":[(elec_wafer_nz.id, wafer_row.id),
                                                        (elec_wafer_rla.id, wafer_row.id),
                                                        (elec_wafer_raf.id, wafer_row.id),
                                                        (elec_wafer_au.id, wafer_row.id),
                                                        (elec_wafer_ci.id, wafer_row.id),
                                                        (elec_wafer_rna.id, wafer_row.id),
                                                        (elec_wafer_ras.id,wafer_row.id),
                                                        (elec_wafer_rer_single.id, wafer_rer.id),
                                                        
                                                        (elec_wafer_rer_multi.id,wafer_multisi_rer.id),
                                                        (elec_wafer_nz.id, wafer_multisi_row.id),
                                                        (elec_wafer_rla.id, wafer_multisi_row.id),
                                                        (elec_wafer_raf.id, wafer_multisi_row.id),
                                                        (elec_wafer_au.id, wafer_multisi_row.id),
                                                        (elec_wafer_ci.id, wafer_multisi_row.id),
                                                        (elec_wafer_rna.id, wafer_multisi_row.id),
                                                        (elec_wafer_ras.id, wafer_multisi_row.id),
                                                  
                                                        ]},
                  
                    

                  
                   #  # Silicon
                  
                        "f27": {"func":vectorize_function(f_elec_intensity_solargraderow),# Ok 17.03
                                              "indices":[(elec_sili_raf.id, si_sg_row.id),
                                                        (elec_sili_au.id, si_sg_row.id),
                                                        (elec_sili_ci.id, si_sg_row.id),
                                                        (elec_sili_nz.id, si_sg_row.id),
                                                        (elec_sili_ras.id, si_sg_row.id),
                                                        (elec_sili_rna.id, si_sg_row.id),
                                                        (elec_sili_rla.id, si_sg_row.id)
                                                  
                                                  
                                                        ]},    

                        
                        "f28": {"func":vectorize_function(f_elec_intensity_solargraderer), # Ok 17.03
                                              "indices":[(electricity_rer_si_sg.id, si_sg_rer.id)
                                                         ]},   
    
    
    
                   #    # Inverter
                    
                    
                    
                      "f29": {"func":vectorize_function(f_input_inverter_m2_panel_AVS),
                                           "indices":[(mark_inv_500kW_kg.id, pv_insta_AVS.id),
                                                      (mark_inv_500kW_kg.id, pv_insta_AVS_single.id),
                                                      (mark_inv_500kW_kg.id, pv_insta_AVS_multi.id),
                                                      (mark_inv_500kW_kg.id, pv_insta_AVS_cis.id),
                                                      (mark_inv_500kW_kg.id, pv_insta_AVS_cdte.id)]},    # ok 17.03
    
                    
                      "f30": {"func":vectorize_function(f_input_inverter_m2_panel_PV),
                                           "indices":[(mark_inv_500kW_kg.id, pv_insta_PV.id),
                                                      (mark_inv_500kW_kg.id, pv_insta_PV_single.id),
                                                      (mark_inv_500kW_kg.id, pv_insta_PV_multi.id),
                                                      (mark_inv_500kW_kg.id, pv_insta_PV_cis.id),
                                                      (mark_inv_500kW_kg.id, pv_insta_PV_cdte.id)]},     # ok 17.03
    
                            
                   #  # iLUC
                  
                      "f31": {"func":vectorize_function(f_NPP_weighted_ha_y_landmarket_cropref),
                                           "indices":[(LUCmarket_cropref.id, wheat_fr_ref.id),
                                                      (LUCmarket_cropref.id, soy_ch_ref.id),
                                                      (LUCmarket_cropref.id, alfalfa_ch_ref.id),
                                                      (LUCmarket_cropref.id, maize_ch_ref.id)]},     # ok 17.03
                      #explore_act(wheat_fr_ref)
                    
                    
                    
                       "f32": {"func":vectorize_function(f_NPP_weighted_ha_y_landmarket_AVS),
                                            "indices":[(LUCmarket_AVS.id, AVS_elec_main.id),
                                                       (LUCmarket_AVS.id, AVS_elec_main_single.id),
                                                       (LUCmarket_AVS.id, AVS_elec_main_multi.id),
                                                       (LUCmarket_AVS.id, AVS_elec_main_cis.id),
                                                       (LUCmarket_AVS.id, AVS_elec_main_cdte.id),
                                                       
                                                       (LUCmarket_AVS.id, AVS_crop_main.id),
                                                       (LUCmarket_AVS.id, AVS_crop_main_single.id),
                                                       (LUCmarket_AVS.id, AVS_crop_main_multi.id),
                                                       (LUCmarket_AVS.id, AVS_crop_main_cis.id),
                                                       (LUCmarket_AVS.id, AVS_crop_main_cdte.id)]},     # ok 17.03
    
                       "f33": {"func":vectorize_function(f_NPP_weighted_ha_y_landmarket_PV),
                                            "indices":[(LUCmarket_PVref.id, PV_ref.id),
                                                       (LUCmarket_PVref.id, PV_ref_single.id),
                                                       (LUCmarket_PVref.id, PV_ref_multi.id),
                                                       (LUCmarket_PVref.id, PV_ref_cis.id),
                                                       (LUCmarket_PVref.id, PV_ref_cdte.id)]},     # ok 17.03
    
                       "f34": {"func":vectorize_function(f_iluc_landmarket_PV),
                                            "indices":[(iluc.id, LUCmarket_PVref.id)]},     # ok 17.03
                 
                       "f35": {"func":vectorize_function(f_iluc_landmarket_AVS),
                                            "indices":[(iluc.id, LUCmarket_AVS.id)]}, 
                  
                       "f36": {"func":vectorize_function(f_iluc_landmarket_cropref),
                                             "indices":[(iluc.id, LUCmarket_cropref.id)]},
                  
                   # # # Agri
    
    
                     "f37": {"func":vectorize_function(f_switch_wheat),
                                          "indices":[(wheat_fr_AVS_elec_main.id, AVS_elec_main.id),
                                                     (wheat_fr_AVS_elec_main.id, AVS_elec_main_single.id),
                                                     (wheat_fr_AVS_elec_main.id, AVS_elec_main_multi.id),
                                                     (wheat_fr_AVS_elec_main.id, AVS_elec_main_cis.id),
                                                     (wheat_fr_AVS_elec_main.id, AVS_elec_main_cdte.id),
                                                     
                                                     (wheat_fr_AVS_crop_main.id, AVS_crop_main.id),
                                                     (wheat_fr_AVS_crop_main.id, AVS_crop_main_single.id),
                                                     (wheat_fr_AVS_crop_main.id, AVS_crop_main_multi.id),
                                                     (wheat_fr_AVS_crop_main.id, AVS_crop_main_cis.id),
                                                     (wheat_fr_AVS_crop_main.id, AVS_crop_main_cdte.id)]} ,             
                  
                     "f38": {"func":vectorize_function(f_switch_soy),
                                         "indices":[(soy_ch_AVS_elec_main.id, AVS_elec_main.id),
                                                    (soy_ch_AVS_elec_main.id, AVS_elec_main_single.id),
                                                    (soy_ch_AVS_elec_main.id, AVS_elec_main_multi.id),
                                                    (soy_ch_AVS_elec_main.id, AVS_elec_main_cis.id),
                                                    (soy_ch_AVS_elec_main.id, AVS_elec_main_cdte.id),
                                                    
                                                    (soy_ch_AVS_crop_main.id, AVS_crop_main.id),
                                                    (soy_ch_AVS_crop_main.id, AVS_crop_main_single.id),
                                                    (soy_ch_AVS_crop_main.id, AVS_crop_main_multi.id),
                                                    (soy_ch_AVS_crop_main.id, AVS_crop_main_cis.id),
                                                    (soy_ch_AVS_crop_main.id, AVS_crop_main_cdte.id)]},   
                     
                     "f39": {"func":vectorize_function(f_switch_alfalfa),
                                          "indices":[(alfalfa_ch_AVS_elec_main.id, AVS_elec_main.id),
                                                     (alfalfa_ch_AVS_elec_main.id, AVS_elec_main_single.id),
                                                     (alfalfa_ch_AVS_elec_main.id, AVS_elec_main_multi.id),
                                                     (alfalfa_ch_AVS_elec_main.id, AVS_elec_main_cis.id),
                                                     (alfalfa_ch_AVS_elec_main.id, AVS_elec_main_cdte.id),
                                           
                                                     (alfalfa_ch_AVS_crop_main.id, AVS_crop_main.id),
                                                     (alfalfa_ch_AVS_crop_main.id, AVS_crop_main_single.id),
                                                     (alfalfa_ch_AVS_crop_main.id, AVS_crop_main_multi.id),
                                                     (alfalfa_ch_AVS_crop_main.id, AVS_crop_main_cis.id),
                                                     (alfalfa_ch_AVS_crop_main.id, AVS_crop_main_cdte.id)]},

                     "f40": {"func":vectorize_function(f_switch_maize),
                                          "indices":[(maize_ch_AVS_elec_main.id, AVS_elec_main.id),
                                                     (maize_ch_AVS_elec_main.id, AVS_elec_main_single.id),
                                                     (maize_ch_AVS_elec_main.id, AVS_elec_main_multi.id),
                                                     (maize_ch_AVS_elec_main.id, AVS_elec_main_cis.id),
                                                     (maize_ch_AVS_elec_main.id, AVS_elec_main_cdte.id),
                                                     
                                                     (maize_ch_AVS_crop_main.id, AVS_crop_main.id),
                                                     (maize_ch_AVS_crop_main.id, AVS_crop_main_single.id),
                                                     (maize_ch_AVS_crop_main.id, AVS_crop_main_multi.id),
                                                     (maize_ch_AVS_crop_main.id, AVS_crop_main_cis.id),
                                                     (maize_ch_AVS_crop_main.id, AVS_crop_main_cdte.id)]},
                     
                     "f41": {"func":vectorize_function(f_output_crop_ref),
                                          "indices":[(soy_ch_ref.id, soy_ch_ref.id),
                                                     (wheat_fr_ref.id, wheat_fr_ref.id),
                                                     (alfalfa_ch_ref.id, alfalfa_ch_ref.id),
                                                     (maize_ch_ref.id, maize_ch_ref.id)
                                                     ]},   
                  
    
                     "f42": {"func":vectorize_function(f_output_crop_avs),
                                          "indices":[(wheat_fr_ref.id, wheat_fr_AVS_elec_main.id),
                                                     (soy_ch_ref.id, soy_ch_AVS_elec_main.id),
                                                     (alfalfa_ch_ref.id, alfalfa_ch_AVS_elec_main.id),
                                                     (maize_ch_ref.id,maize_ch_AVS_elec_main.id)]},               
                                                     
                     "f43": {"func":vectorize_function(f_fert_input_ref),
                                          "indices":[(ammonium_nitrate.id, wheat_fr_ref.id),
                                                     (ammonium_sulfate.id, wheat_fr_ref.id),
                                                     (urea.id, wheat_fr_ref.id),
                                                     (fert_broadcaster.id, wheat_fr_ref.id),
                                                     (ino_P205_fr.id, wheat_fr_ref.id),
                                                     (org_P205.id, wheat_fr_ref.id),
                                                     (packaging_fert_glo.id, wheat_fr_ref.id),
                                                     (carbondioxide_fossil_urea.id, wheat_fr_ref.id),
                                                  
                                                  
                                                     (fert_broadcaster_ch.id, soy_ch_ref.id),
                                                     (green_manure_ch.id, soy_ch_ref.id),
                                                     (nutrient_supply_thomas_meal_ch.id, soy_ch_ref.id),
                                                     (liquidmanure_spr_ch.id, soy_ch_ref.id),
                                                     (packaging_fert_glo.id, soy_ch_ref.id),
                                                     (phosphate_rock_glo.id, soy_ch_ref.id),
                                                     (potassium_chloride_rer.id, soy_ch_ref.id),
                                                     (potassium_sulfate_rer.id, soy_ch_ref.id),
                                                     (single_superphosphate_rer.id, soy_ch_ref.id),
                                                     (solidmanure_spreading_ch.id, soy_ch_ref.id),
                                                     (triplesuperphosphate.id, soy_ch_ref.id),
    
                                                     (fert_broadcaster_ch.id, alfalfa_ch_ref.id),
                                                     (ino_P205_ch.id, alfalfa_ch_ref.id),
                                                     (liquidmanure_spr_ch.id, alfalfa_ch_ref.id),
                                                     (packaging_fert_glo.id, alfalfa_ch_ref.id),
                                                     (solidmanure_spreading_ch.id, alfalfa_ch_ref.id),
                                                     
                                                     (market_for_ammonium_sulfate_maize.id, maize_ch_ref.id),
                                                     (fertilising_by_broadcaster_maize.id, maize_ch_ref.id),
                                                     (market_for_inorganic_nitrogen_fertiliser_as_n_maize.id, maize_ch_ref.id),
                                                     (market_for_urea_maize.id, maize_ch_ref.id),
                                                     (market_for_packaging_for_fertilisers_maize.id, maize_ch_ref.id),
                                                     (market_for_phosphate_rock_beneficiated_maize.id, maize_ch_ref.id),
                                                     (market_for_inorganic_phosphorus_fertiliser_as_p2o5_maize.id, maize_ch_ref.id),
                                                     (green_manure_growing_swiss_integrated_production_until_april_maize.id, maize_ch_ref.id),
                                                     (liquid_manure_spreading_by_vacuum_tanker_maize.id, maize_ch_ref.id),
                                                     (solid_manure_loading_and_spreading_by_hydraulic_loader_and_spreader_maize.id, maize_ch_ref.id),
                                                     (market_for_ammonium_nitrate_maize.id, maize_ch_ref.id),
                                                     (carbon_dioxide_fossil_maizeairnonurbanairorfromhighstacks.id, maize_ch_ref.id)
                                                     
                                                     
                                                     
                                                     
                                                     
                                                     
                                                     
                                                     
                                                     ]},                                        
                               
                     "f44": {"func":vectorize_function(f_fert_input_avs),
                                          "indices":[
                                                      # Elec main
                                              (ammonium_nitrate.id, wheat_fr_AVS_elec_main.id),
                                                     (ammonium_sulfate.id, wheat_fr_AVS_elec_main.id),
                                                     (urea.id, wheat_fr_AVS_elec_main.id),
                                                     (fert_broadcaster.id, wheat_fr_AVS_elec_main.id),
                                                     (ino_P205_fr.id, wheat_fr_AVS_elec_main.id),
                                                     (org_P205.id, wheat_fr_AVS_elec_main.id),
                                                     (packaging_fert_glo.id, wheat_fr_AVS_elec_main.id),
                                                     (carbondioxide_fossil_urea.id, wheat_fr_AVS_elec_main.id),
                                                  
                                                  
                                                     (fert_broadcaster_ch.id, soy_ch_AVS_elec_main.id),
                                                     (green_manure_ch.id, soy_ch_AVS_elec_main.id),
                                                     (nutrient_supply_thomas_meal_ch.id, soy_ch_AVS_elec_main.id),
                                                     (liquidmanure_spr_ch.id, soy_ch_AVS_elec_main.id),
                                                     (packaging_fert_glo.id, soy_ch_AVS_elec_main.id),
                                                     (phosphate_rock_glo.id, soy_ch_AVS_elec_main.id),
                                                     (potassium_chloride_rer.id, soy_ch_AVS_elec_main.id),
                                                     (potassium_sulfate_rer.id, soy_ch_AVS_elec_main.id),
                                                     (single_superphosphate_rer.id, soy_ch_AVS_elec_main.id),
                                                     (solidmanure_spreading_ch.id, soy_ch_AVS_elec_main.id),
                                                     (triplesuperphosphate.id, soy_ch_AVS_elec_main.id),
    
                                                     (fert_broadcaster_ch.id, alfalfa_ch_AVS_elec_main.id),
                                                     (ino_P205_ch.id, alfalfa_ch_AVS_elec_main.id),
                                                     (liquidmanure_spr_ch.id, alfalfa_ch_AVS_elec_main.id),
                                                     (packaging_fert_glo.id, alfalfa_ch_AVS_elec_main.id),
                                                     (solidmanure_spreading_ch.id, alfalfa_ch_AVS_elec_main.id),
                                                     
                                                     
                                                     
                                                     (market_for_ammonium_sulfate_maize.id, maize_ch_AVS_elec_main.id),
                                                     (fertilising_by_broadcaster_maize.id, maize_ch_AVS_elec_main.id),
                                                     (market_for_inorganic_nitrogen_fertiliser_as_n_maize.id, maize_ch_AVS_elec_main.id),
                                                     (market_for_urea_maize.id, maize_ch_AVS_elec_main.id),
                                                     (market_for_packaging_for_fertilisers_maize.id, maize_ch_AVS_elec_main.id),
                                                     (market_for_phosphate_rock_beneficiated_maize.id, maize_ch_AVS_elec_main.id),
                                                     (market_for_inorganic_phosphorus_fertiliser_as_p2o5_maize.id, maize_ch_AVS_elec_main.id),
                                                     (green_manure_growing_swiss_integrated_production_until_april_maize.id, maize_ch_AVS_elec_main.id),
                                                     (liquid_manure_spreading_by_vacuum_tanker_maize.id, maize_ch_AVS_elec_main.id),
                                                     (solid_manure_loading_and_spreading_by_hydraulic_loader_and_spreader_maize.id, maize_ch_AVS_elec_main.id),
                                                     (market_for_ammonium_nitrate_maize.id, maize_ch_AVS_elec_main.id),
                                                     (carbon_dioxide_fossil_maizeairnonurbanairorfromhighstacks.id, maize_ch_AVS_elec_main.id),
                                                     
                                                     
                                                     # Crop main
                                                     
                                                     (ammonium_nitrate.id, wheat_fr_AVS_crop_main.id),
                                                                (ammonium_sulfate.id, wheat_fr_AVS_crop_main.id),
                                                                (urea.id, wheat_fr_AVS_crop_main.id),
                                                                (fert_broadcaster.id, wheat_fr_AVS_crop_main.id),
                                                                (ino_P205_fr.id, wheat_fr_AVS_crop_main.id),
                                                                (org_P205.id, wheat_fr_AVS_crop_main.id),
                                                                (packaging_fert_glo.id, wheat_fr_AVS_crop_main.id),
                                                                (carbondioxide_fossil_urea.id, wheat_fr_AVS_crop_main.id),
                                                             
                                                             
                                                                (fert_broadcaster_ch.id, soy_ch_AVS_crop_main.id),
                                                                (green_manure_ch.id, soy_ch_AVS_crop_main.id),
                                                                (nutrient_supply_thomas_meal_ch.id, soy_ch_AVS_crop_main.id),
                                                                (liquidmanure_spr_ch.id, soy_ch_AVS_crop_main.id),
                                                                (packaging_fert_glo.id, soy_ch_AVS_crop_main.id),
                                                                (phosphate_rock_glo.id, soy_ch_AVS_crop_main.id),
                                                                (potassium_chloride_rer.id, soy_ch_AVS_crop_main.id),
                                                                (potassium_sulfate_rer.id, soy_ch_AVS_crop_main.id),
                                                                (single_superphosphate_rer.id, soy_ch_AVS_crop_main.id),
                                                                (solidmanure_spreading_ch.id, soy_ch_AVS_crop_main.id),
                                                                (triplesuperphosphate.id, soy_ch_AVS_crop_main.id),
    
                                                                (fert_broadcaster_ch.id, alfalfa_ch_AVS_crop_main.id),
                                                                (ino_P205_ch.id, alfalfa_ch_AVS_crop_main.id),
                                                                (liquidmanure_spr_ch.id, alfalfa_ch_AVS_crop_main.id),
                                                                (packaging_fert_glo.id, alfalfa_ch_AVS_crop_main.id),
                                                                (solidmanure_spreading_ch.id, alfalfa_ch_AVS_crop_main.id),
                                                                
                                                                
                                                                (market_for_ammonium_sulfate_maize.id, maize_ch_AVS_crop_main.id),
                                                                (fertilising_by_broadcaster_maize.id, maize_ch_AVS_crop_main.id),
                                                                (market_for_inorganic_nitrogen_fertiliser_as_n_maize.id, maize_ch_AVS_crop_main.id),
                                                                (market_for_urea_maize.id, maize_ch_AVS_crop_main.id),
                                                                (market_for_packaging_for_fertilisers_maize.id, maize_ch_AVS_crop_main.id),
                                                                (market_for_phosphate_rock_beneficiated_maize.id, maize_ch_AVS_crop_main.id),
                                                                (market_for_inorganic_phosphorus_fertiliser_as_p2o5_maize.id, maize_ch_AVS_crop_main.id),
                                                                (green_manure_growing_swiss_integrated_production_until_april_maize.id, maize_ch_AVS_crop_main.id),
                                                                (liquid_manure_spreading_by_vacuum_tanker_maize.id, maize_ch_AVS_crop_main.id),
                                                                (solid_manure_loading_and_spreading_by_hydraulic_loader_and_spreader_maize.id, maize_ch_AVS_crop_main.id),
                                                                (market_for_ammonium_nitrate_maize.id, maize_ch_AVS_crop_main.id),
                                                                (carbon_dioxide_fossil_maizeairnonurbanairorfromhighstacks.id, maize_ch_AVS_crop_main.id)

                                                                
                                                                
                                                                
                                                                
                                                                
                                                                
                                                                
                                                                
                                                                ]},   # Because the output of this virtual activity is always 1 virtual unit. Here we modify the "actual" output which is the crop substitutiog the "AVS" activity                                          
                               
                      "f45": {"func":vectorize_function(f_nutri_emission_ref),
                                            "indices":[(ammonia.id, wheat_fr_ref.id),   #wheat_fr_ref.id
                                                      (dinitrogen_monoxide.id, wheat_fr_ref.id),
                                                      (nitrogen_oxide.id, wheat_fr_ref.id),
                                                      (nitrate.id, wheat_fr_ref.id),
                                                      (phosphate_groundwater.id, wheat_fr_ref.id),
                                                      (phosphate_surfacewater.id, wheat_fr_ref.id),
                                                  
                                                      (ammonia.id, soy_ch_ref.id),
                                                        (dinitrogen_monoxide.id, soy_ch_ref.id),
                                                        (nitrogen_oxide.id, soy_ch_ref.id),
                                                        (nitrate.id, soy_ch_ref.id),
                                                        (phosphate_groundwater.id, soy_ch_ref.id),
                                                        (phosphate_surfacewater.id, soy_ch_ref.id),
                                                  
                                                      (ammonia.id, alfalfa_ch_ref.id),
                                                        (dinitrogen_monoxide.id, alfalfa_ch_ref.id),
                                                        (nitrogen_oxide.id, alfalfa_ch_ref.id),
                                                        (nitrate.id, alfalfa_ch_ref.id),
                                                        (phosphate_groundwater.id, alfalfa_ch_ref.id),
                                                        (phosphate_surfacewater.id, alfalfa_ch_ref.id),
                                                        
                                                      (nitrate_maizewaterground.id, maize_ch_ref.id),
                                                        (nitrogen_oxides_maizeairnonurbanairorfromhighstacks.id, maize_ch_ref.id),
                                                        (phosphate_maizewatersurfacewater.id, maize_ch_ref.id),
                                                        (dinitrogen_monoxide_maizeairnonurbanairorfromhighstacks.id, maize_ch_ref.id),
                                                        (ammonia_maizeairnonurbanairorfromhighstacks.id, maize_ch_ref.id),
                                                        (phosphate_maizewaterground.id, maize_ch_ref.id),
                                                        (phosphorus_maizewatersurfacewater.id, maize_ch_ref.id)
              
                                                        
                                                        
                                                        
                                                          ]},
                  
                      "f46": {"func":vectorize_function(f_nutri_emission_avs),
                                            "indices":[
                                                    # Elec main
                                                    (ammonia.id, wheat_fr_AVS_elec_main.id),
                                                      (dinitrogen_monoxide.id, wheat_fr_AVS_elec_main.id),
                                                      (nitrogen_oxide.id, wheat_fr_AVS_elec_main.id),
                                                      (nitrate.id, wheat_fr_AVS_elec_main.id),
                                                      (phosphate_groundwater.id, wheat_fr_AVS_elec_main.id),
                                                      (phosphate_surfacewater.id, wheat_fr_AVS_elec_main.id),
                                                  
                                                      (ammonia.id, soy_ch_AVS_elec_main.id),
                                                        (dinitrogen_monoxide.id, soy_ch_AVS_elec_main.id),
                                                        (nitrogen_oxide.id, soy_ch_AVS_elec_main.id),
                                                        (nitrate.id, soy_ch_AVS_elec_main.id),
                                                        (phosphate_groundwater.id, soy_ch_AVS_elec_main.id),
                                                        (phosphate_surfacewater.id, soy_ch_AVS_elec_main.id),
                                                  
                                                      (ammonia.id, alfalfa_ch_AVS_elec_main.id),
                                                        (dinitrogen_monoxide.id, alfalfa_ch_AVS_elec_main.id),
                                                        (nitrogen_oxide.id, alfalfa_ch_AVS_elec_main.id),
                                                        (nitrate.id, alfalfa_ch_AVS_elec_main.id),
                                                        (phosphate_groundwater.id, alfalfa_ch_AVS_elec_main.id),
                                                        (phosphate_surfacewater.id, alfalfa_ch_AVS_elec_main.id),
                                                        
                                                        
                                                        
                                                      (nitrate_maizewaterground.id, maize_ch_AVS_elec_main.id),
                                                        (nitrogen_oxides_maizeairnonurbanairorfromhighstacks.id, maize_ch_AVS_elec_main.id),
                                                        (phosphate_maizewatersurfacewater.id, maize_ch_AVS_elec_main.id),
                                                        (dinitrogen_monoxide_maizeairnonurbanairorfromhighstacks.id, maize_ch_AVS_elec_main.id),
                                                        (ammonia_maizeairnonurbanairorfromhighstacks.id, maize_ch_AVS_elec_main.id),
                                                        (phosphate_maizewaterground.id, maize_ch_AVS_elec_main.id),
                                                        (phosphorus_maizewatersurfacewater.id, maize_ch_AVS_elec_main.id),
              
                                  
                                                        
                                                        # Crop main
                                                    (ammonia.id, wheat_fr_AVS_crop_main.id),
                                                      (dinitrogen_monoxide.id, wheat_fr_AVS_crop_main.id),
                                                      (nitrogen_oxide.id, wheat_fr_AVS_crop_main.id),
                                                      (nitrate.id, wheat_fr_AVS_crop_main.id),
                                                      (phosphate_groundwater.id, wheat_fr_AVS_crop_main.id),
                                                      (phosphate_surfacewater.id, wheat_fr_AVS_crop_main.id),
                                                  
                                                      (ammonia.id, soy_ch_AVS_crop_main.id),
                                                        (dinitrogen_monoxide.id, soy_ch_AVS_crop_main.id),
                                                        (nitrogen_oxide.id, soy_ch_AVS_crop_main.id),
                                                        (nitrate.id, soy_ch_AVS_crop_main.id),
                                                        (phosphate_groundwater.id, soy_ch_AVS_crop_main.id),
                                                        (phosphate_surfacewater.id, soy_ch_AVS_crop_main.id),
                                                  
                                                      (ammonia.id, alfalfa_ch_AVS_crop_main.id),
                                                        (dinitrogen_monoxide.id, alfalfa_ch_AVS_crop_main.id),
                                                        (nitrogen_oxide.id, alfalfa_ch_AVS_crop_main.id),
                                                        (nitrate.id, alfalfa_ch_AVS_crop_main.id),
                                                        (phosphate_groundwater.id, alfalfa_ch_AVS_crop_main.id),
                                                        (phosphate_surfacewater.id, alfalfa_ch_AVS_crop_main.id),
                                                        
                                                        
                                                        
                                                      (nitrate_maizewaterground.id, maize_ch_AVS_crop_main.id),
                                                        (nitrogen_oxides_maizeairnonurbanairorfromhighstacks.id, maize_ch_AVS_crop_main.id),
                                                        (phosphate_maizewatersurfacewater.id, maize_ch_AVS_crop_main.id),
                                                        (dinitrogen_monoxide_maizeairnonurbanairorfromhighstacks.id, maize_ch_AVS_crop_main.id),
                                                        (ammonia_maizeairnonurbanairorfromhighstacks.id, maize_ch_AVS_crop_main.id),
                                                        (phosphate_maizewaterground.id, maize_ch_AVS_crop_main.id),
                                                        (phosphorus_maizewatersurfacewater.id, maize_ch_AVS_crop_main.id)
              
      
     
                                                        
                                                        ]},
                  
                  
                      "f47": {"func":vectorize_function(f_machinery_ref),
                                           "indices":[(tillage_rotary_harrow_glo.id, wheat_fr_ref.id),
                                                      (tillage_rotary_spring_tine_glo.id, wheat_fr_ref.id),
                                                      (sowing_glo.id, wheat_fr_ref.id),
                                                      (tillage_ploughing_glo.id, wheat_fr_ref.id),
                                                  
                                                      (tillage_currying_weeder_ch.id, soy_ch_ref.id),
                                                        (tillage_rotary_spring_tine_ch.id, soy_ch_ref.id),
                                                        (sowing_ch.id, soy_ch_ref.id),
                                                  
                                                      (fodder_loading_ch.id, alfalfa_ch_ref.id),
                                                        (rotary_mower_ch.id, alfalfa_ch_ref.id),
                                                        (sowing_ch.id, alfalfa_ch_ref.id),
                                                        (tillage_rotary_spring_tine_ch.id, alfalfa_ch_ref.id),
                                                        (tillage_ploughing_ch.id, alfalfa_ch_ref.id),
                                                        (tillage_rolling_ch.id, alfalfa_ch_ref.id),
                                                        
                                                        
                                                        (hoeing_maize.id, maize_ch_ref.id),
                                                          (tillage_harrowing_by_spring_tine_harrow_maize.id, maize_ch_ref.id),
                                                          (sowing_maize.id, maize_ch_ref.id),
                                                          (market_for_transport_tractor_and_trailer_agricultural_maize.id, maize_ch_ref.id),
                                                          (tillage_ploughing_maize.id, maize_ch_ref.id),
                                                          
                                                        
                                                        
                                                        ]},
                  
                      "f48": {"func":vectorize_function(f_machinery_avs),
                                           "indices":[
                                               # elec main
                                               (tillage_rotary_harrow_glo.id, wheat_fr_AVS_elec_main.id),
                                                      (tillage_rotary_spring_tine_glo.id, wheat_fr_AVS_elec_main.id),
                                                      (sowing_glo.id, wheat_fr_AVS_elec_main.id),
                                                      (tillage_ploughing_glo.id, wheat_fr_AVS_elec_main.id),
                                                  
                                                      (tillage_currying_weeder_ch.id, soy_ch_AVS_elec_main.id),
                                                        (tillage_rotary_spring_tine_ch.id, soy_ch_AVS_elec_main.id),
                                                        (sowing_ch.id, soy_ch_AVS_elec_main.id),
                                                  
                                                      (fodder_loading_ch.id, alfalfa_ch_AVS_elec_main.id),
                                                        (rotary_mower_ch.id, alfalfa_ch_AVS_elec_main.id),
                                                        (sowing_ch.id, alfalfa_ch_AVS_elec_main.id),
                                                        (tillage_rotary_spring_tine_ch.id, alfalfa_ch_AVS_elec_main.id),
                                                        (tillage_ploughing_ch.id, alfalfa_ch_AVS_elec_main.id),
                                                        (tillage_rolling_ch.id, alfalfa_ch_AVS_elec_main.id),
                                                        
                                                        
                                                        (hoeing_maize.id, maize_ch_AVS_elec_main.id),
                                                          (tillage_harrowing_by_spring_tine_harrow_maize.id, maize_ch_AVS_elec_main.id),
                                                          (sowing_maize.id, maize_ch_AVS_elec_main.id),
                                                          (market_for_transport_tractor_and_trailer_agricultural_maize.id, maize_ch_AVS_elec_main.id),
                                                          (tillage_ploughing_maize.id, maize_ch_AVS_elec_main.id),
                                                          
                                                        
                                                # crop main
                                                
                                                
                                               (tillage_rotary_harrow_glo.id, wheat_fr_AVS_crop_main.id),
                                                      (tillage_rotary_spring_tine_glo.id, wheat_fr_AVS_crop_main.id),
                                                      (sowing_glo.id, wheat_fr_AVS_crop_main.id),
                                                      (tillage_ploughing_glo.id, wheat_fr_AVS_crop_main.id),
                                                  
                                                      (tillage_currying_weeder_ch.id, soy_ch_AVS_crop_main.id),
                                                        (tillage_rotary_spring_tine_ch.id, soy_ch_AVS_crop_main.id),
                                                        (sowing_ch.id, soy_ch_AVS_crop_main.id),
                                                  
                                                      (fodder_loading_ch.id, alfalfa_ch_AVS_crop_main.id),
                                                        (rotary_mower_ch.id, alfalfa_ch_AVS_crop_main.id),
                                                        (sowing_ch.id, alfalfa_ch_AVS_crop_main.id),
                                                        (tillage_rotary_spring_tine_ch.id, alfalfa_ch_AVS_crop_main.id),
                                                        (tillage_ploughing_ch.id, alfalfa_ch_AVS_crop_main.id),
                                                        (tillage_rolling_ch.id, alfalfa_ch_AVS_crop_main.id),
                                                        
                                                        
                                                        
                                                        
                                                        (hoeing_maize.id, maize_ch_AVS_crop_main.id),
                                                          (tillage_harrowing_by_spring_tine_harrow_maize.id, maize_ch_AVS_crop_main.id),
                                                          (sowing_maize.id, maize_ch_AVS_crop_main.id),
                                                          (market_for_transport_tractor_and_trailer_agricultural_maize.id, maize_ch_AVS_crop_main.id),
                                                          (tillage_ploughing_maize.id, maize_ch_AVS_crop_main.id),
                                                          
    
                                                        ]},
                  
                     "f49": {"func":vectorize_function(f_water_ref),
                                          "indices":[(water_air.id, wheat_fr_ref.id),
                                                     (water_ground.id, wheat_fr_ref.id),
                                                     (water_surface.id, wheat_fr_ref.id),
                                                     (water_market_irrigation.id, wheat_fr_ref.id),
                                                     
                                                     
                                                     
                                                     
                                                     (water_maizewatersurfacewater.id, maize_ch_ref.id),
                                                     (water_maizewaterground.id, maize_ch_ref.id),
                                                     (market_for_irrigation_maize.id, maize_ch_ref.id)

                                                     
                                                     ]},
                  
                  
                     "f50": {"func":vectorize_function(f_water_avs),
                                          "indices":[(water_air.id, wheat_fr_AVS_elec_main.id),
                                                     (water_ground.id, wheat_fr_AVS_elec_main.id),
                                                     (water_surface.id, wheat_fr_AVS_elec_main.id),
                                                     (water_market_irrigation.id, wheat_fr_AVS_elec_main.id),

                                                     
                                                     
                                                     (water_air.id, wheat_fr_AVS_crop_main.id),
                                                    (water_ground.id, wheat_fr_AVS_crop_main.id),
                                                    (water_surface.id, wheat_fr_AVS_crop_main.id),
                                                     (water_market_irrigation.id, wheat_fr_AVS_crop_main.id),

                                                     (water_maizewatersurfacewater.id, maize_ch_AVS_crop_main.id),
                                                     (water_maizewaterground.id, maize_ch_AVS_crop_main.id),
                                                     (market_for_irrigation_maize.id, maize_ch_AVS_crop_main.id),
                                                    
                                                     (water_maizewatersurfacewater.id, maize_ch_AVS_elec_main.id),
                                                     (water_maizewaterground.id, maize_ch_AVS_elec_main.id),
                                                     (market_for_irrigation_maize.id, maize_ch_AVS_elec_main.id),                                                    
                                                    
                                                    ]},
                  
                    "f51": {"func":vectorize_function(f_carbon_soil_accumulation_ref),
                                         "indices":[(c_soil_accu.id, wheat_fr_ref.id),
                                                    (c_soil_accu.id, soy_ch_ref.id),
                                                    (c_soil_accu.id, alfalfa_ch_ref.id),
                                                    (c_soil_accu.id, maize_ch_ref.id)
                                                    ]},
                  
                    "f52": {"func":vectorize_function(f_carbon_soil_accumulation_avs),
                                         "indices":[(c_soil_accu.id, AVS_elec_main.id),
                                                    (c_soil_accu.id, AVS_elec_main_single.id),
                                                    (c_soil_accu.id, AVS_elec_main_multi.id),
                                                    (c_soil_accu.id, AVS_elec_main_cis.id),
                                                    (c_soil_accu.id, AVS_elec_main_cdte.id),
                                                    
                                                    (c_soil_accu.id, AVS_crop_main.id),
                                                    (c_soil_accu.id, AVS_crop_main_single.id),
                                                    (c_soil_accu.id, AVS_crop_main_multi.id),
                                                    (c_soil_accu.id, AVS_crop_main_cis.id),
                                                    (c_soil_accu.id, AVS_crop_main_cdte.id)]},
       
                    "f53": {"func":vectorize_function(f_carbon_soil_accumulation_pv),
                                         "indices":[(c_soil_accu.id, PV_ref.id),
                                                    (c_soil_accu.id, PV_ref_single.id),
                                                    (c_soil_accu.id, PV_ref_multi.id),
                                                    (c_soil_accu.id, PV_ref_cis.id),
                                                    (c_soil_accu.id, PV_ref_cdte.id)]},
                    
                    "f54": {"func":vectorize_function(f_outputAVS_crop_main),
                                         "indices":[(AVS_crop_main.id, AVS_crop_main.id),
                                                    (AVS_crop_main_single.id, AVS_crop_main_single.id),
                                                    (AVS_crop_main_multi.id, AVS_crop_main_multi.id),
                                                    (AVS_crop_main_cis.id, AVS_crop_main_cis.id),
                                                    (AVS_crop_main_cdte.id, AVS_crop_main_cdte.id)]}
                    
                    
                    
    
                  }
    
    local_vars_param_functions ={"dict_funct": dict_funct,
                                 "dictionnaries":dictionnaries}
    
    global_vars.update(local_vars)

    
    global_vars.update(local_vars_param_functions)
    
    global_vars.update({"modif_impact_marginal_elec":modif_impact_marginal_elec})

    
    

    # We delete the hydro elec for rer # cf Besseau
    indices_array_fix = np.array([(electricity_hydro_row_si_sg.id, si_sg_rer.id),
                                  (electricity_hydro_row_si_sg.id, si_sg_row.id)],  # (electricity_hydro.id,siliconproductionsolar_grade_rer.id)
                                 
                                 dtype=bwp.INDICES_DTYPE)
    
    data_array_fix = np.array(
        [0,0])
    
    
    
    

    global_vars.update({"indices_array_fix":indices_array_fix})
    global_vars.update({"data_array_fix":data_array_fix})



def global_vars():
    # Return the global variables dictionary
    return globals()

def set_activity_exchanges_as_variables(activity,background_db,Biosphere,suffix, includeloc,includecat):
    exchanges = list(activity.exchanges())
    
    for exc in exchanges:
        if exc["type"]!="production":
            # Create a unique variable name based on the exchange name
            variable_name = exc.input['name'].lower().replace(" ", "_").replace(",", "").replace("-", "_")+"_"+suffix
            
            # Retrieve the corresponding activity from the background database
            activity_code = exc.input['code']
            
            activity_type = exc['type']
            
            #print(activity_type)
            
            if activity_type=="technosphere":
                act = background_db.get(activity_code)
                location = act["location"]
                
                if includeloc:
                    variable_name= variable_name+location
                    variable_name = variable_name.replace(" ", "")
                    variable_name = variable_name.replace("-", "")

                global_vars()[variable_name] = act

            elif activity_type=="biosphere":
                
                act_bio = Biosphere.get(activity_code)
                
                
                act_categ = "".join([i for i in act_bio["categories"]])
                
                
                if includecat:
                    variable_name= variable_name+act_categ
                    variable_name = variable_name.replace(" ", "")
                    variable_name = variable_name.replace("-", "")


                
                global_vars()[variable_name] = act_bio
                
            
                



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


####
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

