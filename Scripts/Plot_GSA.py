# -*- coding: utf-8 -*-
"""
Created on Fri Apr 18 10:15:36 2025

@author: pjouannais

Processes and plots GSA results

"""



import numpy as np
from SALib.sample import saltelli
from SALib.analyze import sobol
from util_functions import *
import matplotlib.pyplot as plt

from SALib.analyze import delta
        
        
        
        
import matplotlib.pyplot as plt
import seaborn as sns

        
import pandas as pd
import os


import math



res2 = importpickle("../Results/Server/Sens/Resmedi_4_4_364298size=sensii_uniquelifetime.pkl")




res_uniquelifetime = res2["ei_consequential_3.10_modelremind_pathSSP1-Base_year2070_argshort_slope"]
results_uniquelifetime= res_uniquelifetime[1]  # For all impact categories
inputs_uniquelifetime=res_uniquelifetime[0]

# remove fixed parameters

# Drop columns where there is only one unique value

inputs_uniquelifetime_clean = inputs_uniquelifetime.loc[:, inputs_uniquelifetime.nunique() > 1]

""" Import indexes to plot"""

sens_delta_uniquelifetime = importpickle("../Results/Sens/Sensi_4_10_167828uniquelifetime.pkl")


def plot_delta_indices(results_dict, problem, suffix, top_x=10):
    """
    Plots the top X Delta indices with confidence intervals from multiple Si results.
    
    Parameters:
    - results_dict: dict with keys as result names and values as Si objects.
    - problem: dict containing parameter names.
    - top_x: int, number of top parameters to plot (based on Delta index).
    """
    for key_meth, meth in results_dict.items():
        for key_FU, Si in meth.items():
    
            delta_values = Si['delta']
            delta_conf = Si['delta_conf']
            parameter_names = problem['names']
    
            # Sort parameters by Delta index
            sorted_indices = np.argsort(delta_values)[::-1]  # Descending order
            sorted_delta = np.array(delta_values)[sorted_indices]
            sorted_conf = np.array(delta_conf)[sorted_indices]
            sorted_names = np.array(parameter_names)[sorted_indices]
    
            # Select top_x parameters
            top_delta = sorted_delta[:top_x]
            top_conf = sorted_conf[:top_x]
            top_names = sorted_names[:top_x]
    
            # Plot
            plt.figure(figsize=(10, 6))
            plt.barh(top_names, top_delta, xerr=top_conf, color='skyblue', capsize=4)
            plt.xlabel("Delta Index (Borgonovo)")
            plt.ylabel("Parameters")
            plt.title(f"GSA - Top {top_x} Parameters by Delta ({key_meth},{key_FU},{suffix})")
            plt.gca().invert_yaxis()  # Most important at the top
            plt.grid(axis='x', linestyle='--', alpha=0.7)

            # Fix layout to prevent cutoff
            plt.tight_layout()

            # Save plot
            os.makedirs(os.path.join("../Results/Sensiplots"), exist_ok=True)
            save_path = os.path.join("../Results/Sensiplots", f"{key_meth}_{key_FU}_{suffix}.jpg")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()


"""Unique lifetime"""


# Create a minimal problem structure
problem_uniquelifetime = {
    'num_vars': inputs_uniquelifetime_clean.shape[1],  # Number of variables
    'names': list(inputs_uniquelifetime_clean.columns)  # param names
}


plot_delta_indices(sens_delta_uniquelifetime, problem_uniquelifetime,"unlif")



        

def gsa_results_to_dataframe(results_dict, problem, suffix):
    """
    Converts a nested GSA results dictionary to a tidy DataFrame.

    Parameters:
    - results_dict: dict with keys as method names and values as dicts of FU -> Si results.
    - problem: dict containing parameter names.
    - suffix: str, label for the scenario type (e.g. "tot" or "unlif").

    Returns:
    - DataFrame with columns: Method, FunctionalUnit, Parameter, Delta, Delta_conf, Scenario
    """
    rows = []
    for key_meth, meth in results_dict.items():
        for key_FU, Si in meth.items():
            delta_values = Si['delta']
            delta_conf = Si['delta_conf']
            parameter_names = problem['names']

            for name, val, conf in zip(parameter_names, delta_values, delta_conf):
                rows.append({
                    "Method": key_meth,
                    "FunctionalUnit": key_FU,
                    "Parameter": name,
                    "Delta": val,
                    "Delta_conf": conf,
                    "Scenario": suffix
                })
    return pd.DataFrame(rows)

# Build DataFrames for both cases
df_unlif = gsa_results_to_dataframe(sens_delta_uniquelifetime, problem_uniquelifetime, "unlif")

# Merge and export
df_all = pd.concat([ df_unlif], ignore_index=True)

# Ensure results folder exists
os.makedirs("../Results", exist_ok=True)
csv_path = "../Results/GSA_results_all.csv"
df_all.to_csv(csv_path, index=False)

print(f"✅ GSA results saved to: {csv_path}")

        
        
        


def plot_delta_panel(results_dict, problem, suffix, top_x=10, figsize=(16, 12), save_path=None):
    """
    Plot combined panel of top Delta indices from multiple Si results.

    Parameters:
    - results_dict: dict of {method: {FU: Si_result}}
    - problem: dict with parameter names
    - suffix: string to include in title/filename
    - top_x: how many top parameters per subplot
    - figsize: overall figure size
    - save_path: if given, path to save the figure (jpg/png)
    """
    # Count total subplots
    total_plots = sum(len(fu_dict) for fu_dict in results_dict.values())
    ncols = 3  # or customize
    nrows = math.ceil(total_plots / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, constrained_layout=True)
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = axes.reshape(1, -1)
    elif ncols == 1:
        axes = axes.reshape(-1, 1)

    axes = axes.flatten()

    plot_idx = 0
    for method, fu_dict in results_dict.items():
        for fu, Si in fu_dict.items():
            if plot_idx >= len(axes):
                break

            ax = axes[plot_idx]
            delta_values = Si['delta']
            delta_conf = Si['delta_conf']
            parameter_names = problem['names']

            # Sort parameters by Delta index descending
            sorted_indices = np.argsort(delta_values)[::-1]
            top_indices = sorted_indices[:top_x]

            top_delta = np.array(delta_values)[top_indices]
            top_conf = np.array(delta_conf)[top_indices]
            top_names = np.array(parameter_names)[top_indices]

            ax.barh(top_names, top_delta, xerr=top_conf, color='skyblue', capsize=3)
            ax.set_xlabel("Delta Index (Borgonovo)")
            ax.set_title(f"{fu}", fontsize=9)
            ax.invert_yaxis()
            ax.grid(axis='x', linestyle='--', alpha=0.7)

            plot_idx += 1

    # Remove unused axes if any
    for i in range(plot_idx, len(axes)):
        fig.delaxes(axes[i])

    fig.suptitle(f"GSA - Top {top_x} Parameters by Delta Index ({suffix})", fontsize=14)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=600, bbox_inches='tight')
        plt.close(fig)
        print(f"✅ Panel plot saved to {save_path}")
    else:
        plt.show()





filtered_results = {"global warming potential (GWP100) ILUC": sens_delta_uniquelifetime["global warming potential (GWP100) ILUC"]}  # Keep only one method

plot_delta_panel(
    results_dict=filtered_results,
    problem=problem,
    suffix="tot_singlemethod",
    top_x=10,
    save_path="../Results/Sensiplots/delta_panel_singlemethod_uniquelifetime.jpg"
)
