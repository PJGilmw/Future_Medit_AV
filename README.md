# Code documentation for the Future_Med_AV repository

## Overview of the repository

This repository contains the code used to reproduce the results of the article: *Jouannais.P, Rapella.L, Drobinski.P, Viovy.N, Douziech.M, Marchand.M, Future Mediterranean Agrivoltaics: Prospective Life Cycle Assessment Under Climatic and Socio-Economic Scenarios* 



**Cite this repository:**



[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.16901801.svg)](https://doi.org/10.5281/zenodo.16901801)


### Overview of folders and files:


**Environment**

+ **environment.yml** File needed to create the virtual environment on WINDOWS.

**orchidee_data**

Contains ORCHIDEE-AV outputs.

**Calib yields France**

Contains french agricultural records (agreste data) for calibration.

**Output**

Output folders where results are exported to.

**Scripts** 

+ twenty  **.py** files: python scripts including the ones to execute and the ones containing functions being called. 

+ one R plotting file for distributions plots.

Files, scripts, and their functions'interconnections are mapped below.  
<br>  

<img src="Code_map.jpg"
     alt="Markdown Monster icon"
     style="float: left; margin-right: 10px;" />  
<br>  


**activities_parameters_and_functions**

Collects background activities, sets input parameters and parameterizes the LCA databases. Called for every new database.


**activities_parameters_and_functions_sensi**

Specific version called for the GSA.

**Main_functions**

Main functions to compute the LCAs.


**functions_orchidee_pipeline**

Pipeline functions from ORCHIDEE to the LCA model + random sampling functions


**functions_foreground**

Functions that create the foreground databases corresponding to specific backgrounds. 
Also collect PV parameters embodied in the premise databases (pv efficiencies etc.)


**functions_plot**

Functions which process, plot and export result statistics.


**exploration_functions_bw**

Accessory exploration function.

**Parameters_and_functions**

Definition of the model's parameters and functions.



**Setup_bw_project**

Sets up the intial brightway project.

**Create premise_databases**

Allows creating the premise databases from the ecoinvent 3.10 consequential database and its biosphere and stores them in different bw projects.

**Compute**

Script which calculates the LCAs within a project : for one SSP-IAM, all grid points, all years, all uncertainty. 
Exports the results.

**Compute_for_sensi**

Calculates the LCAs for the GSA.
Exports the results.


**Compute_for_sensi**

Computes LCAs by keeping ecoinvent 3.10 as background database across the years.



**Complement_fixedtilt_PV_outputs**

This script follows the same structure as "Compute"
It is only used to collect the electric production for fixed AV panels from ORCHIDEE-AV in the same format as the impact result files.
This will ease the computation of AV fixed-tilt impacts. 
Exports the results.

**Complement_wheat_markets**

Script which calculates the impact of the necessary global wheat market mixes across all scenarios and years.
Exports the results.
Will be necessary to  process results at the end.

**Complement_elecmarkets**

Script which calculates the impact of the necessary impacts of electric mixes across all scenarios and years.
Exports the results.
Will be necessary to process results at the end.

**Plot_figures**

Processes the results and exports figures for the article and the SI.


**Plot_distributions.R**

R file generating distributions plot.


**GSA** 

Computes GSA indexes from the model's outputs.


**Plot_GSA** 

Processes and plots GSA results.


**Calib_orchidee_yield** 

Computes the list of calibration factors from ORCHIDEE (ERA LAND 5 reanalysis) to actual grain yields obtained in France for 2010-2020.


<br>

### Reproducing results of the article

  &#128680;&#128680;&#128680;**WARNING 
  
The result ensemble was produced using parallel computing on substantial remote computing capacities (large memory/multiple CPUS). 
Several days of computations are necessary to reproduce the results, by using multiple instances with multiple cores. 
The computing functions are written to be called in parallel with the package "ray".

The ORCHIDEE-AV outputs are provided as such and were produced with access to local ORCHIDEE instances.




*Requirements*

+ Miniconda or Anaconda
https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html

+ The ecoinvent license (we used ecoinvent 3.10 consequential)

+ A python interface (e.g., Spyder)

+ A R interface (e.g, Rstudio)

*Step by step procedure:*

1. **Download or clone a local copy of the repository. Keep the folders structure.**

2. **Prepare a conda environment with all needed packages**

+ From terminal or Anaconda/Miniconda terminal access the "Environment" folder. The path depends on where you saved the repository:

```
 cd <yourpathtothefolder/Environment>
```

+ Create the conda environment with all necessary packages using the .yml file corresponding to your OS.

**For Windows:**


```
conda env create --file env_bw_windows.yml
```

+ Activate the newly created environment:

```
conda activate Future_Med_AV
```

+ Install the ecoinvent_interface package from conda:
```
conda install ecoinvent_interface
```

+ From pip, install missing packages:

```
pip install ray==2.40.0

```

```
pip install premise
```

```
pip install seaborn
```

```
pip install netCDF4
```

**For Ubuntu and MACOS:**



Try to install the environment from the environment file and, in case of issues, complete the environment by installing the problematic packages manually. 


3. **Set up the Brigtway2 projects**

+ Open the file **Setup_bw_project.py** in a text editor or python interface and change the password and username for your ecoinvent license. 

+ From the python interface or from command, execute the whole script to prepare the initial Brightway2 project (```python Setup_bw_project.py```) .

+ From the python interface or from command, execute the script **Create_premise_databases.py** to create the different bw projects. Each project will contain the background databases for a distinct combination of SSP, constrain and marginal mix algorithm.

4. **Compute** 

    4.1. ***Compute main outputs***

+ The script **Compute.py** performs the LCAs for all the databases within a selected bw_project. It must be executed independently (e.g., on different servers) for each bw_project. Each execution will export a different result file.
Each new execution requires changing the bw project and the database names within the script.

+ The script **Compute_310.py** performs the LCAs for all years by keeping the background database as ecoinvent 3.10.

    4.2. ***Compute complementary outputs***

+ Execute **Complement_fixedtilt_PV_outputs**, **Complement_wheat_markets**, **Complement_elecmarkets** to produce complementary output files that will be necessary before the final process.


    4.3. ***Compute outputs for GSA***

+ Execute **Compute_for_sensi** to produce the  output file which will the input to the GSA.

5. ***Process results and plot***

+ Execute **Plot_figures.py** which fill up the output folders with the corresponding plots and excel, csv files.

+ Execute **Plot_distributions.R.** to produce the distributions plot (Figure 5). It requires processed outputs from Plot_figures.py. 


6. ***GSA***

+ Execute **GSA.py** will export GSA indices in pickle files.

+ Execute **Plot_GSA.py** to plot the GSA results.

&#128680;
Note that the names of the projects, databases and output files can be modified but it must be done consistently through the scripts.

<br>

<br>
