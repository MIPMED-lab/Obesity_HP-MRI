# Obesity_HP-MRI
Code generated to process mice HP-MRI experimental data and perform model fittings 

- DataProcGEN_Magnitude: Jupyter notebook to process NMR spectroscopic data from a Bruker 3T Maxwell Biospec machine. Python. 
- DataProcGEN_Magnitude_Batch: Same as DataProcGEN_Magnitude, but to process multiple experiments at the same time. Python.
- FunctionsProcess: All processing functions needed for DataProcGEN_Magnitude and DataProcGEN_Magnitude_Batch. Python.
- InitalParameterFitMice: Jupyter notebook to extract experimental data from DataProcGEN_Magnitude and fit the parameters for a mathematical model describing the system. Julia. 
- InitalParameterFitMice_Batch: Same as InitalParameterFitMice, but to fit multiple experiments at the same time. Julia. 
- ModelFunctionsAll: All processing and fitting functions needed by InitalParameterFitMice and InitalParameterFitMice_Batch. Julia. 