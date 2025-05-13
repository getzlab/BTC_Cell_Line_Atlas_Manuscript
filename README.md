# BTC Cell Line Atlas

This repository contains the code used to generate the analyses and figures presented in:  
**"Generation of a biliary tract cancer cell line atlas identifies molecular subtypes and therapeutic targets"**, Vijay, V., and Karisani, N., et al. [Cancer Discovery](https://doi.org/10.1158/2159-8290.CD-24-1383), 2025.


## 📦 Data Availability

The datasets used in this study are available on Figshare:

-  [https://doi.org/10.6084/m9.figshare.28873196.v1](https://doi.org/10.6084/m9.figshare.28873196.v1)

### 📥 Download Instructions

1. Please download the folder **Main_Data/** from Figshare and replace it with the existing Main_Data/ in the `input_data/Main_Data/` directory.
2. Please download the remaining files from Figshare and place them into the `input_data/Additional_Data/DepMap/` directory.
3. Some input files must be downloaded directly from their source; please refer to the readme.txt file in the respective directory.

### Code Structure
The code is organized into separate directories, each corresponding to notebooks of a figure or a particular 
analysis whose plots are in multiple figures in the manuscript. 
Each directory contains the necessary code to generate the plots for that figure or analysis.
which are then saved in a local`output_X/` subdirectory within the same folder.

Set the `working_dir` variable to `your-path/BTC_Cell_Line_Atlas_Manuscript`. The required input files will be loaded either from the`input_data/`, the local`output_X/`,
or the`output_X/` directories of previous figures or analyses.

## 🛠 Requirements

- For information regarding R packages, please refer to the .html files located in the `Figure1/output_fig1/` directory.
- The Python code was developed and tested using Python version 3.10.6 in PyCharm IDE (Professional Edition).
- Required packages listed in `requirements.txt`

To install dependencies:

```bash
pip install -r requirements.txt
```
