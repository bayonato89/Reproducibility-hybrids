# Reproducing prospective test results for RELM and hybrid earthquake forecasting models for California
This repository provides data and code to reproduce prospective test results for six RELM and sixteen multiplicative 
hybrid earthquake forecasting models for California, reported by Bayona et al. (GJI; in review). This experiment
takes about 2 hours on a modern desktop computer if the number of simulations per forecast and per test (expect for the Poisson and NBD N-tests) is set to 1000.

## Code description 
The scripts to execute the experiment can be found in the `code` directory of this repository. This folder contains two 
.py files, namely `download_data.py` and `reproducibility_hybrids.py`, which download forecast files from 
[Zenodo](https://zenodo.org/record/5141567#.Yc2lO1mnxhE), run the computations, and create and store the figures presented in the manuscript. Finally, the
`run_all.sh` file, in the top-level directory, is a shell script that runs the entire experiment by simply typing `bash ./run_all.sh`
in the Terminal.

## Software dependencies
python=3.10.1  
numpy=1.21.5  
pycsep=0.5.0  


## Software dependencies
In order to run this reproducibility package, the user must have a pycsep environment installed and running on her/his machine ('gji-hybrids' in this example). The easiest way to install
pycsep is using `conda`, however; it can also be installed using `pip` or built from source (see the [Documentation on how to install pyCSEP](https://docs.cseptesting.org/getting_started/installing.html)).

```
conda create -n gji-hybrids
conda activate gji-hybrids
conda install --channel conda-forge numpy=1.21.5 pycsep=0.5.0
```

In addition, the user must have access to a Unix shell that has python3 intsalled with the `requests` library (it should be provided with pycsep=0.5.0). If not, she/he can install this library using:

```
conda install requests
```

## Instructions for running 
These instructions assume that the user is "within" the environment, with python3 and the requests library installed. Thus, running the experiment is as simple as:

```
git clone https://github.com/bayonato89/Reproducibility-hybrids.git
cd Reproducibility-hybrids
bash ./run_all.sh
```
