# Bayesian Screening:Multi-test Bayesian Optimization Applied to *in silico* Material Screening
___

This README.md file outlines how to replicate the studies described in the above submission including a description of files used.
This project was conducted with the python programming language and so this project made use of standard python conventions in implementation, doccumentation and repository layout.

## Repository Contents
* `AmiSimTools`: Contains python classes used to load the input data files and conduct the parallel, two stage screenings.
>* `DataTriage.py` - Loads feature and target data from a given csv file and calculates / reshapes data into format required by prospector
>* `SimScreen.py` - Performs the parallel screen as outlined in the submission
* `costs_*_.json`: External files containing the costs of the two different tests conducted for the screens.
>>* `costs_HCOF.json` --> `HCOF.csv`
>>* `costs_HMOF.json` --> `HMOF.csv`
* `HMOF.csv / HCOF/csv`: Data files for the COF and MOF screens conducted. Both files have had their features normalised, with the far right hand columns being the target values.
* `main.py`: The script which is run that performs the screening, making use of the python classes described in `AmiSimTools` and `Prospector_2.py`. This file accepts multiple command line arguments as detailed below.
* `Prospector_2.py`: This file contains the implemented screening class as described in the submission.
* `requirements.txt`: A python requirements file containing the external libraries needed to run the code contained.

## Flags for `main.py`
*  `--data_file`: path to HMOF / HCOF data file
* `--cost`: path to cost json file
* `--budget`: The budget value to be used for the screening
* `--nthreads`: The number of parallel threads / workers to be used for the screening
* `--initial_samples`: The number of initial random samples to be taken to initialise the `prospector`
* `--min_samples`: The number of samples to be assessed before the `prospector` will fit, to ensure sufficient data presence and prevent crashes
* `--acquisition`: The name of the sampling / acquisition function to be used by the `prospector`

## Installation / Set up
Instructions for both conda and venv installations are described.
Both approaches assume the reader has all files saved locally and the cwd is this folder.
The placeholder `<env>` is used and should be replaced by the reader's preferred environment name.
### Venv
```bash
$ python3 -m venv <env>
$ source <env>/bin/activate
$ pip install -r requirements.txt
```
### Conda
```bash
$ conda create -n <env> pip
$ conda activate <env>
$ pip install -r requirements.txt
```

## Running a Simulated Screening
1. Ensure you are in the current working directory
2. Activate either your conda or venv environment with the installed dependencies
3. Run the `main.py` file with the required flags. An example with the default flags is shown below:
```bash
python3 main.py --data_file HCOF.csv --cost costs_HCOF.json --budget 1000 --nthreads 200 --initial_samples 200 --min_samples 20 --acquisition Thompson
```
This screen will produce two files both with the same unique identifier (uuid): 
1. `<uuid>.json` : plain text file in json format recording which flags were used for the screening
2. `<uuid>.csv` : csv file containing a history of the screen, showing the index of the material sampled, which test was conducted, the cost of the test and other relevant meta data

