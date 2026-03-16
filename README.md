This repository contains the code for the comparative learning framework described in "Efficient Story Point Estimation With Comparative Learning".


## Dependencies
The required dependencies must be installed to run the source code.
```
pip install -r requirements.txt
```

## Data
Story point estimation data and its pre-split training, validation and testing splits can be found under Data/GPT2SP/Split/

## Comparative learning experiments
Once the dependencies are installed, run the corresponding files to run the comparative learning experiments with the default parameters. The parameters values can be changed to conduct different experiments.

For comparative experiments using the SBERT-Comparative model, run -
```
python SBERT_Comparative.py
python SBERT_Comparative_no_val.py
```

## Regression experiments
To run the SBERT-Regression model with default parameters, run -

```
python SBERT_Regression.py
```

## Baseline experiments
To run experiments with the replicated FastText-SVM model, run -

```
python FastTextSVM_Replication.py
```

To run experiments with the replicated LinearSVM-Comparative model, run -
```
python LinearSVM_Comparative.py
```

GPT2SP replication results were collected by running scripts publicly made available through the [GPT2SP repository](https://github.com/awsm-research/gpt2sp).


## Human-subject experiments data
Raw data from HSE-1 is available as a single .csv file under the Data/ directory.

Raw data from HSE-2 is available as a zip file under the Data/ directory. Unzip the file in that same directory for the raw data collected through the study.

## Human-subject experiments data analysis
To get the summarized results of HSE-1, run -

```
python HSE1.py
```

To get the summarized results of HSE-2, run -

```
python HSE2.py
```
