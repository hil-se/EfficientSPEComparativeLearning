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

GPT2SP replication results were collected by running scripts publicly made available through the GPT2SP repository.

