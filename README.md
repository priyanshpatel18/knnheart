# Heart Disease Prediction using KNN

This project applies **K-Nearest Neighbors (KNN)** to heart disease datasets from the UCI repository.  
It includes data inspection, preprocessing, model training, hyperparameter tuning, and evaluation.

## Datasets
[Heart Disease Datasets](https://archive-beta.ics.uci.edu/dataset/45/heart+disease)

## Files
- `step2_inspect_v4.py` — Inspect raw Cleveland dataset  
- `step3_clean.py` — Clean and preprocess data  
- `step4_knn.py` — Train and evaluate baseline KNN  
- `step5_bestk.py` — Find best k using cross-validation  
- `step6_final_knn.py` — Final KNN with optimal k  
- `step7_run_knn_all.py` — Run experiments on all datasets (V1, V3, V4)  

## Datasets
The following CSV files are used:
- `data/heart_v1.csv`  
- `data/heart_v3.csv`  
- `data/heart_v4.csv`  

## Installation
```bash
pip install -r requirements.txt
```

## Running

```
python3 step6_final_knn.py
```