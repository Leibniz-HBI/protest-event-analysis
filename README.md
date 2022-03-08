The project "Protest Event in Local News" automates information extraction for protest event analysis from newspaper articles.

This repository provides trained models and prediction scripts.

# Task A: Relevance Classification

A binary classification task whether ot not an articles conatains a relevant protest event description.

Model: `models/task-A/`
Training data state: Bremen, Dresden, Leipzig (2022-03-08)
Performance: ca. F1 = 93 % in-sample, 70.9 % on StZ data

Download the model:
```
cd models/task-A
wget https://pea-fgz-models.s3.eu-central-1.amazonaws.com/task-A/pytorch_model.bin
```

Prediction (on the DSC server add `CUDA_VISIBLE_DEVICES=0 srun` before the command):

```
python task-A_prediction.py --input_csv ~/unlabeled_data.csv --output_csv ~/predicted_data.csv
```

# Task B: Section identification

A binary sentence classification whether or not a sentence contains relevant protest event information. 
