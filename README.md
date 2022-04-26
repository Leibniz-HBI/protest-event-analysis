The project "Protest Event in Local News" automates information extraction for protest event analysis from newspaper articles.

This repository provides trained models and prediction scripts.

The model files `pytorch_model.bin` and `optimizer.pt` are not staged to this repository. 
They can be shared via aws S3 buckets.
To upload a file use this command:

```
aws s3 cp models/task-A/pytorch_model.bin s3://pea-fgz-models/task-A/pytorch_model.bin
```

Caution: uploaded files to this bucket will be public!

# Task A: Relevance Classification

A binary classification task whether ot not an articles conatains a relevant protest event description.

Model: `models/task-A/`
Training data state: Bremen, Dresden, Leipzig, Stuttgart (2022-04-26)
Performance: ca. F1 = 91 % in-sample dev, mcc = 82 % in-sample dev

**Model Download**
```
cd models/task-A
wget https://pea-fgz-models.s3.eu-central-1.amazonaws.com/task-A/pytorch_model.bin
```

**Format Input Data**

As input, a CSV file with ',' as separator, double quote escape strategy and column header is expected. For prediction, the column named `text` is used as input for prediction. The output file is a copy of the input CSV with three additional columns: `pred_label`, `pred_score` and `pred_text`. The latter contains the text after preprocessing that served as input to the classification model.

**Run Prediction**

For prediction, return to the project root directory and run (on the DSC server add `CUDA_VISIBLE_DEVICES=0 srun` before the command):

```
python task-A_prediction.py --input_csv ~/unlabeled_data.csv --output_csv ~/predicted_data.csv
```

# Task B: Section identification

A binary sentence classification whether or not a sentence contains relevant protest event information. 
