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

# Task B and C: Section identification and protest variable spans

A sequence tagging task to predict sentences containing relevant information for protest events (Task B, tagset ["PROTEST", "NONE"] and protest variables (Task C, tagset [CLAIM, FORM, TRAEGER, ZAHL, DATUM, NONE]).

Both tasks are tagged separately. Thus, tokens not tagged as PROTEST still may, for example, be tagged as ZAHL.

Model: Flair MultiTask `models/task-B-C/best-model.pt`
Training data state: 607 gold documents from manual annotation (2022-04-01)
Performance: 
- ca. F1 = 71 % in-sample test for PROTEST tags (positive class), 80 % macro-F1
- ca. F1 = 62,5 % in-sample test for [CLAIM, FORM, TRAEGER, ZAHL, DATUM, NONE] (macro-avg)

**Model Download**
```
cd models/task-B-C
wget https://pea-fgz-models.s3.eu-central-1.amazonaws.com/task-B-C/pytorch_model.bin
```

**Format Input Data**

As input, a CSV file with ',' as separator, double quote escape strategy and column header is expected. For prediction, the column `doc_id` is used to keep alignment of tokens to documents, and the column `text` is used as input for prediction. It makes sense, to only input documents classified as relevant in Task A.

The output file is a CoNLL-like format with tokens in rows and their predictions in colums.

**Run Prediction**

For prediction, return to the project root directory and run (on the DSC server add `CUDA_VISIBLE_DEVICES=0 srun` before the command):

```
python task-B-C_prediction.py --input_csv ~/unlabeled_data.csv --output_csv ~/predicted_data.csv
```