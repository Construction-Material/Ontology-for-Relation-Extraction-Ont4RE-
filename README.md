# Ontology-for-Relation-Extraction-Ont4RE

## Introduction
The code is for the paper "Ontology-based distant supervision for extracting entity-property relations in construction documents"

## Download data
Please visit our another repo [Construction-Dataset-CONSD](https://github.com/Construction-Material/Construction-Dataset-CONSD) and download the datasets into the directory `\benchmark` under the root file.
```
\benchmark
|-CONSD
|   |- train
|   |- val
|   |- test
|   |- rel2id
|   |- ...
|-CONSD_rule
|   |- ...
```
## How to Use
### Step 1 Installation
You need to install all the requirements
```
pip install -r requirements.txt
```

### Step 2 Get pre-trained models
Download the weights of pre-trained models from [Google Drive](https://drive.google.com/drive/folders/1f3cLgYV9SCjSTMJ4rcxNXFJF4rPsz8E6?usp=sharing), and put them into the file `\model` like this:

```
\model
|-casrel
|   |- config.json
|   |- pytorch_model.bin
|   |- ...
|-roberta
|-spert
|-...
```

### Step 3 Label sentence using ontology-based distant supervision
Execute the following command will process the sentences in `corpus.txt` and end up with an automatically labeled dataset `dataset.txt`.
The process takes some time because of the batch computation of semantic similarity between ontology classes and entities in the sentences
(please adjust the parameter ``BATCH_SIZE`` appropriately according to your gpu)
```
python similarity.py
```

### Step 4 Dataset preparation
Execute the following command to split the automatically labeled `dataset.txt` into `train\train.txt`, `val\val.txt` and `test\test.txt` using stratified sampling.
```
python split_dataset.py
```

### Step 5 Train and select the model
Specify the running mode at the command line. For cnn-based architecture models, specify the encoder `cnn`/`pcnn`, you can change the `--encoder` value, using `PCNN`:
```
python example/train_bag_cnn.py --bath_size 20 --bag_size 2 --max_epoch 40 --encoder pcnn --result pcnn
```
For transformer-based architecture models `bert`, `casrel` and others, you can use the following command, and change the `--pretrain_path` value to specify models,
using `BERT`:
```
python example/train_bag_bert.py --bath_size 20 --bag_size 2 --max_epoch 40 --pretrain_path model/bert-base-chinese --result bert-base-chinese
```
Or, you want to use another bert-variants like `RoBERTa`：
```
python example/train_bag_bert.py --bath_size 20 --bag_size 2 --max_epoch 40 --pretrain_path model/roberta --result roberta
```

The evaluation result will be saved in the folder `\result`.

## Change Dataset
Add the values of `--train_file`, `--val_file`, `--test_file`, and `--rel2id_file` in the training command. For training on the `\benchmark\CONSD_rule`:
```
python example/train_bag_bert.py --bath_size 20 --bag_size 2 --max_epoch 40 --pretrain_path model/roberta --result  --train_file benchmark/CONSD/train/train.txt
--val_file benchmark/CONSD/val/val.txt --test_file benchmark/CONSD/test/test.txt --rel2id_file benchmark/CONSD/rel2id/rel2id.json
```

## Citation
If you find our research is helpful to you, please considering for giving a star and citing our paper:

```
Junjie Jiang, Chengke Wu, Wenjie Sun, Yong He, Yuanjun Guo, Yang Su, Zhile Yang. Ontology-based distant supervision for extracting entity-property relations in construction documents.
```

## Acknowledgement
This repository is based on:

[OpenNRE](https://github.com/thunlp/OpenNRE) from THUNLP

[RoBERTa](https://github.com/facebookresearch/fairseq/tree/main/examples/roberta) from Meta Research

[CasRel](https://github.com/weizhepei/CasRel) from Jilin University

[SpERT](https://github.com/lavis-nlp/spert) from LAVIS - NLP Working Group

Thanks for their amazing work!

## Contact
Any question please contact [junj.chiang1102@gmail.com](mailto:junj.chiang1102@gmail.com).