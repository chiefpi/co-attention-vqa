# Co-Attention in VQA

This repository contains the code of CoNet, an end-to-end model for VQA.

## Setup

Download the mini [dataset](https://drive.google.com/open?id=1_VvBqqxPW_5HQxE6alZ7_-SGwbEt2_zn).

## Preprocess

Tokenize raw questions, and link image names, questions, and answers in a json file.

```bash
python preprocess.py
```

## Training

Set training parameters in `run.py`.

```bash
python run.py
```

## Evaluation

Use official scripts to evaluate results.

```bash
python eval.py
```

## Experiments

Results

