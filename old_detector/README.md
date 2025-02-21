GPT-2 Output Detector
=====================

This directory contains the code for working with the GPT-2 output detector model, obtained by fine-tuning a
[RoBERTa model](https://ai.facebook.com/blog/roberta-an-optimized-method-for-pretraining-self-supervised-nlp-systems/)
with [the outputs of the 1.5B-parameter GPT-2 model](https://www.kaggle.com/datasets/abhishek/gpt2-output-data).

## Running the detector model


## Training a new detector model

You can use the provided training script to train a detector model on a new set of datasets.
We recommend using a GPU machine for this task.

```bash
# (on the top-level directory of this repository)
pip install -r requirements.txt
python -m detector.train
```

The training script supports a number of different options; append `--help` to the command above for usage.
